import os
import json
import argparse

import torch
import torchvision
import timm.scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import DIORIncDataset
from detection import transform_
from utils.train_eval_utils import train_one_epoch, evaluate

torch.multiprocessing.set_sharing_strategy('file_system')

with open('config.json') as f:
    config = json.load(f)["DIOR"]


def create_model(num_classes: int = 11):
    '''
    create frcnn model with ResNet50 w/ FPN as backbone

    Args:
        num_classes (int): number of classes of dataset
    '''
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load('checkpoints/DIOR-base-10-50.087.pth', map_location=torch.device('cpu'), weights_only=True))

    return model


def main(args):
    root = config['root']
    mean, std = config['mean'], config['std']
    train_batchSize = config['train_batchSize']
    test_batchSize = config['test_batchSize']
    total_epoch = config['epoch']
    model_savedir = config['savedir']
    print_feq = config['print_feq']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tb_logger = SummaryWriter(log_dir='./tb_logger/', flush_secs=60)

    data_transform = {
        'train': transform_.Compose([
            transform_.ToTensor(),
            transform_.Normalize(mean, std),
            transform_.RandomHorizontalFlip()
        ]),
        'test': transform_.Compose([
            transform_.ToTensor(),
            transform_.Normalize(mean, std)
        ])
    }

    train_dataset = DIORIncDataset(root=root, transform=data_transform['train'], mode='train', phase=args.phase)
    test_dataset = DIORIncDataset(root=root, transform=data_transform['test'], mode='test', phase=args.phase)
    dataloader = {
        'train': DataLoader(dataset=train_dataset, batch_size=train_batchSize,
                            num_workers=20, shuffle=True, collate_fn=DIORIncDataset.collate_fn),
        'test': DataLoader(dataset=test_dataset, batch_size=test_batchSize,
                           num_workers=20, collate_fn=DIORIncDataset.collate_fn)
    }

    model = create_model().to(device)

    finetune_list = ['roi_heads', 'rpn', 'backbone.fpn', 'backbone.body.layer4' ]
    if args.mode == 'finetune':
        for name, param in model.named_parameters():
            cnt = 0
            for string in finetune_list:
                if string not in name:
                    cnt += 1
            if cnt == len(finetune_list):
                param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)

    warmup_epoch = 5
    lr_scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=total_epoch, lr_min=1e-6,
                                                    warmup_t=warmup_epoch, warmup_lr_init=1e-4)

    start_epoch = 1
    if args.resume != '':
        resume = os.path.join(model_savedir, args.resume)
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    test_map = []
    for epoch in range(start_epoch, total_epoch + 1):
        loss, loss_dict, lr = train_one_epoch(model=model,
                                              optimizer=optimizer,
                                              dataloader=dataloader['train'],
                                              device=device,
                                              epoch=epoch,
                                              print_freq=print_feq)
        lr_scheduler.step(epoch)

        if epoch > warmup_epoch:
            # coco_info (list): mAP@[0.5:0,95], mAP@0.5, mAP@0.75, ...
            coco_info = evaluate(model=model,
                                 dataloader=dataloader['test'],
                                 device=device,
                                 phase=args.phase,
                                 print_feq=print_feq)
            mAP50, mAP = coco_info[1], coco_info[0]
            test_map.append(mAP)

            # add tensorboard records
            tb_logger.add_scalar('loss', loss, epoch)
            tb_logger.add_scalar('lr', lr, epoch)
            tb_logger.add_scalar('mAP@[0.5:0.95]', mAP, epoch)
            tb_logger.add_scalar('mAP@0.5', mAP50, epoch)
            for k,v in loss_dict.items():
                tb_logger.add_scalar(k, v, epoch)

            # save weights
            if mAP == sorted(test_map)[-1]:
                save_files = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }
                torch.save(save_files,
                           os.path.join(model_savedir, f"DIOR-{args.mode}-{epoch}-{mAP*100:.4f}-{mAP50*100:.4f}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', default='', help='training state to resume training with')
    parser.add_argument('--phase', default='inc', help='incremental phase')
    parser.add_argument('--mode', default='finetune', help='incremental phase')

    args = parser.parse_args()
    print(args)

    main(args)

