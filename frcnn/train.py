import os
import json
import argparse

import torch
import torchvision
import timm.scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import DIORDataset
from detection import transform_
from utils.train_eval_utils import train_one_epoch, evaluate

torch.multiprocessing.set_sharing_strategy('file_system')

with open('config.json') as f:
    config = json.load(f)["DIOR"]

def create_model(num_classes=21):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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

    train_dataset = DIORDataset(root=root, transform=data_transform['train'], mode='train')
    test_dataset = DIORDataset(root=root, transform=data_transform['test'], mode='test')
    dataloader = {
        'train': DataLoader(dataset=train_dataset, batch_size=train_batchSize,
                            num_workers=20, shuffle=True, collate_fn=train_dataset.collate_fn),
        'test': DataLoader(dataset=test_dataset, batch_size=test_batchSize,
                           num_workers=20, collate_fn=test_dataset.collate_fn)
    }

    model = create_model().to(device)

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

        coco_info = evaluate(model=model,
                             dataloader=dataloader['test'],
                             device=device,
                             print_feq=print_feq)
        mAP = coco_info[0]
        test_map.append(mAP)

        # add tensorboard records
        tb_logger.add_scalar('loss', loss, epoch)
        tb_logger.add_scalar('lr', lr, epoch)
        tb_logger.add_scalar('mAP', mAP, epoch)
        for k,v in loss_dict.items():
            tb_logger.add_scalar(k, v, epoch)

        # save weights
        if mAP == sorted(test_map)[-1] and epoch > warmup_epoch:
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(save_files,
                       os.path.join(model_savedir, "DIOR-{}-{:.3f}.pth".format(epoch, mAP * 100)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', default='', help='training state to resume training with')
    args = parser.parse_args()
    print(args)

    main(args)

