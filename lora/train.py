import os
import json
import argparse
from re import findall
from timm import scheduler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from detection import transform_
from detection.model import create_model, freeze_module
from dataset import DIORIncDataset
from detection.coco_utils import get_coco_api_from_dataset
from utils.train_eval_utils import train_one_epoch, evaluate

torch.multiprocessing.set_sharing_strategy('file_system')


def main(args, tune_list):
    with open('config.json') as f:
        config = json.load(f)[args.dataset]

    total_epoch = config['epoch']
    warmup_epoch = config['warmup_epoch']
    train_batchSize = config['train_batchSize']
    test_batchSize = config['test_batchSize']
    mean, std = config['mean'], config['std']
    root = config['root']
    save_dir = config['save_dir']
    print_feq = config['print_feq']

    tb_logger = SummaryWriter(log_dir='tb_logger', flush_secs=60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    # TODO: DOTA dataset
    train_dataset = DIORIncDataset(root=root, transform=data_transform['train'], mode='train', phase=args.phase)
    test_dataset = DIORIncDataset(root=root, transform=data_transform['test'], mode='test', phase=args.phase)
    dataloader = {
        'train': DataLoader(dataset=train_dataset, batch_size=train_batchSize,
                            num_workers=20, shuffle=True, collate_fn=train_dataset.collate_fn),
        'test': DataLoader(dataset=test_dataset, batch_size=test_batchSize,
                           num_workers=20, collate_fn=test_dataset.collate_fn)
    }

    num_classes = len(train_dataset.class_dict)
    model = create_model(args.backbone, num_classes + 1).to(device)

    if args.partial is not None:
        freeze_module(model, tune_list)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = scheduler.CosineLRScheduler(optimizer, t_initial=total_epoch, lr_min=1e-6,
                                                    warmup_t=warmup_epoch, warmup_lr_init=1e-4)

    start_epoch = 1
    if args.resume != '':
        resume = os.path.join(save_dir, args.resume)
        checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    test_map = []
    # initialize coco_gt for test
    coco_gt = get_coco_api_from_dataset(test_dataset)
    for epoch in range(start_epoch, total_epoch + 1):
        loss, loss_dict, lr = train_one_epoch(model=model,
                                              optimizer=optimizer,
                                              dataloader=dataloader['train'],
                                              device=device,
                                              epoch=epoch,
                                              print_freq=print_feq)
        lr_scheduler.step(epoch)

        # add tensorboard records
        tb_logger.add_scalar('loss', loss, epoch)
        tb_logger.add_scalar('lr', lr, epoch)
        for k, v in loss_dict.items():
            tb_logger.add_scalar(k, v, epoch)

        if epoch > warmup_epoch:
            # coco_info (list): mAP@[0.5:0,95], mAP@0.5, mAP@0.75, ...
            coco_info = evaluate(model=model,
                                 dataloader=dataloader['test'],
                                 coco_gt=coco_gt,
                                 device=device,
                                 phase=args.phase,
                                 print_feq=print_feq)
            mAP50, mAP = coco_info[1], coco_info[0]

            # add tensorboard records
            tb_logger.add_scalar('mAP@[0.5:0.95]', mAP, epoch)
            tb_logger.add_scalar('mAP@0.5', mAP50, epoch)
            
            test_map.append(mAP50)
            # save weights
            if mAP50 == sorted(test_map)[-1]:
                save_file = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }
                backbone = findall(r"\d+", args.backbone)[0]
                filename = f"{args.dataset}_{backbone}_{args.phase}_{mAP50*100:.2f}.pth"
                torch.save(save_file, os.path.join(save_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='DIOR', help='dataset name')
    parser.add_argument('--backbone', default='resnet101', help='model backbone')
    parser.add_argument('--phase', default='joint', help='incremental phase')
    parser.add_argument('--partial', default=None, help='train part of the model')
    parser.add_argument('--resume', default='', help='training state to resume training with')
    args = parser.parse_args()
    print(args)

    tune_list = ['roi_heads', 'rpn', 'backbone.fpn', 'backbone.body.layer4.2' ]

    main(args, tune_list)

