import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

from detection import transform_
from dataset import DIORIncDataset
from detection.model import load_weights
from utils.train_eval_utils import evaluate, inference
from detection.coco_utils import get_coco_api_from_dataset


def main(args):
    with open('config.json') as f:
        config = json.load(f)[args.dataset]
        
    test_batchSize = config['test_batchSize']
    mean, std = config['mean'], config['std']
    root = config['root']
    save_dir = config['save_dir']
    print_feq = config['print_feq']

    weight_path = os.path.join(save_dir, args.filename)
    assert os.path.exists(weight_path)

    model, phase = load_weights(weight_path)
    model.to(device)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transform = transform_.Compose([
        transform_.ToTensor(),
        transform_.Normalize(mean, std)
    ])
    test_dataset = DIORIncDataset(root, transform=data_transform, mode='test', phase=phase)
    dataloader = DataLoader(test_dataset, batch_size=test_batchSize, num_workers=8,
                            pin_memory=True, collate_fn=test_dataset.collate_fn)

    if args.test_mode == 'map':
        coco_gt = get_coco_api_from_dataset(test_dataset)
        evaluate(model=model,
                 dataloader=dataloader,
                 coco_gt=coco_gt,
                 device=device,
                 print_feq=print_feq)

    elif args.test_mode == 'visualize':
        srcdir = os.path.join(root, 'Images')
        inference(model=model,
                  dataloader=dataloader,
                  threshold=0.5,
                  device=device,
                  srcdir=srcdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='DIOR', help='dataset name')
    parser.add_argument('--test_mode', default='map',
                        help='get map over testset or visualize detection results')
    parser.add_argument('--filename', default='DIOR_50_joint_72.68.pth',
                        help='filename of weight')
    args = parser.parse_args()
    print(args)

    main(args)

