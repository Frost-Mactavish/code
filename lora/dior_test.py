import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

from detection import transform_
from dataset import DIORIncDataset
from detection.model import create_model
from utils.train_eval_utils import evaluate, inference
from detection.coco_utils import get_coco_api_from_dataset

torch.multiprocessing.set_sharing_strategy('file_system')


def load_weights(weight_path: str):
    '''
    Args:
        weight_path (str): abs path of weight file
    '''
    assert os.path.exists(weight_path)
    filename = os.path.basename(weight_path)
    backbone_type = f"resnet{filename.split('_')[1]}"
    phase = filename.split('_')[-2]
    assert backbone_type in ['resnet50', 'resnet101']
    assert phase in ['joint', 'base', 'inc']
    
    weight_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    weight_dict = weight_dict['model'] if 'model' in weight_dict else weight_dict

    num_classes = weight_dict['roi_heads.box_predictor.cls_score.bias'].size(0) - 1
    model = create_model(backbone_type, num_classes)
    model.load_state_dict(weight_dict)

    return model, phase


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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model, phase = load_weights(weight_path)
    model.to(device)

    data_transform = transform_.Compose([
        transform_.ToTensor(),
        transform_.Normalize(mean, std)
    ])
    test_dataset = DIORIncDataset(root, transform=data_transform, mode='test', phase=phase)

    if args.test_mode == 'map':
        coco_gt = get_coco_api_from_dataset(test_dataset)
        dataloader = DataLoader(test_dataset, batch_size=test_batchSize, num_workers=8,
                                shuffle=False, collate_fn=test_dataset.collate_fn)
        evaluate(model=model,
                 dataloader=dataloader,
                 coco_gt=coco_gt,
                 device=device,
                 phase=phase,
                 print_feq=print_feq)

    elif args.test_mode == 'visualize':
        srcdir = os.path.join(root, 'Images')
        dataloader = DataLoader(test_dataset, batch_size=5, num_workers=8,
                                shuffle=True, collate_fn=test_dataset.collate_fn)
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

