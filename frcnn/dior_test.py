import os
import json
import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detection import transform_
from dataset import DIORDataset
from utils.train_eval_utils import evaluate, inference

torch.multiprocessing.set_sharing_strategy('file_system')

with open('config.json') as f:
    config = json.load(f)["DIOR"]


def load_weights(weight_path, num_classes=21):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    weights_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    weights_dict = weights_dict['model'] if 'model' in weights_dict else weights_dict
    model.load_state_dict(weights_dict)

    return model


def main(args):
    root = config['root']
    mean, std = config['mean'], config['std']
    batch_size = config['test_batchSize']
    model_savedir = config['savedir']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = load_weights(os.path.join(model_savedir, args.model_file)).to(device)

    data_transform = transform_.Compose([
        transform_.ToTensor(),
        transform_.Normalize(mean, std)
    ])
    dataset = DIORDataset(root, transform=data_transform, mode='test')

    if args.test_mode == 'test':
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=20, shuffle=False,
                                collate_fn=DIORDataset.collate_fn)

        evaluate(model=model, dataloader=dataloader, device=device)

    elif args.test_mode == 'inference':
        srcdir = os.path.join(root, 'Images')
        dataloader = DataLoader(dataset, batch_size=5, num_workers=20, shuffle=True,
                                collate_fn=DIORDataset.collate_fn)
        inference(model=model,
                  dataloader=dataloader,
                  threshold=0.5,
                  device=device,
                  srcdir=srcdir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test_mode', default='test', help='test or inference')
    parser.add_argument('--model_file', default='DIOR-41.4-62.3.pth', help='name of weight file to load with')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(os.path.join(config['savedir'], args.model_file)):
        raise ValueError('指定的模型权重不存在！')

    main(args)

