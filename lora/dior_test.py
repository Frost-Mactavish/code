import os
import json
import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detection import transform_
from dataset import DIORIncDataset
from utils.train_eval_utils import evaluate, inference
torch.multiprocessing.set_sharing_strategy('file_system')

with open('config.json') as f:
    config = json.load(f)["DIOR"]

def load_weights(weight_path: str):
    '''
    Args:
        weight_path (str): abs path to the weight

    Returns:
        model with state dict of specified weight
    '''
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()

    weight_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    weight_dict = weight_dict['model'] if 'model' in weight_dict else weight_dict

    num_classes = weight_dict['roi_heads.box_predictor.cls_score.bias'].size(0)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(weight_dict)

    return model

# def load_weights():
#     base = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
#     inc1 = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
#     inc2 = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
#
#     in_features = base.roi_heads.box_predictor.cls_score.in_features
#     inc1.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11)
#     inc2.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11)
#     inc1.load_state_dict(torch.load('checkpoints/DIOR-base-10-50.087.pth',map_location='cpu',weights_only=True))
#     inc2.load_state_dict(torch.load('checkpoints/DIOR-inc-12-49.855.pth',map_location='cpu',weights_only=True))
#
#     inc1_cls_weight = inc1.state_dict()['roi_heads.box_predictor.cls_score.weight']
#     inc1_cls_bias = inc1.state_dict()['roi_heads.box_predictor.cls_score.bias']
#     inc2_cls_weight = inc2.state_dict()['roi_heads.box_predictor.cls_score.weight']
#     inc2_cls_bias = inc2.state_dict()['roi_heads.box_predictor.cls_score.bias']
#
#     inc1_cls_weight[0, :] = (inc1_cls_weight[0, :] + inc2_cls_weight[0, :]) / 2
#     inc2_cls_weight = inc2_cls_weight[1:, :]
#     inc1_cls_bias[0] = (inc1_cls_bias[0] + inc2_cls_bias[0]) / 2
#     inc2_cls_bias = inc2_cls_bias[1:]
#     cls_weight_list = [inc1_cls_weight, inc2_cls_weight]
#     cls_bias_list = [inc1_cls_bias, inc2_cls_bias]
#     cls_weight, cls_bias = torch.cat(cls_weight_list, dim=0), torch.cat(cls_bias_list, dim=0)
#
#     base.load_state_dict(torch.load('checkpoints/model.pth'))
#     base.load_state_dict(torch.load('checkpoints/model.pth', map_location='cpu', weights_only=True))
#     base.roi_heads.box_predictor = FastRCNNPredictor(in_features, 21)
#     base.state_dict()['roi_heads.box_predictor.cls_score.weight'] = cls_weight
#     base.state_dict()['roi_heads.box_predictor.cls_score.bias'] = cls_bias
#
#     return base


def main(args):
    root = config['root']
    mean, std = config['mean'], config['std']
    batch_size = config['test_batchSize']
    model_savedir = config['savedir']
    weight_path = os.path.join(model_savedir, args.weight_filename)

    if not os.path.exists(weight_path):
        raise ValueError('指定的模型权重不存在！')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = load_weights(weight_path).to(device)

    data_transform = transform_.Compose([
        transform_.ToTensor(),
        transform_.Normalize(mean, std)
    ])
    dataset = DIORIncDataset(root, transform=data_transform, mode='test', phase=args.phase)

    if args.test_mode == 'test':
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=20, shuffle=False,
                                collate_fn=DIORIncDataset.collate_fn)

        evaluate(model=model, dataloader=dataloader, device=device, phase=args.phase, print_feq=250)

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
    parser.add_argument('--weight_filename', default='DIOR-base-10-50.087.pth', help='filename of weight')
    parser.add_argument('--phase', default='base', help='base, inc or joint test')

    args = parser.parse_args()
    print(args)

    main(args)

