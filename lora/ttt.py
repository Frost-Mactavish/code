import os

import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
# in_features = model.roi_heads.box_predictor.cls_score.in_features
#
# weight = torch.load('checkpoints/tmp/DIOR-full-12-51.173.pth', map_location='cpu', weights_only=True)
# num_class = weight['roi_heads.box_predictor.cls_score.bias'].size(0) - 1
#
# cls_prev = (weight['roi_heads.box_predictor.cls_score.weight'], weight['roi_heads.box_predictor.cls_score.bias'])
# bbox_prev = (weight['roi_heads.box_predictor.bbox_pred.weight'], weight['roi_heads.box_predictor.bbox_pred.bias'])
# cls_next = (torch.zeros_like(cls_prev[0][1:]), torch.zeros_like(cls_prev[1][1:]))
# bbox_next = (torch.zeros_like(bbox_prev[0][4:]), torch.zeros_like(bbox_prev[1][4:]))
#
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class * 2 + 1)
#
# for name, _ in model.named_parameters():
#     if 'roi_heads.box_predictor' not in name:
#         model.state_dict()[name] = weight[name]
#
# model.state_dict()['roi_heads.box_predictor.cls_score.weight'][:num_class+1] = cls_prev[0]
# model.state_dict()['roi_heads.box_predictor.cls_score.bias'][:num_class+1] = cls_prev[1]
#
# model.state_dict()['roi_heads.box_predictor.cls_score.weight'][num_class+1:] = cls_next[0]
# model.state_dict()['roi_heads.box_predictor.cls_score.bias'][num_class+1:] = cls_next[1]
#
# model.state_dict()['roi_heads.box_predictor.bbox_pred.weight'][:4*(num_class+1)] = bbox_prev[0]
# model.state_dict()['roi_heads.box_predictor.bbox_pred.bias'][:4*(num_class+1)] = bbox_prev[1]
#
# model.state_dict()['roi_heads.box_predictor.bbox_pred.weight'][4*(num_class+1):] = bbox_next[0]
# model.state_dict()['roi_heads.box_predictor.bbox_pred.bias'][4*(num_class+1):] = bbox_next[1]
#
# torch.save(model.state_dict(), 'checkpoints/full_zero.pth')


#
# part_dict = {
#     'head': ['DIOR-base-10-50.087.pth', 'DIOR-head-15-29.750.pth', 'head_dareties.pth'],
#     'head_rpn': ['DIOR-base-10-50.087.pth', 'DIOR-head_rpn-12-39.863.pth', 'head_rpn_dareties.pth'],
#     'head_rpn_fpn': ['DIOR-base-10-50.087.pth', 'DIOR-head_rpn_fpn-14-46.364.pth', 'head_rpn_fpn_dareties.pth'],
#     'backbone4': ['DIOR-base-10-50.087.pth', 'DIOR-backbone4-10-48.1618-73.2477.pth', 'backbone4_dareties.pth'],
#     'backbone42': ['DIOR-base-10-50.087.pth', 'DIOR-backbone42-15-47.1424-72.6112.pth', 'backbone42_dareties.pth'],
#     'full': ['DIOR-base-10-50.087.pth', 'DIOR-full-12-51.173.pth', 'full_dareties.pth'],
# }
#
# root = 'checkpoints'
# merge = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2()
# in_features = merge.roi_heads.box_predictor.cls_score.in_features
#
# for name, model_list in part_dict.items():
#     save_path = os.path.join(root, f'{name}.pth')
#     file_list = [os.path.join(root, filename) for filename in model_list]
#
#     base_weight = torch.load(file_list[0], map_location='cpu', weights_only=True)
#     inc_weight = torch.load(file_list[1], map_location='cpu', weights_only=True)
#     merge_weight = torch.load(file_list[2], map_location='cpu', weights_only=True)
#
#     bg_cls_weight = merge_weight['roi_heads.box_predictor.cls_score.weight'][0]
#     bg_cls_bias = merge_weight['roi_heads.box_predictor.cls_score.bias'][0]
#     bg_bbox_weight = merge_weight['roi_heads.box_predictor.bbox_pred.weight'][:4]
#     bg_bbox_bias = merge_weight['roi_heads.box_predictor.bbox_pred.bias'][:4]
#
#     base_cls_weight = base_weight['roi_heads.box_predictor.cls_score.weight'][1:]
#     base_cls_bias = base_weight['roi_heads.box_predictor.cls_score.bias'][1:]
#     base_bbox_weight = base_weight['roi_heads.box_predictor.bbox_pred.weight'][4:]
#     base_bbox_bias = base_weight['roi_heads.box_predictor.bbox_pred.bias'][4:]
#
#     inc_cls_weight = inc_weight['roi_heads.box_predictor.cls_score.weight'][1:]
#     inc_cls_bias = inc_weight['roi_heads.box_predictor.cls_score.bias'][1:]
#     inc_bbox_weight = inc_weight['roi_heads.box_predictor.bbox_pred.weight'][4:]
#     inc_bbox_bias = inc_weight['roi_heads.box_predictor.bbox_pred.bias'][4:]
#
#     base_class_num = base_cls_bias.size(0)
#     inc_class_num = inc_cls_bias.size(0)
#     merge_class_num = base_class_num + inc_class_num
#     merge.roi_heads.box_predictor = FastRCNNPredictor(in_features, merge_class_num + 1)
#
#     cls_weight = torch.cat((base_cls_weight, inc_cls_weight), dim=0)
#     cls_bias = torch.cat((base_cls_bias, inc_cls_bias), dim=0)
#     bbox_weight = torch.cat((base_bbox_weight, inc_bbox_weight), dim=0)
#     bbox_bias = torch.cat((base_bbox_bias, inc_bbox_bias), dim=0)
#
#     merge.state_dict()['roi_heads.box_predictor.cls_score.weight'][0] = bg_cls_weight
#     merge.state_dict()['roi_heads.box_predictor.cls_score.bias'][0] = bg_cls_bias
#     merge.state_dict()['roi_heads.box_predictor.bbox_pred.weight'][:4] = bg_bbox_weight
#     merge.state_dict()['roi_heads.box_predictor.bbox_pred.bias'][:4] = bg_bbox_bias
#
#     merge.state_dict()['roi_heads.box_predictor.cls_score.weight'][1:] = cls_weight
#     merge.state_dict()['roi_heads.box_predictor.cls_score.bias'][1:] = cls_bias
#     merge.state_dict()['roi_heads.box_predictor.bbox_pred.weight'][4:] = bbox_weight
#     merge.state_dict()['roi_heads.box_predictor.bbox_pred.bias'][4:] = bbox_bias
#
#     torch.save(merge.state_dict(), save_path)
