# import glob
# import torch
# import torch.nn as nn
# import torchvision
# from torchvision.models import resnet50
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#
# from detection.lora import LoRALinear
#
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11)
# for c in glob.glob('checkpoints/DIOR-base-*.pth'):
#     ckpt = torch.load(c, map_location='cpu', weights_only=True)
#     model.load_state_dict(ckpt['model'])
#     torch.save(model.state_dict(), c)
#
# class FusedFRCNN(torchvision.models.detection.FasterRCNN):
#     def __init__(self):
#         pass
#
# a = FusedFRCNN()
