import os
import copy
import math
from glob import glob
from safetensors.torch import load_file

import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor, FasterRCNN,
                                                      FastRCNNConvFCHead, fasterrcnn_resnet50_fpn_v2)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def create_model(num_classes: int , backbone: str):
    '''
    create frcnn model with ResNet50 w/ FPN as backbone

    Args:
        num_classes (int): number of classes of dataset
        backbone (str): resnet50 or resnet101
    '''
    if backbone == 'resnet50':
        model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    elif backbone == 'resnet101':
        backbone = resnet_fpn_backbone(backbone_name='resnet101', weights='DEFAULT', norm_layer=nn.BatchNorm2d)

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
        box_head = FastRCNNConvFCHead(
            (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
        )

        model = FasterRCNN(
            backbone,
            num_classes=91,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    return model


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=1):
        super().__init__()
        self.base_layer = copy.deepcopy(linear_layer)
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora(x)
        return self.base_layer(x) + self.lora(x)


class LoRAConv2d(nn.Module):
    def __init__(self, conv2d_layer, rank=4, alpha=1):
        pass

    def forward(self, x):
        pass


def convert_to_lora(model: nn.Module):
    '''
    recursively search the model architecture and
    replace all nn.Linear and nn.Conv2d layer with corresponding LoRALayer

    Args:
        model (nn.Module): model to be converted
    '''
    for name, child in model.named_children():
        if len(list(child.children())) > 0:
            convert_to_lora(child)
        if isinstance(child, nn.Linear):
            setattr(model, name, LoRALinear(child))         # same as model._modules[name] = LoRALinear(child)
        if isinstance(child, nn.Conv2d):
            setattr(model, name, LoRAConv2d(child))


def expand_classifier(ckpt_list: list[str]):
    '''
    merge classification head of multiple detection models

    Args:
        ckpt_list (list[str]): state_dict of detection ckpts, assembled in list

    Returns:
        classifier weight and bias concatenated in dim=0, assembled in list
    '''
    cls_weight_list = [ckpt.state_dict()['roi_heads.box_predictor.cls_score.weight'] for ckpt in ckpt_list]
    cls_bias_list = [ckpt.state_dict()['roi_heads.box_predictor.cls_score.bias'] for ckpt in ckpt_list]
    return torch.cat(cls_weight_list, dim=0), torch.cat(cls_bias_list, dim=0)


def freeze_module(tune_list: list[str], model: nn.Module):
    '''
    Args:
        tune_list: (list[str]): list of unfreezed modules
        model (nn.Module): model to be freezed
    '''
    for name, param in model.named_parameters():
        cnt = 0
        for module_name in tune_list:
            if module_name not in name:
                cnt += 1
        if cnt == len(tune_list):
            param.requires_grad = False


def safetensors_to_pth(root: str):
    '''
    convert safetensors format to pth format
    it will deal with all safetensors files in the specified target_dir

    Args:
        root (str):  directory where the safetensors files locate
    '''
    assert os.path.isdir(root)
    for filename in glob(f"{root}/*.safetensors"):
        ckpt = load_file(filename)
        torch.save(ckpt, filename.replace(".safetensors", ".pth"))


def extract_state_dict(root: str):
    '''
    extract model state_dict from .pth files saved during training
    dealing with all .pth file under root diretory

    Args:
        root (str):  diretory under which .pth files are saved
    '''
    assert os.path.isdir(root)
    for filename in glob(f"{root}/*.pth"):
        ckpt = torch.load(filename, map_location='cpu', weights_only=True)
        if 'model' in ckpt:
            torch.save(ckpt['model'], filename)


if __name__ == '__main__':
    root = 'checkpoints'
    extract_state_dict(root)
    safetensors_to_pth(root)