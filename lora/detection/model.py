import os
import copy
import math
import warnings
from glob import glob
from safetensors.torch import load_file

import torch
import torch.nn as nn
from torch.nn.functional import dropout
from torchvision.models import resnet
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FastRCNNConvFCHead, FasterRCNN


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=1, dropout=0):
        super().__init__()
        
        self.scaling = alpha / math.sqrt(rank)

        self.linear = copy.deepcopy(linear_layer)
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else lambda x: x
        
        # LoRA Matrix
        self.lora_A = nn.Parameter(torch.zeros(self.linear.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.linear.out_features))
        
        # weight init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling

        return base_out + lora_out


class LoRAConv2d(nn.Module):
    def __init__(self, conv2d_layer, rank=64, alpha=1, dropout=0):
        super().__init__()

        self.scaling = alpha / math.sqrt(rank)

        self.conv = copy.deepcopy(conv2d_layer)
        for param in self.conv.parameters():
            param.requires_grad = False

        # dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else lambda x: x
        
        kernel_size = self.conv.kernel_size[0]
        in_features = self.conv.in_channels * kernel_size
        rank = rank * kernel_size
        out_features = self.conv.out_channels * self.conv.groups * kernel_size

        # LoRA Matrix
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # weight init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.conv._conv_forward(
            x,
            self.conv.weight + (self.lora_A @ self.lora_B).view(self.conv.weight.shape) * self.scaling,
            self.conv.bias
        )


def create_model(backbone: str, num_classes: int):
    '''
    create frcnn model with ResNet50 or 101 w/ FPN as backbone

    Args:
        num_classes (int): number of dataset classes, excluding background
        backbone (str): resnet50 or resnet101
    '''
    # TODO：for specific functions
    assert backbone in ['resnet50', 'resnet101']

    backbone = resnet.__dict__[backbone](weights='DEFAULT', norm_layer=nn.BatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_layers=3, norm_layer=nn.BatchNorm2d)

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


def load_weights(weight_path: str):
    '''
    Args:
        weight_path (str): abs path of weight file
    '''
    assert os.path.exists(weight_path)
    filename = os.path.basename(weight_path)
    backbone_type = f"resnet{filename.split('_')[1]}"
    # TODO: phase -2, -3
    phase = filename.split('_')[-2]
    assert backbone_type in ['resnet50', 'resnet101']
    assert phase in ['joint', 'base', 'inc']

    weight_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
    weight_dict = weight_dict['model'] if 'model' in weight_dict else weight_dict

    num_classes = weight_dict['roi_heads.box_predictor.cls_score.bias'].size(0) - 1
    model = create_model(backbone_type, num_classes)
    model.load_state_dict(weight_dict)

    return model, phase


def convert_to_lora(model: nn.Module):
    '''
    recursively traverse the model architecture 
    and replace all nn.Linear and nn.Conv2d layer with LoRALayer
    '''
    for name, child in model.named_children():
        if name not in ['bbox_pred', 'cls_logits']:
            if len(list(child.children())) > 0:
                convert_to_lora(child)
            if isinstance(child, nn.Linear):
                setattr(model, name, LoRALinear(child))         # same as model._modules[name] = LoRALinear(child)
            if isinstance(child, nn.Conv2d):
                setattr(model, name, LoRAConv2d(child))


def expand_classifier(source: nn.Module, target: nn.Module):
    '''
    classifier branch expansion

    Returns:
        target model with expanded classifier
    '''
    if not isinstance(source, target):
        warnings.warn(f"source is {type(source)} while target is {type(target)}!")
    assert 'roi_heads.box_predictor.cls_score' in source.state_dict().keys()
    assert 'roi_heads.box_predictor.cls_score' in target.state_dict().keys()

    cls_weight_list = [model.state_dict()['roi_heads.box_predictor.cls_score.weight'][1:] for model in [source, target]]
    cls_bias_list = [model.state_dict()['roi_heads.box_predictor.cls_score.bias'][1:] for model in [source, target]]
    box_weight_list = [model.state_dict()['roi_heads.box_predictor.bbox_pred.weight'][4:] for model in [source, target]]
    box_bias_list = [model.state_dict()['roi_heads.box_predictor.bbox_pred.bias'][4:] for model in [source, target]]

    cls_weight, cls_bias = torch.cat(cls_weight_list, dim=0), torch.cat(cls_bias_list, dim=0)
    box_weight, box_bias = torch.cat(box_weight_list, dim=0), torch.cat(box_bias_list, dim=0)

    in_features = target.roi_heads.box_predictor.cls_score.in_features
    num_classes = box_bias.size(0) + cls_bias.size(0)
    target.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    target.state_dict()['roi_heads.box_predictor.cls_score.weight'] = cls_weight
    target.state_dict()['roi_heads.box_predictor.cls_score.bias'] = cls_bias
    target.state_dict()['roi_heads.box_predictor.bbox_pred.weight'] = box_weight
    target.state_dict()['roi_heads.box_predictor.bbox_pred.bias'] = box_bias

    return target


def freeze_module(model: nn.Module, tune_list: list[str]):
    '''
    freeze modules params except for those in tune_list
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
    convert .safetensors file to .pth
    recursively deal with all .safetensors files under root diretory

    Args:
        root (str):  directory where the safetensors files locate
    '''
    assert os.path.isdir(root)
    sft_list = []
    for path, _, file_list in os.walk(root):
        sft_list += [os.path.join(path, file_name) for file_name in file_list if '.safetensors' in file_name]
    for filename in sft_list:
        ckpt = load_file(filename)
        torch.save(ckpt, filename.replace(".safetensors", ".pth"))


def extract_state_dict(root: str):
    '''
    extract model state_dict from .pth file
    recursively deal with all .pth files under root diretory

    Args:
        root (str):  diretory under which .pth files are saved
    '''
    assert os.path.isdir(root)
    pth_list = []
    for path, _, file_lst in os.walk(root):
        pth_list += [os.path.join(path, file_name) for file_name in file_lst if '.pth' in file_name]
    for filename in pth_list:
        ckpt = torch.load(filename, map_location='cpu', weights_only=True)
        if 'model' in ckpt:
            torch.save(ckpt['model'], filename)


if __name__ == '__main__':
    root = '../checkpoints'
    extract_state_dict(root)
    safetensors_to_pth(root)