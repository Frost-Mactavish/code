import copy
import glob
import math

import torch
import torch.nn as nn

from safetensors.torch import load_file


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


def expand_classifier(ckpt_list: list):
    '''
    merge classification head of multiple detection models

    Args:
        ckpt_list (list): state_dict of detection ckpts, assembled in list

    Returns:
        classifier weight and bias concatenated in dim=0, assembled in list
    '''
    cls_weight_list = [ckpt.state_dict()['roi_heads.box_predictor.cls_score.weight'] for ckpt in ckpt_list]
    cls_bias_list = [ckpt.state_dict()['roi_heads.box_predictor.cls_score.bias'] for ckpt in ckpt_list]
    return torch.cat(cls_weight_list, dim=0), torch.cat(cls_bias_list, dim=0)


def safetensors_to_pth(target_dir: str):
    '''
    convert safetensors format to pth format
    it will deal with all safetensors files in the specified target_dir

    Args:
        target_dir (str):  directory where the safetensors files locate
    '''
    for filename in glob.glob(f"{target_dir}/*.safetensors"):
        ckpt = load_file(filename)
        torch.save(ckpt, filename.replace(".safetensors", ".pth"))


if __name__ == '__main__':
    safetensors_to_pth('../checkpoints')