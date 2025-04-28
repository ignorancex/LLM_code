import torch
import torch.nn as nn
from torch import Tensor
from typing import Any
from models.dep_mhsa.attentions.mhsa_collection import DepMHSA
from models.dep_mhsa.resnets.blocks import BasicBlock
from models.dep_mhsa.resnets.stems import R2Plus1dStem
from models.dep_mhsa.resnets.video_resnet import Conv2DPlus1D, VideoResNet, _video_resnet


def resnet18_depmhsa(*, args=None, **kwargs: Any) -> VideoResNet:

    input_size = kwargs['input_size']
    n_frames_list = [kwargs['n_frames']]
    num_classes = kwargs['num_classes']
    # 11, 6, 3, 2
    for i in range(3):
        n_frames_list.append((n_frames_list[-1]+1)//2)

    input_d2_layer = (input_size[0] // 2, input_size[1] // 2)
    input_d4_layer = (input_size[0] // 4, input_size[1] // 4)
    input_d8_layer = (input_size[0] // 8, input_size[1] // 8)
    input_last_layer = (input_size[0] // 16, input_size[1] // 16)
    input_last_layers = [input_d2_layer, input_d4_layer, input_d8_layer, input_last_layer]
    
    net = _video_resnet(
        BasicBlock,
        [Conv2DPlus1D] * 4,
        [2, 2, 2, 2],
        R2Plus1dStem,
        n_frames_list=n_frames_list
        ,input_last_layers = input_last_layers
        ,attn_makers = [None, None, DepMHSA, DepMHSA]
        ,num_classes = num_classes
        ,q_kernel='133'
        ,k_kernel='311'
        ,v_kernel='311133'
        # **kwargs
    )
    return net
