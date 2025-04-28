from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import torch
import sys
import torch.nn as nn
from torch import Tensor
from torchvision.transforms._presets import VideoClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from einops import rearrange
from models.tencent_medical_net.model import generate_model



def r3d_50_yixin_defined(*, weights: Optional[R3D_18_Weights] = None, progress: bool = True, args=None, **kwargs: Any) -> VideoResNet:
    """Construct 18 layer Resnet3D model.

    .. betastatus:: video module

    Reference: `A Closer Look at Spatiotemporal Convolutions for Action Recognition <https://arxiv.org/abs/1711.11248>`__.

    Args:
        weights (:class:`~torchvision.models.video.R3D_18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.R3D_18_Weights`
            below for more details, and possible values. By default, no
            pre-trained weights are used.
        progress (bool): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.resnet.VideoResNet`` base class.
            Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.R3D_18_Weights
        :members:
    """
    weights = R3D_18_Weights.verify(weights)
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