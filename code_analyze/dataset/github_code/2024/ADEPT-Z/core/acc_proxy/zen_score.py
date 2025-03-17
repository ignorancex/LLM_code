"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-08-15 15:00:05
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:34:53
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch import nn

from .utils import network_weight_gaussian_init

__all__ = ["compute_zen_score"]


def compute_zen_score(
    model: nn.Module,
    mixup_gamma: float,
    resolution: int,
    batch_size: int,
    repeat: int,
    fp16=False,
    device="cuda:0",
    in_channels: int = 3,
) -> float:
    info = {}
    nas_score_list = []
    model.to(device)

    if isinstance(device, str):
        device = torch.device(device)

    with torch.no_grad():
        with torch.autocast(device.type, enabled=fp16):
            for repeat_count in range(repeat):
                network_weight_gaussian_init(model)
                input = torch.randn(
                    size=[batch_size, in_channels, resolution, resolution],
                    device=device,
                )
                input2 = torch.randn(
                    size=[batch_size, in_channels, resolution, resolution],
                    device=device,
                )
                mixup_input = input + mixup_gamma * input2
                output = model.forward_pre_GAP(input)
                mixup_output = model.forward_pre_GAP(mixup_input)

                nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
                nas_score = torch.mean(nas_score)

                # compute BN scaling
                log_bn_scaling_factor = 0.0
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        if m.running_var is not None:
                            bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                            log_bn_scaling_factor += torch.log(bn_scaling_factor)

                nas_score = torch.log(nas_score) + log_bn_scaling_factor
                nas_score_list.append(float(nas_score))

    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)

    info["avg_nas_score"] = float(avg_nas_score)
    info["std_nas_score"] = float(std_nas_score)
    info["avg_precision"] = float(avg_precision)
    return info["avg_nas_score"]
