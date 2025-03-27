"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2023-08-15 14:41:08
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:33:52
"""

"""
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
"""


import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from torch import nn

from .utils import network_weight_gaussian_init

__all__ = ["compute_gradnorm_score"]


def cross_entropy(logit, target):
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss


def compute_gradnorm_score(
    model: nn.Module,
    resolution: int,
    batch_size: int,
    device="cuda:0",
    # device = "cpu",
    fp16: bool = False,
    in_channels: int = 3,
) -> float:
    model.train()
    model.requires_grad_(True)
    model.to(device)
    model.zero_grad()

    if isinstance(device, str):
        device = torch.device(device)
    with torch.autocast(device.type, enabled=fp16):
        network_weight_gaussian_init(model)
        input = torch.randn(
            size=[batch_size, in_channels, resolution, resolution], device=device
        )
        output = model(input)

        num_classes = output.shape[1]
        y = torch.randint(low=0, high=num_classes, size=[batch_size], device=device)

        one_hot_y = F.one_hot(y, num_classes).float()

        loss = cross_entropy(output, one_hot_y)
    loss.backward()
    params = []
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, "grad") and p.grad is not None:
                params.append(p.grad.view(-1))

    grad_norm = torch.cat(params).norm(p=2)

    return grad_norm.item()
