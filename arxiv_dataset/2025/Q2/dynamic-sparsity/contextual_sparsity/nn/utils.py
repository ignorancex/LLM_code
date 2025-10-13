# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
from torch import nn


class Abs(nn.Module):
    """
    Absolute value as an nn.Module layer.
    """

    def forward(self, x):
        return torch.abs(x)
