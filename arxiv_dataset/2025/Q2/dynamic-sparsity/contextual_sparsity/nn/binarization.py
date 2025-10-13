# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
from torch import nn


class ThresholdMask(nn.Module):
    def __init__(self, threshold: float):
        """
        Binarization layer based on the k largest elements on a fixed threshold
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.BoolTensor:
        return (x > self.threshold).squeeze(-1)

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


class TopKMask(nn.Module):
    def __init__(self, k: int):
        """
        Binarization layer based on the k largest elements on the last dimension
        """
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.BoolTensor:
        # Make sure we have at least 2 dims
        if x.ndim == 1:
            x = x.unsqueeze(0)

        assert x.ndim >= 2

        if self.k > x.size(-1):
            mask = torch.ones_like(x).bool()
        else:
            mask = torch.zeros_like(x)

            _, top_idx = torch.topk(x, self.k, dim=-1)
            mask.scatter_(-1, top_idx, 1)

        return mask.bool()

    def extra_repr(self) -> str:
        return f"k={self.k}"


class RandomMask(nn.Module):
    """
    Binarization layer based on random mask
    """

    def forward(self, x: torch.Tensor) -> torch.BoolTensor:
        return torch.rand_like(x) >= self.p

    def extra_repr(self) -> str:
        return f"p={self.p}"


class RandomKMask(TopKMask):
    """
    Binarization layer with k entries set to 1
    """

    def forward(self, x: torch.Tensor) -> torch.BoolTensor:
        return super().forward(torch.rand_like(x))


class StaticMask(nn.Module):
    def __init__(self, mask: torch.BoolTensor):
        """
        Static binarization based on a specified mask
        """
        super(StaticMask, self).__init__()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        assert mask.ndim == 2
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.mask = self.mask.to(x.device)
        assert x.shape[-1] == self.mask.shape[-1]
        if x.ndim == 1:
            return self.mask.squeeze(0)
        elif x.ndim == 2:
            return self.mask.repeat(x.shape[0], 1)
        elif x.ndim == 3:
            return self.mask.unsqueeze(0).repeat(x.shape[0], x.shape[1], 1)
