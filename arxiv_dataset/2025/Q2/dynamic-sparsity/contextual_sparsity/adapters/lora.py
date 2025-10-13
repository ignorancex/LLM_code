# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Any, Dict, List

import torch
from torch import nn

from contextual_sparsity.adapters.base import Adapter
from contextual_sparsity.nn import SparseLinear


class LoRA(Adapter):
    """
    Simple LoRA Adapter for a Linear layer.
    """

    def __init__(self, W: torch.Tensor, rank: int, dropout_rate: float = 0.0):
        super().__init__()
        dtype = W.dtype

        # Usual LoRA initialization
        A = torch.zeros(W.shape[1], rank).to(W.device)
        B = torch.zeros(rank, W.shape[0]).to(W.device)
        A.normal_(0, 1.0 / A.shape[1])

        A = A.to(dtype)
        B = B.to(dtype)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        self.rank = rank

    @staticmethod
    def from_module(linear: nn.Module, **kwargs) -> Adapter:
        if isinstance(linear, SparseLinear):
            W = linear._weight.detach()
        else:
            W = linear.weight.detach()

        adapter = LoRA(W, **kwargs)
        return adapter

    def _hook(
        self,
        module: nn.Module,
        args: List[Any],
        kwargs: Dict[str, Any],
        out: torch.Tensor,
    ) -> torch.Tensor:
        x = args[0]

        # If the Adapter is applied to a Sparse Linear Layer
        if isinstance(module, SparseLinear):
            # Apply the same mask to the input before using the adapter
            if module._col_mask is not None:
                mask = module._col_mask.to(x.device).to(x.dtype).detach()
                x = mask * x

        out = out + self.forward(x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.A
        z = self.dropout(z)
        return z @ self.B

    def __repr__(self):
        return f"LoRA(rank={self.rank}, dropout={self.dropout_rate})"
