# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List, Optional, Union

import torch
import torch.nn as nn


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Union[List[int], int],
    normalization: Optional[str] = None,
):
    """
    Utility to build a simple MLP model
    """
    layers: List[nn.Module] = []
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    else:
        hidden_dims = list(hidden_dims)
    layer_sizes = [input_dim] + hidden_dims

    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if normalization == "batchnorm":
            layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
        elif normalization == "layernorm":
            layers.append(nn.LayerNorm(layer_sizes[i + 1]))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(layer_sizes[-1], output_dim))
    return nn.Sequential(*layers)


class Predictor(nn.Module):
    """
    Abstract predictor class
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim


class SimplePredictor(Predictor):
    """
    Simple predictor class consisting of an MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int]],
        output_dim: int,
    ):
        super(SimplePredictor, self).__init__(input_dim=input_dim, output_dim=output_dim)
        self.hidden_dims = hidden_dims
        self.net = make_mlp(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)

    def forward(self, x):
        original_dtype = x.dtype
        x = x.type(torch.float32)
        return self.net(x).to(original_dtype)
