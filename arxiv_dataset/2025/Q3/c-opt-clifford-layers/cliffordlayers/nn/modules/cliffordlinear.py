# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os

import torch
import torch.nn.functional as F
from torch import nn
import ctypes

from cliffordlayers.utils.load_from_clib import load_from_clib

from ...cliffordkernels import (
    get_1d_clifford_kernel,
    get_2d_clifford_kernel,
    get_3d_clifford_kernel,
)
from ...signature import CliffordSignature


class CliffordLinear(nn.Module):
    """Clifford linear layer.

    Args:
        g (Union[List, Tuple]): Clifford signature tensor.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.

    """

    def __init__(
        self,
        g,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        sig = CliffordSignature(g)

        self.register_buffer("g", sig.g)
        self.dim = sig.dim
        self.n_blades = sig.n_blades

        if self.dim == 1:
            self._get_kernel = get_1d_clifford_kernel
        elif self.dim == 2:
            self._get_kernel = get_2d_clifford_kernel
        elif self.dim == 3:
            self._get_kernel = get_3d_clifford_kernel
        else:
            raise NotImplementedError(
                f"Clifford linear layers are not implemented for {self.dim} dimensions. Wrong Clifford signature."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(self.n_blades, out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialization of the Clifford linear weight and bias tensors.
        # The number of blades is taken into account when calculated the bounds of Kaiming uniform.
        nn.init.kaiming_uniform_(
            self.weight.view(self.out_channels, self.in_channels * self.n_blades),
            a=math.sqrt(5),
        )
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.weight.view(self.out_channels, self.in_channels * self.n_blades)
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.float32
        assert self.weight.dtype == torch.float32
        assert self.bias.dtype == torch.float32
        lib = load_from_clib("matrix_multiplication/matrix_multiplication.so")
        # Define the affine_forward function from the shared library
        lib.affine_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float),    # g
            ctypes.c_int,                    # dim
            ctypes.c_int,                    # n_blades
            ctypes.c_int,                    # in_channels
            ctypes.c_int,                    # out_channels
            ctypes.POINTER(ctypes.c_float),  # weight
            ctypes.POINTER(ctypes.c_float),  # bias
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float)   # output
        ]
        B, _, I = x.shape
        output = torch.zeros((B, self.out_channels, I), dtype=torch.float32)
        lib.affine_forward(ctypes.cast(self.g.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                           self.dim, self.n_blades, self.in_channels, self.out_channels,
                           ctypes.cast(self.weight.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                           ctypes.cast(self.bias.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                           ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                           B,
                           ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float)))
        return output