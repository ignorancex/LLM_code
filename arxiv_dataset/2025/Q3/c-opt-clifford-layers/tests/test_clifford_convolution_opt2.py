# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from cliffordlayers.nn.modules.cliffordconv import (
    CliffordConv1d,
    CliffordConv2d,
    CliffordConv3d,
)

from cliffordlayers.nn.modules.cliffordconv_base import (
    CliffordConv1d as CliffordConv1dBase,
    CliffordConv2d as CliffordConv2dBase,
    CliffordConv3d as CliffordConv3dBase,
)

from cliffordlayers.nn.modules.cliffordconv_opt1 import (
    CliffordConv2d as CliffordConv2dOpt1
)

from cliffordlayers.nn.modules.cliffordconv_opt2 import (
    CliffordConv1d as CliffordConv1dOpt2,
    CliffordConv2d as CliffordConv2dOpt2,
    CliffordConv3d as CliffordConv3dOpt2,
)

import numpy as np

@pytest.mark.skip_in_ci
def test_Clifford1d_conv_opt2():
    n_batches = 64
    in_channels = 1
    x = torch.randn(n_batches, in_channels, 1, 2)
    torch.manual_seed(233)
    clifford_conv = CliffordConv1d(g=[-1], in_channels=in_channels, out_channels=in_channels, kernel_size=1)
    torch.manual_seed(233)
    clifford_conv_opt = CliffordConv1dOpt2(g=[-1], in_channels=in_channels, out_channels=in_channels, kernel_size=1)
    x_out = clifford_conv(x)
    x_out_opt = clifford_conv_opt(x)
    torch.testing.assert_close(x_out, x_out_opt)

@pytest.mark.skip_in_ci
def test_Clifford1d_conv_opt2_6():
    n_batches = 64
    in_channels = 123
    x = torch.randn(n_batches, in_channels, 13, 2)
    torch.manual_seed(233)
    clifford_conv = CliffordConv1d(g=[-1], in_channels=in_channels, out_channels=in_channels, kernel_size=1)
    torch.manual_seed(233)
    clifford_conv_opt = CliffordConv1dOpt2(g=[-1], in_channels=in_channels, out_channels=in_channels, kernel_size=1)
    x_out = clifford_conv(x)
    x_out_opt = clifford_conv_opt(x)
    torch.testing.assert_close(x_out, x_out_opt)

@pytest.mark.skip_in_ci
def test_Clifford2d_conv_opt2():
    n_batches = 64
    in_channels = 1
    x = torch.randn(n_batches, in_channels, 1, 1, 4)
    torch.manual_seed(233)
    clifford_conv = CliffordConv2d(g=[1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=1)
    torch.manual_seed(233)
    clifford_conv_opt = CliffordConv2dOpt2(g=[1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=1)
    x_out = clifford_conv(x)
    x_out_opt = clifford_conv_opt(x)
    torch.testing.assert_close(x_out, x_out_opt)

@pytest.mark.skip_in_ci
def test_Clifford2d_conv_opt2_6():
    n_batches = 64
    in_channels = 78
    out_channels = 96
    x = torch.randn(n_batches, in_channels, 13, 46, 4)
    torch.manual_seed(233)
    clifford_conv = CliffordConv2d(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=7)
    # clifford_conv.bias = nn.Parameter(torch.zeros_like(clifford_conv.bias))
    torch.manual_seed(233)
    clifford_conv_opt = CliffordConv2dOpt2(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=7)
    # clifford_conv_opt1.bias = nn.Parameter(torch.zeros_like(clifford_conv_opt1.bias))
    x_out = clifford_conv(x)
    x_out_opt = clifford_conv_opt(x)
    # print(f"{x_out=}")
    # print(f"{x_out_opt1=}")
    torch.testing.assert_close(x_out, x_out_opt, atol=2e-5, rtol=2e-5)

@pytest.mark.skip_in_ci
def test_Clifford3d_conv_opt2():
    n_batches = 64
    in_channels = 1
    x = torch.randn(n_batches, in_channels, 1, 1, 1, 8)
    torch.manual_seed(233)
    clifford_conv = CliffordConv3d(g=[-1, -1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=1)
    torch.manual_seed(233)
    clifford_conv_opt = CliffordConv3dOpt2(g=[-1, -1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=1)
    x_out = clifford_conv(x)
    x_out_opt = clifford_conv_opt(x)
    torch.testing.assert_close(x_out, x_out_opt)

@pytest.mark.skip_in_ci
def test_Clifford3d_conv_opt2_6():
    n_batches = 64
    in_channels = 23
    out_channels = 23
    x = torch.randn(n_batches, in_channels, 13, 13, 13, 8)
    torch.manual_seed(233)
    clifford_conv = CliffordConv3d(g=[1, -1, 1], in_channels=in_channels, out_channels=out_channels, kernel_size=7)
    # clifford_conv.bias = nn.Parameter(torch.zeros_like(clifford_conv.bias))
    torch.manual_seed(233)
    clifford_conv_opt = CliffordConv3dOpt2(g=[1, -1, 1], in_channels=in_channels, out_channels=out_channels, kernel_size=7)
    # clifford_conv_opt1.bias = nn.Parameter(torch.zeros_like(clifford_conv_opt1.bias))
    x_out = clifford_conv(x)
    x_out_opt = clifford_conv_opt(x)
    # print(f"{x_out=}")
    # print(f"{x_out_opt1=}")
    torch.testing.assert_close(x_out, x_out_opt, atol=2e-5, rtol=2e-5)