# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
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

import pytest

import numpy as np


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_complex_convolution():
    """Test Clifford1d convolution module against complex convolution module using g = [-1]."""
    in_channels = 8
    out_channels = 16
    x = torch.randn(1, in_channels, 128, 2)
    clifford_conv = CliffordConv1d(g=[-1], in_channels=in_channels, out_channels=out_channels, kernel_size=3)
    output_clifford_conv = clifford_conv(x)
    w_c = torch.view_as_complex(torch.stack((clifford_conv.weight[0], clifford_conv.weight[1]), -1))
    b_c = torch.view_as_complex(clifford_conv.bias.permute(1, 0).contiguous())
    input_c = torch.view_as_complex(x)
    output_c = F.conv1d(input_c, w_c, b_c)
    torch.testing.assert_close(output_clifford_conv, torch.view_as_real(output_c))


def test_Clifford1d_conv_shapes():
    """Test shapes of Clifford1d convolution module."""
    in_channels = 8
    x = torch.randn(1, 8, 128, 2)
    clifford_conv = CliffordConv1d(g=[1], in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
    x_out = clifford_conv(x)
    assert x.shape == x_out.shape

def test_Clifford1d_conv_base():
    in_channels = 8
    x = torch.randn(1, 8, 128, 2)
    torch.manual_seed(0)
    clifford_conv = CliffordConv1d(g=[-1], in_channels=in_channels, out_channels=in_channels, kernel_size=3)
    torch.manual_seed(0)
    clifford_conv_base = CliffordConv1dBase(g=[-1], in_channels=in_channels, out_channels=in_channels, kernel_size=3)
    x_out = clifford_conv(x)
    x_out_base = clifford_conv_base(x)
    torch.testing.assert_close(x_out, x_out_base)

def test_Clifford1d_conv_base_2():
    in_channels = 8
    x = torch.randn(2, 8, 128, 2)
    torch.manual_seed(0)
    clifford_conv = CliffordConv1d(g=[-1], in_channels=in_channels, out_channels=in_channels, kernel_size=3)
    torch.manual_seed(0)
    clifford_conv_base = CliffordConv1dBase(g=[-1], in_channels=in_channels, out_channels=in_channels, kernel_size=3)
    x_out = clifford_conv(x)
    x_out_base = clifford_conv_base(x)
    torch.testing.assert_close(x_out, x_out_base)

def test_Clifford2d_conv_base():
    in_channels = 8
    x = torch.randn(1, 8, 128, 128, 4)
    torch.manual_seed(233)
    clifford_conv = CliffordConv2d(g=[1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=16)
    torch.manual_seed(233)
    clifford_conv_base = CliffordConv2dBase(g=[1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=16)
    x_out = clifford_conv(x)
    x_out_base = clifford_conv_base(x)
    torch.testing.assert_close(x_out, x_out_base)

def test_Clifford2d_conv_base_2():
    in_channels = 8
    x = torch.randn(2, 8, 128, 128, 4)
    torch.manual_seed(233)
    clifford_conv = CliffordConv2d(g=[1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=16)
    torch.manual_seed(233)
    clifford_conv_base = CliffordConv2dBase(g=[1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=16)
    x_out = clifford_conv(x)
    x_out_base = clifford_conv_base(x)
    torch.testing.assert_close(x_out, x_out_base)

@pytest.mark.skip_in_ci
def test_Clifford3d_conv_base():
    in_channels = 8
    x = torch.randn(1, 8, 32, 32, 32, 8)
    torch.manual_seed(0)
    clifford_conv = CliffordConv3d(g=[1, 1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=5)
    torch.manual_seed(0)
    clifford_conv_base = CliffordConv3dBase(g=[1, 1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=5)
    x_out = clifford_conv(x)
    x_out_base = clifford_conv_base(x)
    torch.testing.assert_close(x_out, x_out_base)

def test_Clifford2d_conv_opt1():
    n_batches = 8
    in_channels = 1
    x = torch.randn(n_batches, in_channels, 1, 1, 4)
    torch.manual_seed(233)
    clifford_conv = CliffordConv2d(g=[1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=1)
    torch.manual_seed(233)
    clifford_conv_opt1 = CliffordConv2dOpt1(g=[1, -1], in_channels=in_channels, out_channels=in_channels, kernel_size=1)
    x_out = clifford_conv(x)
    x_out_opt1 = clifford_conv_opt1(x)
    torch.testing.assert_close(x_out, x_out_opt1)

def test_Clifford2d_conv_opt1_2():
    n_batches = 8
    in_channels = 5
    out_channels = 3
    x = torch.randn(n_batches, in_channels, 1, 1, 4)
    torch.manual_seed(233)
    clifford_conv = CliffordConv2d(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    # clifford_conv.bias = nn.Parameter(torch.zeros_like(clifford_conv.bias))
    torch.manual_seed(233)
    clifford_conv_opt1 = CliffordConv2dOpt1(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    # clifford_conv_opt1.bias = nn.Parameter(torch.zeros_like(clifford_conv_opt1.bias))
    x_out = clifford_conv(x)
    x_out_opt1 = clifford_conv_opt1(x)
    print(f"{x_out=}")
    print(f"{x_out_opt1=}")
    torch.testing.assert_close(x_out, x_out_opt1)

def test_Clifford2d_conv_opt1_3():
    n_batches = 8
    in_channels = 5
    out_channels = 3
    x = torch.randn(n_batches, in_channels, 1, 1, 4)
    torch.manual_seed(233)
    clifford_conv = CliffordConv2d(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    # clifford_conv.bias = nn.Parameter(torch.zeros_like(clifford_conv.bias))
    torch.manual_seed(233)
    clifford_conv_opt1 = CliffordConv2dOpt1(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    # clifford_conv_opt1.bias = nn.Parameter(torch.zeros_like(clifford_conv_opt1.bias))
    x_out = clifford_conv(x)
    x_out_opt1 = clifford_conv_opt1(x)
    print(f"{x_out=}")
    print(f"{x_out_opt1=}")
    torch.testing.assert_close(x_out, x_out_opt1)

def test_Clifford2d_conv_opt1_4():
    n_batches = 16
    in_channels = 123
    out_channels = 456
    x = torch.randn(n_batches, in_channels, 1, 1, 4)
    torch.manual_seed(233)
    clifford_conv = CliffordConv2d(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    # clifford_conv.bias = nn.Parameter(torch.zeros_like(clifford_conv.bias))
    torch.manual_seed(233)
    clifford_conv_opt1 = CliffordConv2dOpt1(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    # clifford_conv_opt1.bias = nn.Parameter(torch.zeros_like(clifford_conv_opt1.bias))
    x_out = clifford_conv(x)
    x_out_opt1 = clifford_conv_opt1(x)
    # print(f"{x_out=}")
    # print(f"{x_out_opt1=}")
    torch.testing.assert_close(x_out, x_out_opt1)

def test_Clifford2d_conv_opt1_5():
    n_batches = 8
    in_channels = 1
    out_channels = 1
    x = torch.randn(n_batches, in_channels, 123, 456, 4)
    torch.manual_seed(233)
    clifford_conv = CliffordConv2d(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=7)
    # clifford_conv.bias = nn.Parameter(torch.zeros_like(clifford_conv.bias))
    torch.manual_seed(233)
    clifford_conv_opt1 = CliffordConv2dOpt1(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=7)
    # clifford_conv_opt1.bias = nn.Parameter(torch.zeros_like(clifford_conv_opt1.bias))
    x_out = clifford_conv(x)
    x_out_opt1 = clifford_conv_opt1(x)
    # print(f"{x_out=}")
    # print(f"{x_out_opt1=}")
    torch.testing.assert_close(x_out, x_out_opt1)

@pytest.mark.skip_in_ci
def test_Clifford2d_conv_opt1_6():
    n_batches = 48
    in_channels = 78
    out_channels = 96
    x = torch.randn(n_batches, in_channels, 13, 46, 4)
    torch.manual_seed(233)
    clifford_conv = CliffordConv2d(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=7)
    # clifford_conv.bias = nn.Parameter(torch.zeros_like(clifford_conv.bias))
    torch.manual_seed(233)
    clifford_conv_opt1 = CliffordConv2dOpt1(g=[1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=7)
    # clifford_conv_opt1.bias = nn.Parameter(torch.zeros_like(clifford_conv_opt1.bias))
    x_out = clifford_conv(x)
    x_out_opt1 = clifford_conv_opt1(x)
    # print(f"{x_out=}")
    # print(f"{x_out_opt1=}")
    torch.testing.assert_close(x_out, x_out_opt1, atol=2e-5, rtol=2e-5)

def test_Clifford2d_conv_shapes():
    """Test shapes of Clifford2d convolution module."""
    in_channels = 8
    x = torch.randn(1, 8, 128, 128, 4)
    clifford_conv = CliffordConv2d(
        g=[1, 1], in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1
    )
    x_out = clifford_conv(x)
    clifford_conv_rotation = CliffordConv2d(
        g=[-1, -1], in_channels=8, out_channels=8, kernel_size=3, padding=1, rotation=True
    )
    x_out_rot = clifford_conv_rotation(x)
    assert x.shape == x_out.shape
    assert x.shape == x_out_rot.shape


def test_Clifford2d_conv_params():
    """Test parameters of Clifford2d convolution using g = [-1, -1] vs Clifford2d rotational convolution.
    When bias is set to False the ration needs to be 4/5.
    """
    in_channels = 8
    out_channels = 16
    torch.randn(1, in_channels, 128, 128, 4)
    clifford_conv = CliffordConv2d(
        g=[-1, -1], in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False
    )
    clifford_conv_rotation = CliffordConv2d(
        g=[-1, -1],
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
        bias=False,
        rotation=True,
    )
    torch.testing.assert_close(float(count_params(clifford_conv) / count_params(clifford_conv_rotation)), 0.8)


def test_Clifford3d_conv_shapes():
    """Test shapes of Clifford2d convolution module."""
    in_channels = 8
    x = torch.randn(1, in_channels, 32, 32, 32, 8)
    clifford_conv = CliffordConv3d(
        g=[1, 1, 1], in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1
    )
    x_out = clifford_conv(x)
    assert x.shape == x_out.shape
