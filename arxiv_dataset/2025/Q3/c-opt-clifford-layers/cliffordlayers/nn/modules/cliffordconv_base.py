# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ctypes
import math
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair, _single, _triple

from cliffordlayers.utils.load_from_clib import load_from_clib

from ...cliffordkernels import (
    get_1d_clifford_kernel,
    get_2d_clifford_kernel,
    get_2d_clifford_rotation_kernel,
    get_3d_clifford_kernel,
)
from ...signature import CliffordSignature
from ..functional.utils import clifford_convnd


class _CliffordConvNd(nn.Module):
    """Base class for all Clifford convolution modules."""

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        bias: bool,
        padding_mode: str,
        rotation: bool = False,
    ) -> None:
        super().__init__()
        assert all(i == 1 for i in stride)
        assert all(i == 0 for i in padding)
        assert all(i == 1 for i in dilation)
        assert groups == 1
        assert rotation is False
        assert all(kernel_size[0] == i for i in kernel_size)

        lib = load_from_clib("convolutional/conv_base.so")
        # void conv_base(float* g, int n_batches, int dim, int n_blades, int in_channels, int d1, int d2, int d3, int out_channels, int filter_size, float* weight, float* bias, float* x, float* output) {
        lib.conv_base.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib = lib

        sig = CliffordSignature(g)
        # register as buffer as we want the tensor to be moved to the same device as the module
        self.register_buffer("g", sig.g)
        self.dim = sig.dim
        self.n_blades = sig.n_blades
        if rotation:
            assert (
                self.dim == 2
            ), "2d rotational Clifford layers are only available for g = [-1, -1]. Make sure you have the right signature."

        if self.dim == 1:
            self._get_kernel = get_1d_clifford_kernel
        elif self.dim == 2 and rotation:
            self._get_kernel = get_2d_clifford_rotation_kernel
        elif self.dim == 2:
            self._get_kernel = get_2d_clifford_kernel
        elif self.dim == 3:
            self._get_kernel = get_3d_clifford_kernel
        else:
            raise NotImplementedError(
                f"Clifford convolution not implemented for {self.dim} dimensions. Wrong Clifford signature."
            )

        if padding_mode != "zeros":
            raise NotImplementedError(f"Padding mode {padding_mode} not implemented.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.rotation = rotation

        self.weight = nn.ParameterList(
            [nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size)) for _ in range(self.n_blades)]
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)

        if rotation:
            self.scale_param = nn.Parameter(torch.Tensor(self.weight[0].shape))
            self.zero_kernel = nn.Parameter(torch.zeros(self.weight[0].shape), requires_grad=False)
            self.weight.append(self.scale_param)
            self.weight.append(self.zero_kernel)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialization of the Clifford convolution weight and bias tensors.
        The number of blades is taken into account when calculated the bounds of Kaiming uniform.
        """
        for blade, w in enumerate(self.weight):
            # Weight initialization for Clifford weights.
            if blade < self.n_blades:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    torch.Tensor(
                        self.out_channels, int(self.in_channels * self.n_blades / self.groups), *self.kernel_size
                    )
                )
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(w, -bound, bound)
            # Extra weights for 2d Clifford rotation layer.
            elif blade == self.n_blades:
                assert self.rotation is True
                # Default channel_in / channel_out initialization for scaling params.
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            elif blade == self.n_blades + 1:
                # Nothing to be done for zero kernel.
                pass
            else:
                raise ValueError(
                    f"Wrong number of Clifford weights. Expected {self.n_blades} weight tensors, and 2 extra tensors for rotational kernels."
                )

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                torch.Tensor(self.out_channels, int(self.in_channels * self.n_blades / self.groups), *self.kernel_size)
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, conv_fn: callable) -> torch.Tensor:
        if self.bias is not None:
            b = self.bias.view(-1)
        else:
            b = None
        print(f"{len(self.weight)=}")
        print(f"{self.weight[0].shape=}")
        output_blades, w = self._get_kernel(self.weight, self.g)
        print(f"{w.shape=}")
        return clifford_convnd(
            conv_fn,
            x,
            output_blades,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}" ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"

        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"

        return s.format(**self.__dict__)


class CliffordConv1d(_CliffordConvNd):
    """1d Clifford convolution.

    Args:
        g (Union[tuple, list, torch.Tensor]): Clifford signature.
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        padding (int): padding added to both sides of the input.
        dilation (int): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        padding_mode (str): Padding to use.
    """

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = _single(padding)
        dilation_ = _single(dilation)

        super().__init__(
            g,
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            groups,
            bias,
            padding_mode,
        )
        if not self.dim == 1:
            raise NotImplementedError("Wrong Clifford signature for CliffordConv1d.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = ctypes.cast(self.g.data_ptr(), ctypes.POINTER(ctypes.c_float))
        n_batches = x.shape[0]
        dim = self.dim
        n_blades = self.n_blades
        in_channels = self.in_channels
        D_dim = x.shape[2:-1]
        d1 = D_dim[0]
        out_channels = self.out_channels
        filter_size = self.kernel_size[0]
        weight = torch.stack(list(self.weight), dim=0)
        output = torch.zeros((n_batches, out_channels, d1-filter_size+1, n_blades), dtype=torch.float32)
        self.lib.conv_base(
            g,
            n_batches,
            dim,
            n_blades,
            in_channels,
            d1,
            -1,
            -1,
            out_channels,
            filter_size,
            ctypes.cast(weight.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.bias.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        )
        return output


class CliffordConv2d(_CliffordConvNd):
    """2d Clifford convolution.

    Args:
        g (Union[tuple, list, torch.Tensor]): Clifford signature.
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int]]): padding added to both sides of the input.
        dilation (Union[int, Tuple[int, int]]): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        padding_mode (str): Padding to use.
        rotation (bool): If True, enables the rotation kernel for Clifford convolution.
    """

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        rotation: bool = False,
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)

        super().__init__(
            g,
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            groups,
            bias,
            padding_mode,
            rotation,
        )
        if not self.dim == 2:
            raise NotImplementedError("Wrong Clifford signature for CliffordConv2d.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = ctypes.cast(self.g.data_ptr(), ctypes.POINTER(ctypes.c_float))
        n_batches = x.shape[0]
        dim = self.dim
        n_blades = self.n_blades
        in_channels = self.in_channels
        D_dim = x.shape[2:-1]
        d1 = D_dim[0]
        d2 = D_dim[1]
        out_channels = self.out_channels
        filter_size = self.kernel_size[0]
        weight = torch.stack(list(self.weight), dim=0)
        output = torch.zeros((n_batches, out_channels, d1-filter_size+1, d2-filter_size+1, n_blades), dtype=torch.float32)
        self.lib.conv_base(
            g,
            n_batches,
            dim,
            n_blades,
            in_channels,
            d1,
            d2,
            -1,
            out_channels,
            filter_size,
            ctypes.cast(weight.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.bias.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        )
        return output


class CliffordConv3d(_CliffordConvNd):
    """3d Clifford convolution.

    Args:
        g (Union[tuple, list, torch.Tensor]): Clifford signature.
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (Union[int, Tuple[int, int, int]]): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int, int]]): Stride of the convolution.
        padding (Union[int, Tuple[int, int, int]]): padding added to all sides of the input.
        dilation (Union[int, Tuple[int, int, int]]): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        bias (bool): If True, adds a learnable bias to the output.
        padding_mode (str): Padding to use.
    """

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)

        super().__init__(
            g,
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            groups,
            bias,
            padding_mode,
        )
        if not self.dim == 3:
            raise NotImplementedError("Wrong Clifford signature for CliffordConv3d.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = ctypes.cast(self.g.data_ptr(), ctypes.POINTER(ctypes.c_float))
        n_batches = x.shape[0]
        dim = self.dim
        n_blades = self.n_blades
        in_channels = self.in_channels
        D_dim = x.shape[2:-1]
        d1 = D_dim[0]
        d2 = D_dim[1]
        d3 = D_dim[2]
        out_channels = self.out_channels
        filter_size = self.kernel_size[0]
        weight = torch.stack(list(self.weight), dim=0)
        output = torch.zeros((n_batches, out_channels, d1-filter_size+1, d2-filter_size+1, d3-filter_size+1, n_blades), dtype=torch.float32)
        self.lib.conv_base(
            g,
            n_batches,
            dim,
            n_blades,
            in_channels,
            d1,
            d2,
            d3,
            out_channels,
            filter_size,
            ctypes.cast(weight.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(self.bias.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        )
        return output
