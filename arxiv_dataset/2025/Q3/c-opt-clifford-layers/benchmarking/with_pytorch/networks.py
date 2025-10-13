from typing import Callable, Union, List
import torch
from torch import nn
from torch.nn import functional as F

from cliffordlayers.nn.modules.cliffordconv import CliffordConv2d, CliffordConv3d
from cliffordlayers.nn.modules.groupnorm import CliffordGroupNorm2d, CliffordGroupNorm3d


class CliffordBasicBlock2d(nn.Module):
    """2D building block for Clifford ResNet architectures.

    Args:
        g (Union[tuple, list, torch.Tensor]): Signature of Clifford algebra.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (Callable, optional): Activation function. Defaults to F.gelu.
        kernel_size (int, optional): Kernel size of Clifford convolution. Defaults to 3.
        stride (int, optional): Stride of Clifford convolution. Defaults to 1.
        padding (int, optional): Padding of Clifford convolution. Defaults to 1.
        rotation (bool, optional): Wether to use rotational Clifford convolution. Defaults to False.
        norm (bool, optional): Wether to use Clifford (group) normalization. Defaults to False.
        num_groups (int, optional): Number of groups when using Clifford (group) normalization. Defaults to 1.
    """

    expansion: int = 1

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        activation: List[Callable],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        norm: bool = False,
        num_groups: int = 1,
        conv2d_class = CliffordConv2d
    ) -> None:
        super().__init__()
        self.conv1 = conv2d_class(
            g,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.conv2 = conv2d_class(
            g,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.norm1 = CliffordGroupNorm2d(g, num_groups, in_channels) if norm else nn.Identity()
        self.norm2 = CliffordGroupNorm2d(g, num_groups, out_channels) if norm else nn.Identity()
        self.activation = activation

    def __repr__(self):
        return "CliffordBasicBlock2d"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation[0](self.norm1(x)))
        out = self.conv2(self.activation[1](self.norm2(out)))
        # in original Microsoft implementation, the residual connection is added
        # we don't do that here because we didn't optimize for padding
        return out

class CliffordBasicBlock3d(nn.Module):
    """3D building block for Clifford ResNet architectures.

    Args:
        g (Union[tuple, list, torch.Tensor]): Signature of Clifford algebra.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (List[Callable]): List of activation functions.
        kernel_size (int, optional): Kernel size of Clifford convolution. Defaults to 3.
        stride (int, optional): Stride of Clifford convolution. Defaults to 1.
        padding (int, optional): Padding of Clifford convolution. Defaults to 0.
        rotation (bool, optional): Whether to use rotational Clifford convolution. Defaults to False.
        norm (bool, optional): Whether to use Clifford (group) normalization. Defaults to False.
        num_groups (int, optional): Number of groups when using Clifford (group) normalization. Defaults to 1.
        conv3d_class (class, optional): Convolution class to use. Defaults to CliffordConv3d.
    """

    expansion: int = 1

    def __init__(
        self,
        g: Union[tuple, list, torch.Tensor],
        in_channels: int,
        out_channels: int,
        activation: List[Callable],
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        norm: bool = False,
        num_groups: int = 1,
        conv3d_class = CliffordConv3d
    ) -> None:
        super().__init__()
        self.conv1 = conv3d_class(
            g,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.conv2 = conv3d_class(
            g,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.norm1 = CliffordGroupNorm3d(g, num_groups, in_channels) if norm else nn.Identity()
        self.norm2 = CliffordGroupNorm3d(g, num_groups, out_channels) if norm else nn.Identity()
        self.activation = activation

    def __repr__(self):
        return "CliffordBasicBlock3d"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation[0](self.norm1(x)))
        out = self.conv2(self.activation[1](self.norm2(out)))
        # in original Microsoft implementation, the residual connection is added
        # we don't do that here because we didn't optimize for padding
        return out