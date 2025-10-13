"""
Custom modules.
"""

import math

import torch
from torch import nn
from torchvision.ops import Permute, StochasticDepth

from .weight_init_1v0 import init_conv, init_linear


# Activation functions


class PMish(nn.Module):
    """
    Parameterized version of Mish activation function [arXiv:1908.08681].

    As weight -> -inf, PMish -> Mish
       weight -> +inf, PMish -> identity function
    """

    def __init__(self, init=0.0):
        super().__init__()
        weight = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(weight, init)
        self.register_parameter("weight", weight)

    def __repr__(self):
        return f"PMish(weight={self.weight.item()})"

    def forward(self, x):
        alpha = nn.functional.sigmoid(self.weight)
        return x * nn.functional.tanh(
            alpha + nn.functional.softplus(x)  # pylint: disable=E1102
        )


# Activation utilities


def act_sel(act_type):
    """
    Return activation functions by their name.
    """
    assert act_type in (None, "linear", "relu", "gelu", "mish", "pmish")
    if act_type == "relu":
        return nn.ReLU()
    if act_type == "gelu":
        return nn.GELU()
    if act_type == "mish":
        return nn.Mish()
    if act_type == "pmish":
        return PMish(init=0.0)
    return None


def act_sel_init(act_type, layer):
    """
    Return activation functions by their name, and init provided layer
    accordingly.
    """
    assert act_type in (None, "linear", "relu", "gelu", "mish", "pmish")
    # Select activation module
    act = act_sel(act_type)
    # Init layer weights
    kaiming_args = None
    if act_type == "pmish":
        kaiming_args = (0.5, "fan_in")
    if act_type is None:
        init_linear(layer, init_gain_act="linear")
    else:
        init_linear(layer, act_type, kaiming_args)
    return act


# Normalization utilities


class GroupView(torch.nn.Module):
    """
    Module returning a view of a tensor where groups of channels form a new
    dimension.
    """

    def __init__(self, dim, groups):
        super().__init__()
        self.dim = dim
        self.groups = groups

    def forward(self, x):
        return x.view(*x.size()[:-1], self.groups, self.dim)


class UnGroupView(torch.nn.Module):
    """
    Module returning a view of a tensor where the group dimension is flatten
    with channels.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flatten(x, start_dim=-2)


def _norm_layers_1d(norm_type, size, eps=1e-6):
    layers = []
    # Determine affine
    affine = False
    if norm_type in ("bna", "lna"):
        affine = True
    # Batch norm
    if norm_type in ("bn", "bna"):
        layers.append(nn.BatchNorm1d(size, eps=eps, affine=affine))
    # Layer norm
    elif norm_type in ("ln", "lna"):
        layers.append(nn.LayerNorm(size, eps=eps, elementwise_affine=affine))
    return layers


def _norm_layers_2d(
    norm_type, size, channel_first=True, ln_groups=1, ln_grouped=False, eps=1e-6
):
    layers = []
    # Determine affine
    affine = False
    if norm_type in ("bna", "lna"):
        affine = True
    # Batch norm
    if norm_type in ("bn", "bna"):
        if not channel_first:
            layers.append(Permute([0, 3, 1, 2]))
        layers.append(nn.BatchNorm2d(size, eps=eps, affine=affine))
        if not channel_first:
            layers.append(Permute([0, 2, 3, 1]))
    # Layer norm
    elif norm_type in ("ln", "lna"):
        if channel_first:
            layers.append(Permute([0, 2, 3, 1]))
        if (not ln_grouped) and (ln_groups > 1):
            layers.append(GroupView(size // ln_groups, ln_groups))
        layers.append(
            nn.LayerNorm(size // ln_groups, eps=eps, elementwise_affine=affine)
        )
        if (not ln_grouped) and (ln_groups > 1):
            layers.append(UnGroupView())
        if channel_first:
            layers.append(Permute([0, 3, 1, 2]))
    return layers


# Convolution utilities


def _conv_spatial_size(size, kernel_size, stride=1, padding=0, dilation=1):
    size = size + 2 * padding - dilation * (kernel_size - 1) - 1
    return math.floor(float(size) / stride + 1)


# Linear block


class LinearBlock(nn.Module):
    """
    Block of linear layers interleaved with activation functions, normalization
    and dropout layers.
    """

    def __init__(
        self,
        lin_sizes,
        act_type="relu",
        norm_type="bna",
        dropout=0.0,
        dropout_min_size=16,
        output_act=True,
        pre_norm=False,
    ):
        super().__init__()
        assert len(lin_sizes) > 1, "lin_sizes requires at least 2 values."
        last_lin = len(lin_sizes) - 2
        if output_act:
            last_lin += 1
        # Linear blocks
        layers = []
        for i, (size_in, size_out) in enumerate(
            zip(lin_sizes[:-1], lin_sizes[1:])
        ):
            if pre_norm:
                layers.extend(_norm_layers_1d(norm_type, size_in))
            bias = (
                pre_norm or (norm_type not in ("bn", "bna")) or (i == last_lin)
            )
            layers.append(nn.Linear(size_in, size_out, bias=bias))
            if i != last_lin:
                act = act_sel_init(act_type, layers[-1])
                if not pre_norm:
                    layers.extend(_norm_layers_1d(norm_type, size_out))
                if act is not None:
                    layers.append(act)
                if dropout and (size_out >= dropout_min_size):
                    layers.append(nn.Dropout(dropout))
            else:
                init_linear(layers[-1], init_gain_act="linear")
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


# LinNeXt modules


class LinNeXtBlock(nn.Module):
    """
    Sub-block of LinNeXt module.
    """

    def __init__(
        self,
        dim,
        layer_scale,
        stochastic_depth_prob,
        act_type="gelu",
        norm_type="lna",
    ):
        super().__init__()
        # Layers
        layers = _norm_layers_1d(norm_type, dim)
        layers.extend(
            [
                nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
                act_sel(act_type),
                nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            ]
        )
        self.block = nn.Sequential(*layers)
        # Parameters
        self.layer_scale = nn.Parameter(torch.ones(dim) * layer_scale)
        # Stochastic depth
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x):
        result = self.layer_scale * self.block(x)
        result = self.stochastic_depth(result)
        result += x
        return result


class LinNeXt(nn.Module):
    """
    Block of linear layers inspired by the ConvNeXt architecture.

    It incorporates activation functions, normalization layer, and a stochastic
    depth dropout mechanism.
    """

    def __init__(
        self,
        lin_sizes,
        n_block_per_stage=3,
        stochastic_depth_prob=0.0,
        min_stochastic_depth_prob=0.0,
        layer_scale=1e-6,
        act_type="gelu",
        norm_type="lna",
    ):
        super().__init__()
        n_stage = len(lin_sizes) - 1
        assert n_stage, "lin_sizes requires at least 2 values."
        # Layers
        layers = []
        if isinstance(n_block_per_stage, int):
            n_block_per_stage = (n_block_per_stage,) * n_stage
        n_blocks = 0
        for i in n_block_per_stage:
            n_blocks += i
        block_id = 0
        for size_in, size_out, n_block_per_stage_i in zip(
            lin_sizes[:-1], lin_sizes[1:], n_block_per_stage
        ):
            # Bottleneck blocks
            stage = []
            for _ in range(n_block_per_stage_i):
                # Adjust stochastic depth probability based on the depth of the
                # block
                sd_prob = block_id / (n_blocks - 1.0)
                sd_prob *= stochastic_depth_prob - min_stochastic_depth_prob
                sd_prob += min_stochastic_depth_prob
                stage.append(
                    LinNeXtBlock(
                        size_in, layer_scale, sd_prob, act_type, norm_type
                    )
                )
                block_id += 1
            layers.append(nn.Sequential(*stage))
            # Downsampling
            downsampling = _norm_layers_1d(norm_type, size_in)
            downsampling.append(nn.Linear(size_in, size_out, bias=True))
            layers.append(nn.Sequential(*downsampling))
        self.features = nn.Sequential(*layers)
        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_linear(m, init_gain_act=0.02)

    def forward(self, x):
        return self.features(x)


# Convolution 2d block


class Conv2dBlock(nn.Module):
    """
    Block of convolution layers interleaved with activation functions,
    2-d normalization and 2-d dropout layers.
    """

    def __init__(
        self,
        conv_sizes,
        kernel_size=3,
        stride=2,
        padding=0,
        act_type="relu",
        norm_type="bna",
        dropout=0.0,
        dropout_min_size=16,
        output_act=True,
        pre_norm=False,
        groups=None,
    ):
        super().__init__()
        self.n_block = len(conv_sizes) - 1
        assert len(conv_sizes) > 1, "conv_sizes requires at least 2 values."
        last_conv = len(conv_sizes) - 2
        if output_act:
            last_conv += 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if groups is None:
            groups = (1,) * (len(conv_sizes) - 1)
        assert len(groups) == (len(conv_sizes) - 1), "Invalid groups length."
        # Convolution blocks
        layers = []
        for i, (size_in, size_out, groups_i) in enumerate(
            zip(conv_sizes[:-1], conv_sizes[1:], groups)
        ):
            if pre_norm:
                layers.extend(
                    _norm_layers_2d(norm_type, size_in, ln_groups=groups_i)
                )
            bias = (
                pre_norm or (norm_type not in ("bn", "bna")) or (i == last_conv)
            )
            layers.append(
                nn.Conv2d(
                    size_in,
                    size_out,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    groups=groups_i,
                    bias=bias,
                )
            )
            if i != last_conv:
                act = act_sel_init(act_type, layers[-1])
                if not pre_norm:
                    layers.extend(
                        _norm_layers_2d(norm_type, size_out, ln_groups=groups_i)
                    )
                if act is not None:
                    layers.append(act)
                if dropout and (size_out >= dropout_min_size):
                    layers.append(nn.Dropout2d(dropout))
            else:
                init_conv(layers[-1], init_gain_act="linear")
        self.features = nn.Sequential(*layers)

    def get_output_spatial_size(self, size):
        """
        Return the output spatial shape of the module.
        """
        for _ in range(self.n_block):
            size = _conv_spatial_size(
                size, self.kernel_size, self.stride, self.padding
            )
        return size

    def forward(self, x):
        return self.features(x)


# ConvNeXt modules


class ConvNeXtStem(nn.Module):
    """
    Stem sub-block of ConvNeXt module.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        norm_type="lna",
        groups=1,
        overlap=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.padding = 0
        if overlap:
            self.padding = self.kernel_size // 2
            self.kernel_size *= 2
        # Layers
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=groups,
                bias=True,
            )
        ]
        layers.extend(
            _norm_layers_2d(norm_type, out_channels, ln_groups=groups)
        )
        self.stem = nn.Sequential(*layers)
        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_conv(m, init_gain_act=0.02, init_bias="uniform")

    def get_output_spatial_size(self, size):
        """
        Return the output spatial shape of the module.
        """
        return _conv_spatial_size(
            size, self.kernel_size, self.stride, self.padding
        )

    def forward(self, x):
        return self.stem(x)


class ConvNeXtBlock(nn.Module):
    """
    Sub-block of ConvNeXtMain module.
    """

    def __init__(
        self,
        dim,
        layer_scale,
        stochastic_depth_prob,
        act_type="gelu",
        norm_type="lna",
        groups=1,
    ):
        super().__init__()
        # Layers
        layers = [
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        ]
        if norm_type in ("bn", "bna"):
            layers.extend(_norm_layers_2d(norm_type, dim))
            layers.append(Permute([0, 2, 3, 1]))
            if groups > 1:
                layers.append(GroupView(dim // groups, groups))
        else:
            layers.append(Permute([0, 2, 3, 1]))
            if groups > 1:
                layers.append(GroupView(dim // groups, groups))
            layers.extend(
                _norm_layers_2d(
                    norm_type,
                    dim,
                    channel_first=False,
                    ln_groups=groups,
                    ln_grouped=True,
                )
            )
        layers.extend(
            [
                nn.Linear(
                    in_features=dim // groups,
                    out_features=4 * dim // groups,
                    bias=True,
                ),
                act_sel(act_type),
                nn.Linear(
                    in_features=4 * dim // groups,
                    out_features=dim // groups,
                    bias=True,
                ),
            ]
        )
        if groups > 1:
            layers.append(UnGroupView())
        layers.append(Permute([0, 3, 1, 2]))
        self.block = nn.Sequential(*layers)
        # Parameters
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        # Stochastic depth
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x):
        result = self.layer_scale * self.block(x)
        result = self.stochastic_depth(result)
        result += x
        return result


class ConvNeXtMain(nn.Module):
    """
    Main sub-block of ConvNeX module.
    """

    def __init__(
        self,
        conv_sizes,
        n_block_per_stage=3,
        stochastic_depth_prob=0.0,
        min_stochastic_depth_prob=0.0,
        layer_scale=1e-6,
        act_type="gelu",
        norm_type="lna",
        groups=None,
        overlap=False,
    ):
        super().__init__()
        self.n_stage = len(conv_sizes) - 1
        assert self.n_stage, "conv_sizes requires at least 2 values."
        if groups is None:
            groups = (1,) * self.n_stage
        assert len(groups) == self.n_stage, "Invalid groups length."
        if overlap:
            down_kernel_size, down_padding = 4, 1
        else:
            down_kernel_size, down_padding = 2, 0
        # Layers
        layers = []
        if isinstance(n_block_per_stage, int):
            n_block_per_stage = (n_block_per_stage,) * self.n_stage
        n_blocks = 0
        for i in n_block_per_stage:
            n_blocks += i
        block_id = 0
        for size_in, size_out, groups_i, n_block_per_stage_i in zip(
            conv_sizes[:-1], conv_sizes[1:], groups, n_block_per_stage
        ):
            # Bottleneck blocks
            stage = []
            for _ in range(n_block_per_stage_i):
                # Adjust stochastic depth probability based on the depth of the
                # block
                sd_prob = block_id / (n_blocks - 1.0)
                sd_prob *= stochastic_depth_prob - min_stochastic_depth_prob
                sd_prob += min_stochastic_depth_prob
                stage.append(
                    ConvNeXtBlock(
                        size_in,
                        layer_scale,
                        sd_prob,
                        act_type,
                        norm_type,
                        groups_i,
                    )
                )
                block_id += 1
            layers.append(nn.Sequential(*stage))
            # Downsampling
            downsampling = _norm_layers_2d(
                norm_type, size_in, ln_groups=groups_i
            )
            downsampling.append(
                nn.Conv2d(
                    size_in,
                    size_out,
                    kernel_size=down_kernel_size,
                    stride=2,
                    padding=down_padding,
                    groups=groups_i,
                )
            )
            layers.append(nn.Sequential(*downsampling))
        self.features = nn.Sequential(*layers)
        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_linear(m, init_gain_act=0.02)
            elif isinstance(m, nn.Conv2d):
                init_conv(m, init_gain_act=0.02)

    def get_output_spatial_size(self, size):
        """
        Return the output spatial shape of the module.
        """
        for _ in range(self.n_stage):
            size = _conv_spatial_size(size, 2, 2)
        return size

    def forward(self, x):
        return self.features(x)


class ConvNeXt(nn.Module):
    """
    Block of layers following the ConvNeXt architecture.

    It incorporates activation functions, normalization layer, and a stochastic
    depth dropout mechanism.

    Modified from: https://github.com/pytorch/vision/blob/main/torchvision/
    models/convnext.py
    """

    def __init__(
        self,
        conv_sizes,
        stem_kernel_size=4,
        n_block_per_stage=3,
        stochastic_depth_prob=0.0,
        min_stochastic_depth_prob=0.0,
        layer_scale=1e-6,
        act_type="gelu",
        norm_type="lna",
        groups=None,
        overlap=False,
        stem_dropout=0.0,
    ):
        super().__init__()
        assert len(conv_sizes) >= 1, "conv_sizes requires at least 2 values."
        if groups is None:
            groups = (1,) * (len(conv_sizes) - 1)
        assert len(groups) == (len(conv_sizes) - 1), "Invalid groups length."
        if isinstance(overlap, bool):
            overlap = (overlap, overlap)
        # ConvNeXt blocks
        layers = []
        if stem_kernel_size is not None:
            layers.append(
                ConvNeXtStem(
                    conv_sizes[0],
                    conv_sizes[1],
                    stem_kernel_size,
                    norm_type,
                    groups[0],
                    overlap[0],
                )
            )
            conv_sizes = conv_sizes[1:]
            groups = groups[1:]
        if stem_dropout:
            layers.append(nn.Dropout2d(stem_dropout))
        if len(conv_sizes) > 1:
            layers.append(
                ConvNeXtMain(
                    conv_sizes,
                    n_block_per_stage,
                    stochastic_depth_prob,
                    min_stochastic_depth_prob,
                    layer_scale,
                    act_type,
                    norm_type,
                    groups,
                    overlap[1],
                )
            )
        self.features = nn.Sequential(*layers)

    def get_output_spatial_size(self, size):
        """
        Return the output spatial shape of the module.
        """
        for m in self.features.children():
            if isinstance(m, (ConvNeXtStem, ConvNeXtMain)):
                size = m.get_output_spatial_size(size)
        return size

    def forward(self, x):
        return self.features(x)
