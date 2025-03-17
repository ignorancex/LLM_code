"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-18 22:48:11
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:35:58
"""

from typing import Union

import torch
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn
from torch.types import _size

from core.models.layers.activation import ReLUN

__all__ = [
    "VGG8",
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19",
]

cfg_32 = {
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

cfg_64 = {
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "GAP"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "GAP"],
    "vgg13": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "GAP",
    ],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "GAP",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "GAP",
    ],
}


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: bool = True,
        act_thres: float = 6.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(
            in_features,
            out_features,
            bias=bias,
        )

        self.activation = ReLUN(act_thres, inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = False,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        activation: bool = True,
        act_thres: float = 6.0,
        bn_affine: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            stride=stride,
            padding=padding,
        )

        self.bn = nn.BatchNorm2d(
            out_channels, affine=bn_affine, track_running_stats=bn_affine
        )

        self.activation = ReLUN(act_thres, inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class VGG(nn.Module):
    """VGG."""

    _conv_linear = (
        nn.Conv2d,
        nn.Linear,
    )

    def __init__(
        self,
        vgg_name: str,
        img_height,
        img_width,
        in_channels,
        num_classes,
        bias=False,
        act_thres=6,
        device="cuda:0",
        bn_affine=True,
    ) -> None:
        self.vgg_name = vgg_name
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bias = bias
        self.act_thres = act_thres
        self.bn_affine = bn_affine
        self.device = device

        super().__init__()
        self.build_layers()

    def build_layers(self):
        cfg = cfg_32 if self.img_height == 32 else cfg_64
        self.features, convNum = self._make_layers(cfg[self.vgg_name])
        # build FC layers
        ## linear layer use the last miniblock
        if (
            self.img_height == 64 and self.vgg_name == "vgg8"
        ):  ## model is too small, do not use dropout
            classifier = []
        else:
            classifier = [nn.Dropout(0.5)]

        classifier += [
            LinearBlock(
                512,
                self.num_classes,
                bias=self.bias,
                activation=False,
                act_thres=self.act_thres,
            )
        ]
        self.classifier = nn.Sequential(*classifier)

    def reset_parameters(self, random_state=None):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def get_parameter_groups(self, weight_decay: float = 0, lr: float = 1e-3):
        param_optimizer = list(self.named_parameters())

        no_decay = ["bias", "weight"]
        no_decay_group = [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ]
        no_decay_reduce_lr_group = []
        for m in self.modules():
            if isinstance(m, self._conv_linear):
                for name, p in m.named_parameters():
                    if "weight" in name:
                        no_decay_reduce_lr_group.append(p)
        no_decay_group = list(set(no_decay_group) - set(no_decay_reduce_lr_group))
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
                "lr": lr,
            },  # no decay for finetuning/distill
            {"params": no_decay_group, "weight_decay": 0.0, "lr": lr},
            {"params": no_decay_reduce_lr_group, "weight_decay": 0.0, "lr": lr / 10},
        ]
        return optimizer_grouped_parameters

    def _make_layers(self, cfg):
        layers = []
        in_channel = self.in_channels
        convNum = 0

        for x in cfg:
            # MaxPool2d
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == "GAP":
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
            else:
                # conv + BN + RELU
                layers += [
                    ConvBlock(
                        in_channel,
                        x,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=self.bias,
                        activation=True,
                        act_thres=self.act_thres,
                        bn_affine=self.bn_affine,
                    )
                ]
                in_channel = x
                convNum += 1
        return nn.Sequential(*layers), convNum

    def forward_pre_GAP(self, x: Tensor) -> Tensor:
        for layer in self.features:
            if isinstance(layer, (nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                break
            x = layer(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def VGG8(*args, **kwargs):
    return VGG("vgg8", *args, **kwargs)


def VGG11(*args, **kwargs):
    return VGG("vgg11", *args, **kwargs)


def VGG13(*args, **kwargs):
    return VGG("vgg13", *args, **kwargs)


def VGG16(*args, **kwargs):
    return VGG("vgg16", *args, **kwargs)


def VGG19(*args, **kwargs):
    return VGG("vgg19", *args, **kwargs)


if __name__ == "__main__":
    pass
