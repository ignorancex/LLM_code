"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-11-18 22:48:11
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:35:49
"""

import torch
from pyutils.general import logger as lg
from torch import Tensor, nn

from core.models.layers.super_conv2d import SuperBlockConv2d
from core.models.layers.super_linear import SuperBlockLinear

from .cnn import ConvBlock, LinearBlock, SuperOCNN

__all__ = [
    "SuperVGG8",
    "SuperVGG11",
    "SuperVGG13",
    "SuperVGG16",
    "SuperVGG19",
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


class VGG(SuperOCNN):
    """MZI VGG (Shen+, Nature Photonics 2017). Support sparse backpropagation. Blocking matrix multiplication."""

    _conv_linear = (SuperBlockConv2d, SuperBlockLinear)
    _conv = (SuperBlockConv2d,)
    _linear = (SuperBlockLinear,)

    def __init__(
        self,
        vgg_name: str,
        *args,
        **kwargs,
    ) -> None:
        self.vgg_name = vgg_name
        super().__init__(*args, **kwargs)

    def build_layers(self):
        cfg = cfg_32 if self.img_height == 32 else cfg_64
        self.features, convNum = self._make_layers(cfg[self.vgg_name])
        # build FC layers
        ## lienar layer use the last miniblock
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
                mini_block=self.block_list[-1],
                bias=self.bias,
                w_bit=self.w_bit,
                in_bit=self.in_bit,
                v_max=self.v_max,
                photodetect=self.photodetect,
                device=self.device,
                activation=False,
                act_thres=self.act_thres,
                verbose=self.verbose,
            )
        ]
        self.classifier = nn.Sequential(*classifier)

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
                        mini_block=self.block_list[0],
                        bias=self.bias,
                        in_bit=self.in_bit,
                        w_bit=self.w_bit,
                        v_max=self.v_max,
                        photodetect=self.photodetect,
                        device=self.device,
                        activation=True,
                        act_thres=self.act_thres,
                        bn_affine=self.bn_affine,
                        verbose=self.verbose,
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

    def load_from_teacher(self, teacher):
        bn_modules_student = []
        conv_modules_student = []
        linear_modules_student = []

        bn_modules_teacher = []
        conv_modules_teacher = []
        linear_modules_teacher = []

        # student model:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_modules_student.append(m)
            elif isinstance(m, self._conv):
                conv_modules_student.append(m)
            elif isinstance(m, self._linear):
                linear_modules_student.append(m)

        # teacher model:
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_modules_teacher.append(m)
            elif isinstance(m, nn.Conv2d):
                conv_modules_teacher.append(m)
            elif isinstance(m, nn.Linear):
                linear_modules_teacher.append(m)

        # map batch norm
        for bn_stu, bn_tea in zip(bn_modules_student, bn_modules_teacher):
            bn_stu.weight.data.copy_(bn_tea.weight.data)
            bn_stu.bias.data.copy_(bn_tea.bias.data)

        # map bias
        for conv_stu, conv_tea in zip(conv_modules_student, conv_modules_teacher):
            if conv_tea.bias is not None:
                conv_stu.bias.data.copy_(conv_tea.bias.data)

        for linear_stu, linear_tea in zip(
            linear_modules_student, linear_modules_teacher
        ):
            if linear_tea.bias is not None:
                linear_stu.bias.data.copy_(linear_tea.bias.data)

        # map linear/conv layers
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=1e-2,
        )

        # N = 1000
        # N = 2000
        N = 3000

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=N, last_epoch=-1
        )
        target = torch.cat(
            [
                m.weight.data.flatten()
                for m in conv_modules_teacher + linear_modules_teacher
            ]
        )

        def _build_weight_stu(conv_modules, linear_modules):
            modules = conv_modules + linear_modules
            weights = torch.cat([m.build_weight().flatten() for m in modules])
            return weights

        _build_weight = _build_weight_stu

        for i in range(N):
            weights = _build_weight(conv_modules_student, linear_modules_student)
            loss = torch.nn.functional.mse_loss(weights, target, reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i % 10 == 0 or i == N - 1:
                lg.info(f"Step: {i}, loss={loss.item():.4f}")

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def SuperVGG8(*args, **kwargs):
    return VGG("vgg8", *args, **kwargs)


def SuperVGG11(*args, **kwargs):
    return VGG("vgg11", *args, **kwargs)


def SuperVGG13(*args, **kwargs):
    return VGG("vgg13", *args, **kwargs)


def SuperVGG16(*args, **kwargs):
    return VGG("vgg16", *args, **kwargs)


def SuperVGG19(*args, **kwargs):
    return VGG("vgg19", *args, **kwargs)


if __name__ == "__main__":
    pass
