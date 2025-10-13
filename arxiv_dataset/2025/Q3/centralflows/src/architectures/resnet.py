from dataclasses import dataclass
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Architecture, make_activation


@dataclass
class Resnet(Architecture):
    # activation: Callable = nn.GELU()
    activation: str = 'gelu'
    width: int = 16
    stage_sizes: Tuple[int] = (3, 3, 3)

    def create(self, input_shape, output_dim):
        conv3x3 = partial(nn.Conv2d, kernel_size=3, padding=1, bias=False)
        conv1x1 = partial(nn.Conv2d, kernel_size=3, padding=1, bias=False)
        norm_layer = partial(GroupNorm, groups=self.width)
        activation = make_activation(self.activation)
        base_width = self.width
        stage_sizes = self.stage_sizes
        scale = sum(stage_sizes) ** (-1 / 2)

        class Block(nn.Module):
            def __init__(self, in_planes, planes, stride=1):
                super().__init__()
                self.conv1 = conv3x3(in_planes, planes, stride=stride)
                self.bn1 = norm_layer(planes)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = norm_layer(planes)
                self.shortcut = nn.Identity()
                if stride != 1:
                    self.shortcut = nn.Sequential(
                        conv1x1(in_planes, planes, stride=stride),
                        norm_layer(planes),
                    )
                nn.init.zeros_(self.bn2.weight)

            def forward(self, x):
                shortcut = self.shortcut(x)
                x = self.bn1(self.conv1(x))
                x = activation(x)
                x = self.bn2(self.conv2(x))
                return shortcut + x * scale

        class ResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = nn.Sequential(
                    conv3x3(input_shape[0], base_width, stride=1),
                    activation,
                )
                stages = []
                width = base_width
                for i, stage_size in enumerate(stage_sizes):
                    stride = 1 if i == 0 else 2
                    stages.append(
                        nn.Sequential(
                            Block(width, stride * width, stride=stride),
                            *[
                                Block(stride * width, stride * width, stride=1)
                                for _ in range(stage_size - 1)
                            ]
                        )
                    )
                    width *= stride
                self.stages = nn.Sequential(*stages)
                image_size = input_shape[1] // 2 ** (len(stage_sizes) + 1)
                flat_dim = width * image_size**2
                self.out_proj = nn.Linear(flat_dim, output_dim)
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu"
                        )
                nn.init.zeros_(self.out_proj.weight)
                nn.init.zeros_(self.out_proj.bias)

            def forward(self, x):
                x = self.in_proj(x)
                x = self.stages(x)
                x = F.avg_pool2d(x, 4)
                x = x.view(x.size(0), -1)
                x = self.out_proj(x)
                return x

        return ResNet()


class GroupNorm(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], self.groups, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (var + 1e-6).sqrt()
        x = x.reshape(shape)
        view = [1, -1] + [1] * len(shape[2:])
        return x * self.weight.view(*view) + self.bias.view(*view)
