#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 16:37
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : resnet.py
"""


import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_channel=3):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(num_channel, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ResNetBackbone(nn.Module):

    def __init__(self, num_layers=101, pretrained=True, num_channel=3):
        super().__init__()

        blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}[num_layers]

        self.backbone = ResNet(Bottleneck, blocks, num_channel=num_channel)

        if pretrained:
            loaded = torch.load('./pt_weights/resnet101-imagenet.pth')
            # loaded = torch.load('./dp/modules/backbones/pretrained_models/resnet101-imagenet.pth',
            #                               map_location="cpu")
            # loaded = model_zoo.load_url(
            #     models.resnet.model_urls['resnet{}'.format(num_layers)])

            if num_channel == 3:
                loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']], 1) 
            else :
                loaded['conv1.weight'] = torch.cat([loaded['conv1.weight'].mean(1).unsqueeze(1)], 1)

            new_params = self.backbone.state_dict().copy()
            for i in loaded:
                i_parts = i.split('.')
                if not i_parts[0] == 'fc':
                    new_params['.'.join(i_parts[0:])] = loaded[i]

            self.backbone.load_state_dict(new_params)

    def forward(self, input):
        return self.backbone(input)
