from tkinter import N
from typing import Any
import torch
from torch import Tensor
from torch import nn
import math
from model.nets_factory import *
# from nets_factory import *


class AlexNet_shallow(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet_shallow, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class AlexNet_deep(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet_deep, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
    
    def forward(self, x):
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class AlexNet_deep_shape(nn.Module):
     def __init__(self, num_classes):
        super(AlexNet_deep_shape, self).__init__()
        tmp_list = []
        for i in range(9):
            tmp_list += [ResnetBlock(dim=192, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_bias=False, use_dropout=False)]
        self.res_block_sequential = nn.Sequential(*tmp_list)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

     def forward(self, x):
        x = self.res_block_sequential(x)
        ret = self.avgpool(x)
        return x, ret


class AlexNet_Reconstructor(nn.Module):
    def __init__(self):
        super(AlexNet_Reconstructor, self).__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=7, stride=5, padding=1, output_padding=3)
        self.norm1 = nn.BatchNorm2d(128)
        self.convt2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=3, padding=1, output_padding=2)
        self.norm2 = nn.BatchNorm2d(64)
        self.convt3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=3, padding=1, output_padding=2)
        self.norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, padding=0)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.convt1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.convt2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.convt3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.tanh(x)
        return x


class AlexNet_Reconstructor_shape(nn.Module):
    def __init__(self):
        super(AlexNet_Reconstructor_shape, self).__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels=192, out_channels=128, kernel_size=5, stride=3, padding=1, output_padding=2)
        self.norm1 = nn.BatchNorm2d(128)
        self.convt2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=3, padding=1, output_padding=2)
        self.norm2 = nn.BatchNorm2d(64)
        self.convt3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7, padding=0)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.convt1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.convt2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.convt3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.tanh(x)
        return x


class AlexNet_Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(AlexNet_Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class AlexNet_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet_Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def alexnet_shallow(num_classes=100):
    return AlexNet_shallow(num_classes=100)


def alexnet_deep(num_classes=100):
    return AlexNet_deep(num_classes=num_classes)


def alexnet_deep_shape(num_classes=100):
    return AlexNet_deep_shape(num_classes=num_classes)



# model1 = alexnet_shallow(num_classes=2)
# model2 = AlexNet_deep_shape(num_classes=2)
# x = model1(x)
# x,y = model2(x)
# print(x.shape, y.shape)
