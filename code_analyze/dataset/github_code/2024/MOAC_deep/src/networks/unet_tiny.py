# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from numpy import fft
import numpy as np

## U-Net utils
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2, do not change (H,W)"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # only double H,w
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2] # (B,C,H,W)
        diffX = x2.size()[3] - x1.size()[3]
        # print('pad info:',diffX,diffY)  # debug

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class TinyUNet(nn.Module):
    def __init__(self, n_channels, n_out_channel, bilinear=True):
        super(TinyUNet, self).__init__()
        self.n_channels = n_channels
        self.n_out_channel = n_out_channel
        self.bilinear = bilinear
        factor = 2 if bilinear else 1   # when use biliner method, pre reduce the channel 

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256 // factor))
        # # test deepen the network
        # self.down2 = (Down(128, 256))
        # self.down3 = (Down(256, 512 // factor))
        # self.up0 = (Up(512, 256 // factor, bilinear))

        self.up1 = (Up(256, 128 // factor, bilinear))   # C: 128 + 128, max channel 256
        self.up2 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_out_channel))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # # test deepen the network
        # x4 = self.down3(x3)
        # x = self.up0(x4,x3)
        # x = self.up1(x, x2)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        # xc = x1 + x     # residual path
        logits = self.outc(x)  # use xc if enable residual path, otherwise x
        return logits