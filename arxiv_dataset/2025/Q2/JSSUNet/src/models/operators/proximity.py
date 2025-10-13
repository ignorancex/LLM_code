import torch
from torch import nn

from src.models.operators.basic_layers import ConvRelu, ResBlock, RCAB


class ProxSR(nn.Module):
    def __init__(self, iter, channels, features, kernel_size):
        super(ProxSR, self).__init__()
        layers = []
        layers.append(ConvRelu(in_channels=channels, out_channels=features, kernel_size=kernel_size))
        for _ in range (iter):
            layers.append(ResBlock(features=features, kernel_size=kernel_size))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=kernel_size//2))
        self.ResNet = nn.Sequential(*layers)
    def forward(self, x):
        return self.ResNet(x)+x

class ProxSSR(nn.Module):
    def __init__(self, channels, kernel_size, features, reduction, n_resblocks):
        super(ProxSSR, self).__init__()
        self.pre = nn.Conv2d(channels, features, kernel_size=3, padding=1,bias=False)
        blocks = [RCAB(features, kernel_size, reduction, res_scale=1) for _ in range(n_resblocks)]
        blocks.append(nn.Conv2d(features, channels, kernel_size, padding=(kernel_size // 2), bias=True))
        self.module = nn.Sequential(*blocks)

    def forward(self, x):
        res = self.pre(x)
        res = self.module(res)
        res += x
        return res

class ProxFus(nn.Module):
    def __init__(self, iter, ms_channels,hs_channels, features, kernel_size):
        super(ProxFus, self).__init__()
        layers = []
        layers.append(ConvRelu(in_channels=ms_channels+hs_channels, out_channels=features, kernel_size=kernel_size))
        for _ in range (iter):
            layers.append(ResBlock(features=features, kernel_size=kernel_size))
        layers.append(nn.Conv2d(in_channels=features, out_channels=hs_channels, kernel_size=kernel_size, padding=kernel_size//2))
        self.ResNet = nn.Sequential(*layers)
    def forward(self, u, ms):
        return self.ResNet(torch.cat([u,ms],dim=1))+u