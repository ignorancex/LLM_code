from sympy import factorint
from torch import nn


class Downsampling(nn.Module):
    def __init__(self, channels, sampling, groups=1):
        super(Downsampling, self).__init__()
        self.sampling = sampling
        conv_layers = []
        for p, exp in factorint(sampling).items():
            kernel = 2*(p)+1
            for _ in range(0, exp):
                conv_layers.append(nn.Conv2d(in_channels=channels,
                                             out_channels=channels,
                                             kernel_size=kernel,
                                             padding=kernel // 2,
                                             stride=p,
                                             bias=False,
                                             groups=groups))
        self.conv = nn.Sequential(*conv_layers)
    def forward(self, input):
        return self.conv(input)



class ResidualGroup(nn.Module):
    def __init__(self, channels, kernel_size, features, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        self.pre = nn.Conv2d(channels, features, kernel_size=3, padding=1)
        post = nn.Conv2d(features, channels, kernel_size, padding=(kernel_size // 2), bias=True)
        blocks = [RCAB(features, kernel_size, reduction, res_scale=1) for _ in range(n_resblocks)]
        blocks.append(post)
        self.module = nn.Sequential(*blocks)

    def forward(self, x):
        res = self.pre(x)
        res = self.module(res)
        res += x
        return res