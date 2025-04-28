"""
Credits: original code from https://github.com/milesial/Pytorch-UNet
1-dimensional version of the UNet architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# inplace = False


class DoubleConv1d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        mid_channels=None,
        padding=1,
        bias=False,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down1d(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, down_sampling=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(down_sampling), DoubleConv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1d(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride
        )
        self.conv = DoubleConv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_signal_length = x2.size()[2] - x1.size()[2]

        x1 = F.pad(
            x1, [diff_signal_length // 2, diff_signal_length - diff_signal_length // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv1d(nn.Module):
    """Final convolutional layer"""
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class UNet1d(nn.Module):
    def __init__(
        self,
        n_in_channels,
        n_out_channels,
        kernel_size=3,
        padding=1,
        sizes=(16, 32, 64, 128, 256),
        down_sampling=(2, 2, 2, 2),
    ):
        super(UNet1d, self).__init__()
        self.inc = DoubleConv1d(n_in_channels, sizes[0], kernel_size=kernel_size, padding=padding)
        self.downs = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.downs.append(Down1d(sizes[i], sizes[i + 1], kernel_size=kernel_size, padding=padding, down_sampling=down_sampling[i]))
        self.ups = nn.ModuleList()
        for i in reversed(range(len(sizes) - 1)):
            self.ups.append(Up1d(sizes[i + 1], sizes[i], kernel_size=kernel_size, padding=padding))
        self.outc = OutConv1d(sizes[0], n_out_channels, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)
        down_outputs = [x]
        for down in self.downs:
            x = down(x)
            down_outputs.append(x)
        down_outputs = down_outputs[:-1][::-1]  # Exclude the last output and reverse
        for idx, up in enumerate(self.ups):
            x = up(x, down_outputs[idx])
        x = self.outc(x)
        return x
        # x1 = self.inc(x)
        # x2 = self.down1(x1) 
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # f = self.outc(x)
        # return f