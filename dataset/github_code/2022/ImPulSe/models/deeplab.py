import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASPPConv3d(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation,
                dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        ]
        super().__init__(*modules)


class _ASPPPooling3d(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(_ASPPPooling3d, self).__init__(
            nn.AdaptiveAvgPool3d(2),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-3:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='trilinear',
            align_corners=False)


class _ASPP3d(nn.Module):

    def __init__(self, in_channels, atrous_rates, drop_prob, out_channels):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(_ASPPConv3d(in_channels, out_channels, rate))

        modules.append(_ASPPPooling3d(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(len(self.convs) * out_channels, out_channels, 1,
                bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(drop_prob)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)

        return self.project(res)


class DeepLabDecoder3d(nn.Sequential):

    def __init__(self, in_channels, atrous_rates, drop_prob, out_channels,
            num_classes, scale_factor):
        layers = [
            _ASPP3d(in_channels, atrous_rates, drop_prob, out_channels),
            nn.Conv3d(out_channels, num_classes, 1),
            nn.Upsample(scale_factor=scale_factor, mode="trilinear",
                align_corners=True)
        ]

        super().__init__(*layers)


class DeepLab3d(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features[-1])

        return output
