import torch.nn as nn
import torch.nn.functional as F


class ConvLayer3d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel, bias=False,
            norm=nn.InstanceNorm3d, actv=nn.LeakyReLU, drop_prob=0):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel, padding=kernel // 2,
                bias=bias)
        ]
        if norm is not None:
            layers.append(norm(out_channels))
        if actv is not None:
            layers.append(actv())
        if drop_prob > 0:
            layers.append(nn.Dropout(drop_prob))
        super().__init__(*layers)


class ConvBlock3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, norm, num_layers,
            drop_prob, upsample=False, residual=True):
        super().__init__()

        conv_layers = [ConvLayer3d(in_channels, out_channels, kernel,
            norm=norm, drop_prob=drop_prob) if i == 0 else
            ConvLayer3d(out_channels, out_channels, kernel,
            norm=norm, drop_prob=drop_prob)
            for i in range(num_layers)]
        self.conv_layers = nn.Sequential(*conv_layers)
        self.residual = residual
        if self.residual:
            self.res_layer = ConvLayer3d(in_channels, out_channels, 1,
                norm=norm, actv=None, drop_prob=drop_prob)
        self.upsample = upsample

    def forward(self, x):
        output = self.conv_layers(x)

        if self.residual:
            res = self.res_layer(x)
            output = res + output

        if self.upsample:
            output = F.interpolate(output, scale_factor=2, mode="trilinear",
                align_corners=True)

        return output
