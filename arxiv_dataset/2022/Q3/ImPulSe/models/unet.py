import torch
import torch.nn as nn

from .conv import ConvBlock3d, ConvLayer3d


class UNetDecoder(nn.Module):

    def __init__(self, num_channels, num_layers, num_classes):
        super().__init__()

        self.conv_blocks = nn.ModuleList()

        for i in range(len(num_channels) - 1):
            in_channels = num_channels[i] if i == 0 else num_channels[i] * 2
            out_channels = num_channels[i + 1]
            self.conv_blocks.append(
                ConvBlock3d(in_channels, out_channels, 3,
                nn.BatchNorm3d, num_layers, 0, True, False)
            )

        self.output_layer = ConvLayer3d(num_channels[-1] * 2, num_classes, 1,
            True, nn.BatchNorm3d, None)

    def forward(self, features):
        in_features = features[-1]
        for i in range(len(self.conv_blocks)):
            out_features = self.conv_blocks[i](in_features)
            in_features = torch.cat((out_features, features[-(i + 2)]), dim=1)

        output = self.output_layer(in_features)

        return output


class UNet(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)

        return output
