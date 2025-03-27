import torch.nn as nn


class FCNDecoder3d(nn.Sequential):

    def __init__(self, in_channels, scale_factor, num_classes):
        layers = [
            nn.Conv3d(in_channels, num_classes, 1),
            nn.ConvTranspose3d(num_classes, num_classes, scale_factor,
                scale_factor)
        ]

        super().__init__(*layers)


class FCN3d(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features[-1])

        return output
