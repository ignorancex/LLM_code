import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ConvBlock3d, ConvLayer3d


class ImplicitEncoder(nn.Module):

    def __init__(self, in_channels, num_channels, num_layers):
        super().__init__()

        self.stem = ConvLayer3d(in_channels, num_channels[0], 7, True)
        self.conv_blocks = nn.ModuleList([ConvBlock3d(num_channels[i],
            num_channels[i + 1], 3, nn.InstanceNorm3d, num_layers, 0)
            for i in range(len(num_channels) - 1)])
        self.downsample = nn.MaxPool3d(2)

    def forward(self, x):
        in_feature = self.stem(x)

        features = [in_feature]
        for i in range(len(self.conv_blocks)):
            out_feature = self.conv_blocks[i](in_feature)
            in_feature = self.downsample(out_feature)
            features.append(in_feature)

        return features


class _LocalNorm3d(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        output = self.norm(x)
        output = output.permute(0, 4, 1, 2, 3)

        return output.contiguous()


class ImplicitDecoder(nn.Module):

    def __init__(self, num_channels, num_classes, num_layers, drop_prob):
        super().__init__()
        # MLP made up of consecutive 1x1x1 conv layers
        self.conv_blocks = nn.ModuleList([ConvBlock3d(num_channels[i],
            num_channels[i + 1], 1, _LocalNorm3d, num_layers, drop_prob)
            for i in range(len(num_channels) - 1)])
        self.output_layer = ConvLayer3d(num_channels[-1], num_classes, 1,
            True, None, None, drop_prob)

    def forward(self, x):
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x)

        output = self.output_layer(x)

        return output


class ImplicitAutoEncoder(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def make_point_encoding(self, features, grids):
        # interpolate features at continuous locations and concatenate
        # zyx -> xyz
        grids = torch.flip(grids, dims=(-1,))
        point_encodings = [F.grid_sample(features[i], grids, "bilinear",
            align_corners=True) for i in range(len(features))]
        # zyx -> xyz
        grids = torch.flip(grids, dims=(-1,))
        point_encodings.insert(0, grids.permute(0, 4, 1, 2, 3))
        point_encodings = torch.cat(point_encodings, dim=1)

        return point_encodings

    def forward(self, x, grids):
        # encode spatial features
        features = self.encoder(x)

        # calculate point encodings with features and coordinates
        point_encodings = self.make_point_encoding(features, grids)

        # decode features and calculate SDF
        output = self.decoder(point_encodings)

        return output
