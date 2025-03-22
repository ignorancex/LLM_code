from sade.models import registry, layers, layerspp

import torch
import torch.nn as nn

# from monai.networks.blocks.segresnet_block import (
# #     get_conv_layer,
#     get_upsample_layer,
# )
# from monai.networks.layers.factories import Dropout
# from monai.networks.layers.utils import get_norm_layer
# from monai.utils import UpsampleMode

default_initializer = layers.default_init
MultiSequential = layers.MultiSequential
AttentionBlock = layerspp.ChannelAttentionBlock3d
get_conv_layer = layerspp.get_conv_layer


# @registry.register_model(name="toy3d")
class ToyNet(registry.BaseScoreModel):
    def __init__(self, config):
        super().__init__(config)

        self.init_filters = 8
        time_embedding_sz = 32
        self.convInit = torch.nn.Conv3d(
            in_channels=2,
            out_channels=self.init_filters,
            kernel_size=1,
            padding=0,
            stride=1,
        )

        self.projection = layerspp.GaussianFourierProjection(
            embedding_size=time_embedding_sz // 2, scale=0, learnable=False
        )

        self.block1 = layerspp.ResnetBlockBigGANpp(
            in_channels=8, temb_dim=time_embedding_sz
        )

        self.block2 = layerspp.ResnetBlockBigGANpp(
            in_channels=8, temb_dim=time_embedding_sz, downsample=True
        )

        self.upsample1 = nn.Upsample(scale_factor=(2, 2, 2))

        self.block3 = layerspp.ResnetBlockBigGANpp(
            in_channels=8, temb_dim=time_embedding_sz, downsample=False
        )

        self.convFinal = torch.nn.Conv3d(
            in_channels=8, out_channels=2, kernel_size=1, padding=0, stride=1
        )

    def forward(self, x, scale):
        # gets rid of single dimensions
        scale = scale.squeeze()
        # x: N=batch_size, C=2, H,W,D
        x = self.convInit(x)
        emb = self.projection(scale)
        # x: N, C=8, H,W,D  emb: N, 32
        print("Embedding shape:", emb.shape)

        y1 = self.block1(x, emb)

        # x: N, C=8, H//2,W//2,D//2
        x = self.block2(x, emb)
        x = self.upsample1(x)

        x = self.block3(x + y1, emb)
        x = self.convFinal(x)

        return x
