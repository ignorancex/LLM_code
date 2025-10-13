# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""ConvNextV2 vision backbone."""

import einops as eo
from timm import create_model
from timm.layers import LayerNorm2d
from torch import nn

from ._registry import register_model


@register_model
class ConvNextV2(nn.Module):
    """Extract a feature pyramid out of pre-trained ConvNextV2."""

    def __init__(self, output_dim=256, size="l", dropout=0.2, pretrained=False):
        """Set up a variant of ConvNextV2 and the feature extractors."""
        super().__init__()

        self.output_dim = output_dim
        self.size = size
        self.dropout = dropout
        self.pretrained = pretrained

        # Initialize network
        backbone_ref = {
            "t": ["convnextv2_tiny.fcmae", [96, 192, 384, 768]],
            "b": ["convnextv2_base.fcmae", [128, 256, 512, 1024]],
            "l": ["convnextv2_large.fcmae", [192, 384, 768, 1536]],
            "h": ["convnextv2_huge.fcmae", [352, 704, 1408, 2816]],
        }
        network_name, feat_dims = backbone_ref[self.size]
        network = create_model(network_name, pretrained=self.pretrained)
        self.stem = network.stem
        self.stages = network.stages

        # Layers to fuse features across scales
        self._init_feature_fusion(feat_dims, output_dim)
        self.norm_px_feats = LayerNorm2d(output_dim)
        if self.dropout > 0:
            self.px_dropout = nn.Dropout2d(self.dropout)

    def _init_feature_fusion(self, feat_dims, output_dim):
        """Initialize linear layers for feature extraction at different levels."""
        self.project_0 = nn.Conv2d(feat_dims[0], output_dim, 2, 2)
        self.project_1 = nn.Conv2d(feat_dims[1], output_dim, 1, 1)
        self.project_2 = nn.Conv2d(feat_dims[2], output_dim * 4, 1, 1)
        self.project_3 = nn.Conv2d(feat_dims[3], output_dim * 16, 1, 1)

    def forward(self, x):
        """Run ConvNextV2 inference and return a list of feature maps."""
        # Run ConvNext stages
        x = self.stem(x)
        feat_pyramid = []
        for stage in self.stages:
            x = stage(x)
            feat_pyramid.append(x)

        # Fuse into single tensor at 1/8th input resolution
        f0 = self.project_0(feat_pyramid[0])
        f1 = self.project_1(feat_pyramid[1])
        f2 = self.project_2(feat_pyramid[2])
        f3 = self.project_3(feat_pyramid[3])
        f2 = eo.rearrange(f2, "... (c h2 w2) h w -> ... c (h h2) (w w2)", h2=2, w2=2)
        f3 = eo.rearrange(f3, "... (c h2 w2) h w -> ... c (h h2) (w w2)", h2=4, w2=4)
        px_feats = f0 + f1 + f2 + f3
        px_feats = self.norm_px_feats(px_feats)
        if self.dropout > 0:
            px_feats = self.px_dropout(px_feats)

        return px_feats
