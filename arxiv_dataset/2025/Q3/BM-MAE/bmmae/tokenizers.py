from typing import Tuple

import torch
import torch.nn as nn
from monai.networks.blocks import PatchEmbeddingBlock
from monai.utils import ensure_tuple_rep


class MRITokenizer(nn.Module):
    def __init__(
        self,
        patch_size: Tuple[int, ...],
        img_size: Tuple[int, ...],
        hidden_size: int,
        in_channels: int = 1,
        num_heads: int = 12,
        proj_type: str = "conv",
        pos_embed_type: str = "sincos",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        if proj_type == "conv":
            w = self.patch_embedding.patch_embeddings.weight.data
            nn.init.xavier_uniform_(w.view(w.size(0), -1))

        if pos_embed_type == "sincos":
            # disable the gradient of the position embeddings (not done in the used monai version)
            self.patch_embedding.position_embeddings.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)
