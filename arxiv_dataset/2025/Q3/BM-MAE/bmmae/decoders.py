from typing import Tuple
import torch.nn as nn
import numpy as np
import torch
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from monai.networks.blocks import TransformerBlock
from monai.networks.layers import trunc_normal_


class Decoder(nn.Module):
    def __init__(
        self,
        decoder_hidden_size: int = 384,
        num_layers: int = 4,
        mlp_dim: int = 768,
        dropout_rate: float = 0.0,
        num_heads: int = 12,
        qkv_bias: bool = True,
        img_size: Tuple[int, ...] = (128, 128, 128),
        patch_size: Tuple[int, ...] = (16, 16, 16),
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.modalities = ["t1", "t1ce", "t2", "flair"]
        self.patch_per_modality = np.prod([img_size[i] // patch_size[i] for i in range(len(img_size))])
        self.decoder_hidden_size = decoder_hidden_size
        self.mask_tokens = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))
        trunc_normal_(self.mask_tokens, std=0.02)
        self.decoder_embed = nn.Linear(768, decoder_hidden_size)
        self.decoder_norm = nn.LayerNorm(decoder_hidden_size)
    
        self.decoder_pred = nn.Linear(decoder_hidden_size, 16 * 16 * 16)

        grid_size = []
        for in_size, pa_size in zip(img_size, patch_size):
            grid_size.append(in_size // pa_size)

        self.position_embeddings = build_sincos_position_embedding(grid_size, self.decoder_hidden_size, spatial_dims=3)

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias,
                    save_attn=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.modality_embeddings = nn.ParameterDict(
            {modality: nn.Parameter(torch.zeros(1, 1, self.decoder_hidden_size)) for modality in self.modalities}
        )
        for modality in self.modalities:
            trunc_normal_(self.modality_embeddings[modality], std=0.02)

    def forward(self, encoded_tokens, ids_restore):
        encoded_tokens = self.decoder_embed(encoded_tokens)
    
        x = self.unshuffle_and_add_modality_embeddings(encoded_tokens, ids_restore)
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]
    
        x = {modality: x[:, i * self.patch_per_modality : (i + 1) * self.patch_per_modality] for i, modality in enumerate(self.modalities)}
        return x

    def unshuffle_and_add_modality_embeddings(self, encoded_tokens, ids_restore):
        encoded_tokens_without_cls = encoded_tokens[:, 1:, :]
        
        mask_tokens = self.mask_tokens.repeat(
            encoded_tokens.shape[0], ids_restore.shape[1] + 1 - encoded_tokens.shape[1], 1
        )
        x_ = torch.cat([encoded_tokens_without_cls, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))
        
        modalities_embedding = []
        for modality in self.modalities:
            modality_embedding = self.modality_embeddings[modality].repeat(
                encoded_tokens.shape[0], x_.shape[1] // len(self.modality_embeddings), 1
            )

            modalities_embedding.append(modality_embedding + self.position_embeddings)

        modalities_embedding = torch.cat(modalities_embedding, dim=1)

        x_ = x_ + modalities_embedding
        x_ = torch.cat([encoded_tokens[:, :1, :], x_], dim=1)
        
        return x_