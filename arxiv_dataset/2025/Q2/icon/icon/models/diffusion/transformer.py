"""
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
"""
import torch
from torch import nn
from torch import Tensor
from typing import Optional, List
from icon.models.diffusion.positional_embedding import SinusoidalPosEmb

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py#L10
class TransformerForDiffusion(nn.Module):

    def __init__(
        self,
        obs_horizon: int,
        prediction_horizon: int,
        input_dim: int,
        cond_dim: int,
        embed_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        causal_attn: Optional[bool] = True
    ) -> None:
        super().__init__()
        self.timestep_embed = SinusoidalPosEmb(embed_dim)
        self.input_embed = nn.Linear(input_dim, embed_dim)
        self.obs_cond_embed = nn.Linear(cond_dim, embed_dim)
        self.input_pos_embed = nn.Parameter(torch.zeros(1, prediction_horizon, embed_dim))
        self.cond_pos_embed = nn.Parameter(torch.zeros(1, 1 + obs_horizon, embed_dim))
        # Encoder
        if num_encoder_layers == 0:
            self.encoder = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.Mish(),
                nn.Linear(4 * embed_dim, embed_dim)
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4 * embed_dim,
                dropout=0.3,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_encoder_layers
            )
        # Decoder
        self.causal_attn = causal_attn
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.3,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, input_dim)
        self.init_weights()

    def init_weights(self) -> None:

        def _init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                weight_names = [
                    'in_proj_weight',
                    'q_proj_weight',
                    'k_proj_weight',
                    'v_proj_weight'
                ]
                for name in weight_names:
                    weight = getattr(module, name)
                    if weight is not None:
                        nn.init.normal_(weight, std=0.02)
                bias_names = [
                    'in_proj_bias',
                    'bias_k',
                    'bias_v'
                ]
                for name in bias_names:
                    bias = getattr(module, name)
                    if bias is not None:
                        nn.init.zeros_(bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
            elif isinstance(module, TransformerForDiffusion):
                nn.init.normal_(module.input_pos_embed, std=0.02)
                nn.init.normal_(module.cond_pos_embed, std=0.02)
                
        self.apply(_init_weights)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        obs_cond: Tensor
    ) -> Tensor:
        """
        Args:
            timesteps (torch.Tensor): diffusion timesteps (batch_size,).
            obs_cond (torch.Tensor): observation conditionings (batch_size, obs_horizon, cond_dim).
        """
        # Encoding.
        timestep_embed = self.timestep_embed(timesteps)
        obs_cond_embed = self.obs_cond_embed(obs_cond)
        cond = torch.cat([timestep_embed.unsqueeze(1), obs_cond_embed], dim=1)
        cond = cond + self.cond_pos_embed
        cond = self.encoder(cond)
        # Decoding
        x = self.input_embed(x)
        x = x + self.input_pos_embed
        memory = cond
        tgt = x
        if self.causal_attn:
            H = x.shape[1]
            T = cond.shape[1]
            tgt_mask = torch.triu(torch.ones(H, H)).bool().transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
            h, t = torch.meshgrid(
                torch.arange(H),
                torch.arange(T),
                indexing='ij'
            )
            memory_mask = h >= (t - 1) 
            memory_mask = memory_mask.float().masked_fill(memory_mask == 0, float('-inf')).masked_fill(memory_mask == 1, float(0.0))
            tgt_mask, memory_mask = tgt_mask.to(tgt.device), memory_mask.to(memory.device)
        else:
            tgt_mask = None
            memory_mask = None
        x = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        x = self.output_norm(x)
        x = self.output_head(x)
        return x