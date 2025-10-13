"""
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab
"""
import math
import torch
import torch.nn as nn
from torch import Tensor

# Adapted from https://github.com/HenryWJL/diffusion_policy/blob/main/diffusion_policy/model/diffusion/positional_embedding.py#L5
class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        half_dim = dim // 2
        pos_embed = torch.exp(torch.arange(half_dim) * -math.log(10000) / (half_dim - 1))
        self.register_buffer('pos_embed', pos_embed)

    def forward(self, x: Tensor) -> Tensor:
        pos_embed = x.unsqueeze(-1) * self.pos_embed.unsqueeze(0)
        pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
        return pos_embed