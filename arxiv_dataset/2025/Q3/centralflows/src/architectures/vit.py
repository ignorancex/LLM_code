from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .utils import replace_layernorm, UntrainableLayerNorm, TrainableLayerNorm

from .base import Architecture 


@dataclass
class VIT(Architecture):
    patch_size: int = 4
    dim: int = 64
    depth: int = 4
    mlp_dim: int = 256
    num_heads: int = 8
    pool: Literal["cls", "mean"] = "cls" 
    replace_ln: bool = True
    trainable_ln: bool = True
    simple: bool = True
    init_scale: float = 0.0 # setting this to 1.0 trains faster but with higher sharpness at init

    def create(self, input_shape, output_dim):
        from vit_pytorch import ViT, SimpleViT

        if self.simple:
            model = SimpleViT(
                image_size=input_shape[-1],
                patch_size = self.patch_size,
                num_classes = output_dim,
                dim = self.dim,
                depth = self.depth,
                heads = self.num_heads,
                mlp_dim = self.mlp_dim
            )
            with torch.no_grad(): 
                model.linear_head.weight.mul_(self.init_scale)
                model.linear_head.bias.mul_(self.init_scale)
            # with torch.no_grad(): 
            #     model.linear_head.weight.zero_()
            #     model.linear_head.bias.zero_()
        else:
            model = ViT(
                image_size=input_shape[-1],
                patch_size=self.patch_size,
                num_classes=output_dim,
                dim=self.dim,
                depth=self.depth,
                heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                pool=self.pool,
                dropout=0.0,
                emb_dropout=0.0,
            )
            with torch.no_grad(): 
                model.mlp_head.weight.mul_(self.init_scale)
                model.mlp_head.bias.mul_(self.init_scale)

        if self.replace_ln:
            replace_layernorm(
                model, TrainableLayerNorm if self.trainable_ln else UntrainableLayerNorm
            )
           


        return model

