import torch
import torch.nn as nn 
from timm.models.vision_transformer import Block


class ViT_b(nn.Module):
    def __init__(self, ret_layers):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(
                dim=768,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(12)])
        ckpt = torch.load("/opt/data/private/MST_AGI/checkpoints/Meta-Transformer_base_patch16_encoder.pth")
        self.blocks.load_state_dict(ckpt, strict=True)
        
        self.ret_layers = ret_layers
        if self.ret_layers[0] == -1 and len(self.ret_layers) == 1:
            self.ret_layers = [len(self.blocks)-1]
        
    def forward(self, x):
        ret = []
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            if i in self.ret_layers:
                ret.append(x)
        return ret