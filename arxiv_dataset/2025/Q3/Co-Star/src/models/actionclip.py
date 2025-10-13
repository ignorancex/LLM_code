# import torch
import torch.nn as nn
from einops import rearrange

# MLP class for the feedforward network
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or (in_features // 2)  # Further reduced MLP hidden size
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Attention class implementing factorized space-time attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, attn_drop=0.2, proj_drop=0.2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Factorized Attention Block that supports divided space-time attention
class FactorizedAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.5, qkv_bias=False, drop=0.1, attn_drop=0.1, drop_path=0.1, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = nn.LayerNorm(dim)
            self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)
            nn.init.constant_(self.temporal_fc.weight, 0)
            nn.init.constant_(self.temporal_fc.bias, 0)

        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type == 'divided_space_time':
            xt = x[:, 1:, :]  # Skip CLS token
            xt = rearrange(xt, '(b t) hw m -> (b hw) t m', b=B, t=T)
            xt = self.temporal_attn(self.temporal_norm1(xt))
            xt = rearrange(xt, '(b hw) t m -> (b t) hw m', b=B, t=T)
            xt = self.temporal_fc(xt)
            x[:, 1:, :] += xt

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Modified CLIP vision model with factorized space-time attention
class ModifiedCLIPVisionModel(nn.Module):
    def __init__(self, vision_model, num_frames=16, num_classes=8, attention_type='divided_space_time', drop=0.1, attn_drop=0.1, drop_path=0.1):
        super().__init__()
        self.vision_model = vision_model
        self.num_frames = num_frames
        self.attention_type = attention_type
        self.num_classes = num_classes

        hidden_size = vision_model.config.hidden_size

        # Use only the first 6 layers to reduce parameters
        self.blocks = nn.ModuleList([
            FactorizedAttentionBlock(
                dim=hidden_size,
                num_heads=2,  # Reduced number of heads
                mlp_ratio=1.5,  # Reduced MLP ratio
                qkv_bias=True,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                attention_type=self.attention_type
            ) for block in vision_model.encoder.layers[:6]  # Use only the first 6 layers
        ])

        self.norm = nn.LayerNorm(hidden_size)

        # Global Average Pooling layer
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        B, T, C, H, W = pixel_values.shape

        x = rearrange(pixel_values, 'b t c h w -> (b t) c h w')

        x = self.vision_model.embeddings(x)
        x = self.vision_model.pre_layrnorm(x)

        for blk in self.blocks:
            x = blk(x, B, T, W)

        x = self.norm(x)
        x = x.mean(dim=1)

        x = rearrange(x, '(b t) m -> b m t', b=B, t=T)
        x = self.gap(x)

        x = x.squeeze(dim=-1)
        x = self.fc(x)
        return x
