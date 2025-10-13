#!/usr/bin/env python
# coding=utf-8
"""
M2: Dual-Encoder with Deep Fusion Architecture for pan-sharpening
@Description: Independent MS and PAN encoders with CrossAttention-based deep fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


def to_3d(x):
    """Convert 4D tensor to 3D"""
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    """Convert 3D tensor to 4D"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def pair(x):
    """Ensure tuple format"""
    return (x, x) if not isinstance(x, tuple) else x


class BiasFree_LayerNorm(nn.Module):
    """Bias-free LayerNorm"""
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """LayerNorm with bias"""
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Flexible LayerNorm"""
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network"""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class ChannelAttention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention"""
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class CrossAttention(nn.Module):
    """Cross-Attention for MS-PAN fusion"""
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms, pan):
        b, c, h, w = ms.shape

        kv = self.kv_dwconv(self.kv(pan))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(ms))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """Transformer Block with Channel Attention and FFN"""
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = ChannelAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CrossTransformerBlock(nn.Module):
    """Cross-Transformer Block for MS-PAN interaction"""
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(CrossTransformerBlock, self).__init__()

        self.norm_ms = LayerNorm(dim, LayerNorm_type)
        self.norm_pan = LayerNorm(dim, LayerNorm_type)
        self.cross_attn = CrossAttention(dim, num_heads, bias)
        self.norm_ffn = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, ms, pan):
        # Cross-attention: MS queries PAN
        ms = ms + self.cross_attn(self.norm_ms(ms), self.norm_pan(pan))
        ms = ms + self.ffn(self.norm_ffn(ms))
        return ms


class OverlapPatchEmbed(nn.Module):
    """Overlapped image patch embedding with 3x3 Conv"""
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    """Downsampling module"""
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Upsampling module"""
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.ConvTranspose2d(n_feat, n_feat // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class DeepFusionModule(nn.Module):
    """Deep Fusion Module using CrossAttention for MS-PAN interaction"""
    def __init__(self, dim, num_heads=8, num_blocks=2, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(DeepFusionModule, self).__init__()
        
        self.fusion_blocks = nn.ModuleList([
            CrossTransformerBlock(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks)
        ])
        
        # Final fusion projection
        self.fusion_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms_feat, pan_feat):
        """
        Deep fusion between MS and PAN features
        Args:
            ms_feat: MS features [B, C, H, W]
            pan_feat: PAN features [B, C, H, W]
        Returns:
            fused_feat: Fused features [B, C, H, W]
        """
        fused = ms_feat
        for block in self.fusion_blocks:
            fused = block(fused, pan_feat)
        
        fused = self.fusion_proj(fused)
        return fused


class Net(nn.Module):
    """M2: Dual-Encoder with Deep Fusion Architecture"""
    
    def __init__(self, args=None, num_channels=4, base_filter=None):
        super(Net, self).__init__()
        
        # Parse arguments with defaults
        if args is None:
            args = {}
        args = args.get('model', {}) if isinstance(args, dict) and 'model' in args else args
        
        # Handle different argument formats for compatibility
        if hasattr(args, 'base_filter'):
            base_filter = args.base_filter
        elif hasattr(args, 'get'):
            base_filter = args.get('base_filter', 64)
        elif isinstance(args, dict) and 'base_filter' in args:
            base_filter = args['base_filter']
            
        if base_filter is None:
            base_filter = 64
        
        # Configuration parameters
        self.num_channels = getattr(args, 'num_channels', num_channels) if hasattr(args, 'num_channels') else args.get('num_channels', num_channels)
        self.downsample_levels = getattr(args, 'downsample_levels', 0) if hasattr(args, 'downsample_levels') else args.get('downsample_levels', 0)
        self.base_filter = base_filter
        
        # Model configuration based on downsample levels
        if self.downsample_levels == 0:
            dim = base_filter
            num_heads = [2]
        elif self.downsample_levels == 2:
            dim = base_filter
            num_heads = [1, 2]
        else:
            dim = base_filter
            num_heads = [1, 1, 2, 4]
        
        num_refinement_blocks = 1
        ffn_expansion_factor = 2
        bias = False
        LayerNorm_type = 'WithBias'

        # MS Encoder - Independent path for multispectral
        self.ms_patch_embed = OverlapPatchEmbed(self.num_channels, dim, bias)
        self.ms_encoder_layers = nn.ModuleList([])
        self.ms_down_layers = nn.ModuleList([])
        
        # PAN Encoder - Independent path for panchromatic
        self.pan_patch_embed = OverlapPatchEmbed(1, dim, bias)
        self.pan_encoder_layers = nn.ModuleList([])
        self.pan_down_layers = nn.ModuleList([])
        
        # Build independent encoders
        dim_level = dim
        for i in range(self.downsample_levels):
            self.ms_encoder_layers.append(nn.Sequential(*[
                TransformerBlock(dim=dim_level, num_heads=num_heads[i], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(2)
            ]))
            
            self.pan_encoder_layers.append(nn.Sequential(*[
                TransformerBlock(dim=dim_level, num_heads=num_heads[i], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(2)
            ]))
            
            if i < self.downsample_levels - 1:
                self.ms_down_layers.append(Downsample(dim_level))
                self.pan_down_layers.append(Downsample(dim_level))
                dim_level = dim_level * 2

        # Deep Fusion Module
        self.deep_fusion = DeepFusionModule(
            dim=dim_level, 
            num_heads=num_heads[-1], 
            num_blocks=3,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])
        self.reduce_chan_levels = nn.ModuleList([])
        
        for i in range(self.downsample_levels):
            if i > 0:
                self.up_layers.append(Upsample(dim_level))
                dim_level = dim_level // 2
                self.reduce_chan_levels.append(nn.Conv2d(dim_level * 2, dim_level, kernel_size=1, bias=bias))
            
            self.decoder_layers.append(nn.Sequential(*[
                TransformerBlock(dim=dim_level, num_heads=num_heads[self.downsample_levels - 1 - i], 
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(2)
            ]))

        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type) 
            for _ in range(num_refinement_blocks)
        ])

        # Output projection
        self.output = nn.Conv2d(dim, self.num_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, ms, bms, pan, **kwargs):
        """
        Forward pass with dual encoders and deep fusion
        Args:
            ms: multispectral image (low resolution) - not used directly
            bms: bicubic upsampled multispectral image [B, C, H, W]
            pan: panchromatic image [B, 1, H, W]
        """
        # Independent embedding for MS and PAN
        ms_feat = self.ms_patch_embed(bms)
        pan_feat = self.pan_patch_embed(pan)
        
        # Store encoder outputs for skip connections
        ms_enc_outputs = []
        pan_enc_outputs = []
        
        # Independent encoding
        for i in range(self.downsample_levels):
            ms_feat = self.ms_encoder_layers[i](ms_feat)
            pan_feat = self.pan_encoder_layers[i](pan_feat)
            
            ms_enc_outputs.append(ms_feat)
            pan_enc_outputs.append(pan_feat)
            
            if i < self.downsample_levels - 1:
                ms_feat = self.ms_down_layers[i](ms_feat)
                pan_feat = self.pan_down_layers[i](pan_feat)

        # Deep Fusion at bottleneck
        fused_feat = self.deep_fusion(ms_feat, pan_feat)

        # Decoder with skip connections
        out_dec = fused_feat
        for i in range(self.downsample_levels):
            if i > 0:
                out_dec = self.up_layers[i-1](out_dec)
                # Skip connection from MS encoder
                skip_feat = ms_enc_outputs[self.downsample_levels - 1 - i]
                out_dec = torch.cat([out_dec, skip_feat], dim=1)
                out_dec = self.reduce_chan_levels[i-1](out_dec)
            
            out_dec = self.decoder_layers[i](out_dec)

        # Refinement
        out_dec = self.refinement(out_dec)

        # Output projection with residual connection
        out_dec = self.output(out_dec) + bms

        return out_dec

    def get_model_info(self):
        """Get model configuration information"""
        return {
            'name': 'M2_DualEncoder_DeepFusion',
            'num_channels': self.num_channels,
            'base_filter': self.base_filter,
            'downsample_levels': self.downsample_levels,
            'architecture': 'Dual independent encoders with deep fusion',
            'fusion_type': 'CrossAttention-based deep fusion'
        }


if __name__ == "__main__":
    # Test different configurations
    
    # Test no downsampling
    args_no_ds = {'num_channels': 4, 'downsample_levels': 0, 'base_filter': 48}
    model_no_ds = Net(args_no_ds)
    print(f"No downsample model info: {model_no_ds.get_model_info()}")
    
    # Test 2-level downsampling
    args_2ds = {'num_channels': 4, 'downsample_levels': 2, 'base_filter': 48}
    model_2ds = Net(args_2ds)
    print(f"2-level downsample model info: {model_2ds.get_model_info()}")
    
    # Test 4-level downsampling (default)
    args_4ds = {'num_channels': 4, 'downsample_levels': 4, 'base_filter': 48}
    model_4ds = Net(args_4ds)
    print(f"4-level downsample model info: {model_4ds.get_model_info()}")
    
    # Test with dummy data
    ms = torch.randn(1, 4, 64, 64)
    bms = torch.randn(1, 4, 256, 256)
    pan = torch.randn(1, 1, 256, 256)
    
    with torch.no_grad():
        try:
            output = model_4ds(ms, bms, pan)
            print(f"Output shape: {output.shape}")
            print("Model test passed!")
        except Exception as e:
            print(f"Model test failed: {e}")