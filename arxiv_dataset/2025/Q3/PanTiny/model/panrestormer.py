#!/usr/bin/env python
# coding=utf-8
"""
PanRestormer: XRestormer-based architecture for pan-sharpening, as one encoder PanTiny
@Description: Transformer-based architecture with configurable downsampling levels
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from torch import einsum


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


class OverlapPatchEmbed(nn.Module):
    """Overlapped image patch embedding with 3x3 Conv"""
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    """Downsampling module - fixed for correct channel handling"""
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        # Use strided convolution for downsampling instead of PixelUnshuffle
        self.body = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Upsampling module - fixed for correct channel handling"""
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        # Use transposed convolution for upsampling
        self.body = nn.ConvTranspose2d(n_feat, n_feat // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class PanFusionModule(nn.Module):
    """Simple pan-sharpening fusion module"""
    def __init__(self, dim, fusion_type='conv'):
        super(PanFusionModule, self).__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'conv':
            # Simple 1x1 convolution fusion (similar to TextIF)
            self.fusion = nn.Conv2d(dim + 1, dim, kernel_size=1, bias=False)
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attention = nn.Sequential(
                nn.Conv2d(dim + 1, dim // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, 1, kernel_size=1),
                nn.Sigmoid()
            )
            self.fusion = nn.Conv2d(dim + 1, dim, kernel_size=1, bias=False)
        else:
            # Simple concatenation
            self.fusion = nn.Conv2d(dim + 1, dim, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, ms_feat, pan):
        """
        Args:
            ms_feat: MS features [B, C, H, W]
            pan: PAN image [B, 1, H, W]
        """
        # Concatenate MS features with PAN
        fused_input = torch.cat([ms_feat, pan], dim=1)
        
        if self.fusion_type == 'attention':
            # Use attention weights
            attention_weights = self.attention(fused_input)
            fused_feat = self.fusion(fused_input)
            return fused_feat * attention_weights + ms_feat * (1 - attention_weights)
        else:
            return self.fusion(fused_input)


class Net(nn.Module):
    """PanRestormer: XRestormer-based pan-sharpening network"""
    
    def __init__(self, args=None, num_channels=4, base_filter=None):
        super(Net, self).__init__()
        
        # Parse arguments with defaults
        if args is None:
            args = {}
        args = args.get('model', {}) # special
        
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
        self.downsample_levels = getattr(args, 'downsample_levels', 4) if hasattr(args, 'downsample_levels') else args.get('downsample_levels', 4)
        self.fusion_type = getattr(args, 'fusion_type', 'conv') if hasattr(args, 'fusion_type') else args.get('fusion_type', 'conv')
        self.base_filter = base_filter  # Store base_filter
        
        # Model configuration based on downsample levels - adjusted for reasonable parameter count
        if self.downsample_levels == 0:
            # No downsampling version
            dim = base_filter // 2  # Use base_filter//2 instead of base_filter
            num_blocks = [2]  # Reduced for reasonable parameter count
            num_heads = [2]
        elif self.downsample_levels == 2:
            # 2-level downsampling
            dim = base_filter // 4  # Much smaller start
            num_blocks = [2, 2]  # Reduced
            num_heads = [1, 2]
        else:
            # Default 4-level downsampling - much more conservative
            dim = base_filter // 8  # Very small start for 4-level
            num_blocks = [1, 1, 2, 2]  # Much reduced from [1, 2, 2, 2]
            num_heads = [1, 1, 2, 4]  # Much reduced from [1, 2, 4, 8]
        
        num_refinement_blocks = 1  # Much reduced from 2
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'

        # Input projection
        self.patch_embed = OverlapPatchEmbed(self.num_channels + 1, dim, bias)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        self.down_layers = nn.ModuleList([])
        
        dim_level = dim
        for i in range(self.downsample_levels):
            encoder_layer = nn.Sequential(*[
                TransformerBlock(dim=dim_level, num_heads=num_heads[i], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) 
                for _ in range(num_blocks[i])
            ])
            self.encoder_layers.append(encoder_layer)
            
            if i < self.downsample_levels - 1:
                self.down_layers.append(Downsample(dim_level))
                dim_level = int(dim_level * 2)

        # Bottleneck (latent processing)
        if self.downsample_levels > 0:
            self.latent = nn.Sequential(*[
                TransformerBlock(dim=dim_level, num_heads=num_heads[-1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) 
                for _ in range(num_blocks[-1])
            ])
        else:
            self.latent = nn.Sequential(*[
                TransformerBlock(dim=dim_level, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) 
                for _ in range(num_blocks[0])
            ])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])
        self.reduce_chan_levels = nn.ModuleList([])
        
        for i in range(self.downsample_levels):
            if i > 0:
                self.up_layers.append(Upsample(dim_level))
                self.reduce_chan_levels.append(nn.Conv2d(dim_level // 2, dim_level // 2, kernel_size=1, bias=bias))
                dim_level = dim_level // 2

            decoder_layer = nn.Sequential(*[
                TransformerBlock(dim=dim_level, num_heads=num_heads[self.downsample_levels - 1 - i], 
                               ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) 
                for _ in range(num_blocks[self.downsample_levels - 1 - i])
            ])
            self.decoder_layers.append(decoder_layer)

        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type) 
            for _ in range(num_refinement_blocks)
        ])

        # Fusion module 
        self.fusion_module = PanFusionModule(dim, self.fusion_type)

        # Output projection
        self.output = nn.Conv2d(dim, self.num_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, ms, bms, pan, **kwargs):
        """
        Forward pass
        Args:
            ms: multispectral image (low resolution) - not used directly
            bms: bicubic upsampled multispectral image [B, C, H, W]
            pan: panchromatic image [B, 1, H, W]
        """
        # Initial concatenation of bms and pan
        inp_enc_level0 = torch.cat([bms, pan], dim=1)
        
        # Input embedding
        inp_enc_level0 = self.patch_embed(inp_enc_level0)
        
        # Store encoder outputs for skip connections
        enc_outputs = []
        
        # Encoder
        out_enc = inp_enc_level0
        for i in range(self.downsample_levels):
            out_enc = self.encoder_layers[i](out_enc)
            enc_outputs.append(out_enc)
            if i < self.downsample_levels - 1:
                out_enc = self.down_layers[i](out_enc)

        # Bottleneck
        out_enc = self.latent(out_enc)

        # Decoder with skip connections
        out_dec = out_enc
        for i in range(self.downsample_levels):
            if i > 0:
                out_dec = self.up_layers[i - 1](out_dec)
                out_dec = self.reduce_chan_levels[i - 1](out_dec)
                out_dec = out_dec + enc_outputs[self.downsample_levels - 1 - i]
            
            out_dec = self.decoder_layers[i](out_dec)

        # Refinement
        out_dec = self.refinement(out_dec)

        # Additional fusion with PAN if needed
        if hasattr(self, 'fusion_module'):
            # Resize PAN to match feature map size if needed
            _, _, h, w = out_dec.shape
            pan_resized = F.interpolate(pan, size=(h, w), mode='bilinear', align_corners=False)
            out_dec = self.fusion_module(out_dec, pan_resized)

        # Output projection
        out_dec = self.output(out_dec) + bms

        return out_dec

    def get_model_info(self):
        """Get model configuration information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'name': f'PanRestormer_DS{self.downsample_levels}',
            'num_channels': self.num_channels,
            'downsample_levels': self.downsample_levels,
            'fusion_type': self.fusion_type,
            'total_params': total_params
        }


if __name__ == "__main__":
    # Test different configurations
    
    # Test no downsampling
    args_no_ds = {'num_channels': 4, 'downsample_levels': 0, 'fusion_type': 'conv'}
    model_no_ds = Net(args_no_ds)
    print(f"No downsample model info: {model_no_ds.get_model_info()}")
    
    # Test 2-level downsampling
    args_2ds = {'num_channels': 4, 'downsample_levels': 2, 'fusion_type': 'conv'}
    model_2ds = Net(args_2ds)
    print(f"2-level downsample model info: {model_2ds.get_model_info()}")
    
    # Test 4-level downsampling (default)
    args_4ds = {'num_channels': 4, 'downsample_levels': 4, 'fusion_type': 'conv'}
    model_4ds = Net(args_4ds)
    print(f"4-level downsample model info: {model_4ds.get_model_info()}")
    
    # Test with dummy data
    ms = torch.randn(1, 4, 64, 64)
    bms = torch.randn(1, 4, 256, 256)
    pan = torch.randn(1, 1, 256, 256)
    
    with torch.no_grad():
        out = model_2ds(ms, bms, pan)
        print(f"Output shape: {out.shape}")
