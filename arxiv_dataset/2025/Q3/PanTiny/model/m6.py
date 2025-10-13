#!/usr/bin/env python
# coding=utf-8
"""
M6: Enhanced fusion strategies with gated conv and deepfusion v2
@Description: Based on M4/M5 experiments, focusing on effective fusion strategies
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


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention for deepfusion"""
    def __init__(self, dim, num_heads, bias):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # MS as query, PAN as key/value
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.kv_proj = nn.Conv2d(1, dim * 2, kernel_size=1, bias=bias)  # PAN input
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms_feat, pan):
        """
        Args:
            ms_feat: MS features [B, C, H, W]
            pan: PAN image [B, 1, H, W]
        """
        b, c, h, w = ms_feat.shape

        # MS as query
        q = self.q_dwconv(self.q_proj(ms_feat))
        
        # PAN as key/value
        kv = self.kv_dwconv(self.kv_proj(pan))
        k, v = kv.chunk(2, dim=1)

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


class GatedConvFusion(nn.Module):
    """Gated convolution fusion inspired by gated_dconv"""
    def __init__(self, dim, bias=False):
        super(GatedConvFusion, self).__init__()
        
        # Gated convolution for fusion
        self.project_in = nn.Conv2d(dim + 1, dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms_feat, pan):
        """
        Args:
            ms_feat: MS features [B, C, H, W]
            pan: PAN image [B, 1, H, W]
        """
        # Resize PAN to match MS feature size
        _, _, h, w = ms_feat.shape
        pan_resized = F.interpolate(pan, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concatenate and apply gated convolution
        fused_input = torch.cat([ms_feat, pan_resized], dim=1)
        x = self.project_in(fused_input)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        
        return x


class M6FusionModule(nn.Module):
    """M6 Fusion Module with multiple strategies"""
    def __init__(self, dim, fusion_type='gated_conv', bias=False):
        super(M6FusionModule, self).__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'gated_conv':
            # Gated convolution fusion (proven effective)
            self.fusion = GatedConvFusion(dim, bias)
        elif fusion_type == 'channel_attention':
            # Channel attention fusion (M4 proven)
            self.num_heads = 2
            self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
            self.qkv = nn.Conv2d(dim + 1, dim * 3, kernel_size=1, bias=bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        elif fusion_type == 'deepfusion_5layers':
            # 5-layer deep fusion (M5 proven)
            self.cross_attentions = nn.ModuleList([
                CrossModalAttention(dim, 2, bias) for _ in range(5)
            ])
            self.norms = nn.ModuleList([
                LayerNorm(dim, 'WithBias') for _ in range(5)
            ])
        elif fusion_type == 'deepfusion_v2':
            # NEW: Deepfusion v2 with gated conv in each layer
            self.gated_fusions = nn.ModuleList([
                GatedConvFusion(dim, bias) for _ in range(5)
            ])
            self.norms = nn.ModuleList([
                LayerNorm(dim, 'WithBias') for _ in range(5)
            ])
        else:
            # Simple 1x1 convolution fusion (baseline)
            self.fusion = nn.Conv2d(dim + 1, dim, kernel_size=1, bias=bias)

    def forward(self, ms_feat, pan):
        """
        Args:
            ms_feat: MS features [B, C, H, W]
            pan: PAN image [B, 1, H, W]
        """
        # Resize PAN to match MS feature size
        _, _, h, w = ms_feat.shape
        pan_resized = F.interpolate(pan, size=(h, w), mode='bilinear', align_corners=False)

        if self.fusion_type == 'gated_conv':
            return self.fusion(ms_feat, pan_resized)

        elif self.fusion_type == 'channel_attention':
            # Channel attention fusion
            fused_input = torch.cat([ms_feat, pan_resized], dim=1)
            b, c, h, w = fused_input.shape
            qkv = self.qkv_dwconv(self.qkv(fused_input))
            q, k, v = qkv.chunk(3, dim=1)

            # Reduce to target dimension for attention
            target_dim = ms_feat.shape[1]
            q = q[:, :target_dim]
            k = k[:, :target_dim]
            v = v[:, :target_dim]

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

        elif self.fusion_type == 'deepfusion_5layers':
            # 5-layer deep fusion
            result = ms_feat
            for i in range(5):
                cross_feat = self.cross_attentions[i](result, pan_resized)
                result = result + self.norms[i](cross_feat)
            return result

        elif self.fusion_type == 'deepfusion_v2':
            # NEW: Deepfusion v2 with gated conv
            result = ms_feat
            for i in range(5):
                gated_feat = self.gated_fusions[i](result, pan_resized)
                result = result + self.norms[i](gated_feat)
            return result

        else:
            # Simple fusion
            fused_input = torch.cat([ms_feat, pan_resized], dim=1)
            return self.fusion(fused_input)


class Net(nn.Module):
    """M6: Enhanced fusion strategies with gated conv and deepfusion v2"""

    def __init__(self, args=None, num_channels=4, base_filter=None):
        super(Net, self).__init__()

        # Parse arguments with defaults (identical to PanRestormer)
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
        self.downsample_levels = getattr(args, 'downsample_levels', 0) if hasattr(args, 'downsample_levels') else args.get('downsample_levels', 0)
        self.fusion_type = getattr(args, 'fusion_type', 'gated_conv') if hasattr(args, 'fusion_type') else args.get('fusion_type', 'gated_conv')
        self.model_scale = getattr(args, 'model_scale', 'normal') if hasattr(args, 'model_scale') else args.get('model_scale', 'normal')
        self.base_filter = base_filter  # Store base_filter

        # Model configuration aligned with PanRestormer
        # Only support DS0 (no downsampling) based on experiment results
        if self.downsample_levels != 0:
            raise ValueError(f"M6 only supports downsample_levels=0, got {self.downsample_levels}")

        # Model scaling configuration
        if self.model_scale == 'large_200k':
            # Large model targeting ~200K parameters
            dim = 38  # Adjusted to reach ~200K
            num_blocks = [4]  # Moderate blocks
            num_heads = [4]  # Moderate heads
            num_refinement_blocks = 4
        else:
            # Normal model targeting ~60K parameters (with deepfusion_v2)
            dim = 28  # Increased to reach 60K with dpv2
            num_blocks = [3]  # More blocks
            num_heads = [2]  # Moderate heads
            num_refinement_blocks = 3  # More refinement blocks

        ffn_expansion_factor = 2.0  # Reduced from 2.66
        bias = False
        LayerNorm_type = 'WithBias'

        # Input projections - identical to PanRestormer
        self.patch_embed = OverlapPatchEmbed(self.num_channels, dim, bias)

        # Encoder - single level (DS0 only)
        self.encoder = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        # Refinement - identical to PanRestormer
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])

        # M6 Fusion module - key innovation
        self.fusion_module = M6FusionModule(dim, self.fusion_type, bias)

        # Output projection - identical to PanRestormer
        self.output = nn.Conv2d(dim, self.num_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, ms, bms, pan, **kwargs):
        """
        Forward pass with M6 fusion strategies
        Args:
            ms: multispectral image (low resolution) - not used directly
            bms: bicubic upsampled multispectral image [B, C, H, W]
            pan: panchromatic image [B, 1, H, W]
        """
        # Input embedding - identical to PanRestormer
        inp_enc = self.patch_embed(bms)

        # Encoder - identical to PanRestormer
        out_enc = self.encoder(inp_enc)

        # M6 Fusion with PAN - key innovation
        out_enc = self.fusion_module(out_enc, pan)

        # Refinement - identical to PanRestormer
        out_dec = self.refinement(out_enc)

        # Output projection - identical to PanRestormer
        out_dec = self.output(out_dec) + bms

        return out_dec

    def get_model_info(self):
        """Get model configuration information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'name': f'M6_{self.fusion_type}_{self.model_scale}',
            'num_channels': self.num_channels,
            'fusion_type': self.fusion_type,
            'model_scale': self.model_scale,
            'base_filter': self.base_filter,
            'total_params': total_params,
            'architecture': 'PanRestormer + M6 fusion strategies (DS0 only)'
        }


if __name__ == "__main__":
    # Test M6 configurations

    print("M6 Model Parameter Analysis:")
    print("=" * 60)

    # Test configurations for M6
    test_configs = [
        # Normal scale models (~60K baseline + fusion overhead)
        {'fusion_type': 'conv', 'model_scale': 'normal'},  # Baseline (1x1 conv)
        {'fusion_type': 'gated_conv', 'model_scale': 'normal'},  # Gated conv
        {'fusion_type': 'channel_attention', 'model_scale': 'normal'},  # Channel attention
        {'fusion_type': 'deepfusion_5layers', 'model_scale': 'normal'},  # Deepfusion 5 layers
        {'fusion_type': 'deepfusion_v2', 'model_scale': 'normal'},  # NEW: Deepfusion v2

        # Large scale models (~200K)
        {'fusion_type': 'deepfusion_v2', 'model_scale': 'large_200k'},  # Large deepfusion v2
    ]

    for i, config in enumerate(test_configs):
        try:
            args = {'model': config}
            model = Net(args)
            info = model.get_model_info()

            config_name = f"{config['fusion_type']}_{config['model_scale']}"
            print(f"  Config {i+1} ({config_name}): {info['total_params']/1000:.1f}K params")

        except Exception as e:
            print(f"  Config {i+1}: Failed - {e}")

    # Test forward pass
    print("\nForward pass test:")
    ms = torch.randn(1, 4, 64, 64)
    bms = torch.randn(1, 4, 128, 128)
    pan = torch.randn(1, 1, 128, 128)

    for i, config in enumerate(test_configs[:3]):  # Test first 3 configs
        try:
            args = {'model': config}
            model = Net(args)
            model.eval()

            with torch.no_grad():
                output = model(ms, bms, pan)

            config_str = f"Config{i+1}"
            print(f"✓ {config_str}: Input {bms.shape} -> Output {output.shape}")

        except Exception as e:
            config_str = f"Config{i+1}"
            print(f"✗ {config_str}: Failed - {e}")

    print("\nM6 Design Summary:")
    print("- 4 fusion strategies: gated_conv, channel_attention, deepfusion_5layers, deepfusion_v2")
    print("- 2 model scales: normal (~60K), large_200k (~200K)")
    print("- NEW: deepfusion_v2 with gated conv in each layer")
    print("- Aligned with PanRestormer architecture")
    print("- Target: baseline+dpv2 ~60K, large_200k+dpv2 ~200K")
