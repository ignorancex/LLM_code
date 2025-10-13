#!/usr/bin/env python
# coding=utf-8
"""
M5: Enhanced Dual-Encoder with Advanced Cross-Modal Fusion
@Description: Based on M4 analysis and PanMamba insights, focusing on DS0 with enhanced fusion
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


class CrossModalAttention(nn.Module):
    """Enhanced Cross-Modal Attention inspired by PanMamba"""
    def __init__(self, dim, num_heads, bias):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # MS as query, PAN as key/value (inspired by PanMamba)
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
    """Downsampling module - identical to M4"""
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        # Use strided convolution for downsampling instead of PixelUnshuffle
        self.body = nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Upsampling module - identical to M4"""
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        # Use transposed convolution for upsampling
        self.body = nn.ConvTranspose2d(n_feat, n_feat // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class SimpleFusionModule(nn.Module):
    """Simple fusion module - back to basics with proven strategies"""
    def __init__(self, dim, fusion_type='channel_attention'):
        super(SimpleFusionModule, self).__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'channel_attention':
            # M4 proven channel attention fusion
            self.num_heads = 2
            self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
            self.qkv = nn.Conv2d(dim + 1, dim * 3, kernel_size=1, bias=False)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        elif fusion_type == 'deepfusion_5layers':
            # 5-layer deep fusion (for comparison)
            self.cross_attentions = nn.ModuleList([
                CrossModalAttention(dim, 2, False) for _ in range(5)
            ])
            self.norms = nn.ModuleList([
                LayerNorm(dim, 'WithBias') for _ in range(5)
            ])
        else:
            # Simple 1x1 convolution fusion (baseline)
            self.fusion = nn.Conv2d(dim + 1, dim, kernel_size=1, bias=False)

    def forward(self, ms_feat, pan):
        """
        Args:
            ms_feat: MS features [B, C, H, W]
            pan: PAN image [B, 1, H, W]
        """
        # Resize PAN to match MS feature size
        _, _, h, w = ms_feat.shape
        pan_resized = F.interpolate(pan, size=(h, w), mode='bilinear', align_corners=False)

        if self.fusion_type == 'channel_attention':
            # M4 proven channel attention fusion
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

        else:
            # Simple fusion
            fused_input = torch.cat([ms_feat, pan_resized], dim=1)
            return self.fusion(fused_input)


class AdvancedOutputModule(nn.Module):
    """Advanced output module inspired by refine.py"""
    def __init__(self, dim, num_channels, output_type='enhanced_conv'):
        super(AdvancedOutputModule, self).__init__()
        self.output_type = output_type

        if output_type == 'enhanced_conv':
            # Enhanced convolution output (baseline)
            self.output = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
            )
        elif output_type == 'channel_attention_output':
            # Channel attention enhanced output (keep for reference)
            self.conv_in = nn.Conv2d(dim, dim, 3, stride=1, padding=1)
            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 4, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, dim, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
            self.process = nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 3, stride=1, padding=1)
            )
            self.conv_last = nn.Conv2d(dim, num_channels, 3, stride=1, padding=1)
        elif output_type == 'qkv_channel_attention_output':
            # QKV-based channel attention output (new experiment)
            self.num_heads = 2
            self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

            # QKV projection for channel attention
            self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

            # Final output projection
            self.output_conv = nn.Conv2d(dim, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            # Simple 3x3 convolution output (PanRestormer baseline)
            self.output = nn.Conv2d(dim, num_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def spatial_shift(self, x, shift_pixel=1):
        """Simple spatial shift operation"""
        b, c, h, w = x.shape
        # Split channels into 4 groups for different shift directions
        c_per_group = c // 4

        x1 = x[:, :c_per_group]  # shift left
        x2 = x[:, c_per_group:c_per_group*2]  # shift right
        x3 = x[:, c_per_group*2:c_per_group*3]  # shift up
        x4 = x[:, c_per_group*3:]  # shift down

        # Apply shifts
        x1_shifted = torch.roll(x1, shifts=-shift_pixel, dims=3)
        x2_shifted = torch.roll(x2, shifts=shift_pixel, dims=3)
        x3_shifted = torch.roll(x3, shifts=-shift_pixel, dims=2)
        x4_shifted = torch.roll(x4, shifts=shift_pixel, dims=2)

        return torch.cat([x1_shifted, x2_shifted, x3_shifted, x4_shifted], dim=1)

    def forward(self, x):
        if self.output_type == 'enhanced_conv':
            return self.output(x)

        elif self.output_type == 'channel_attention_output':
            # Channel attention enhanced output
            out = self.conv_in(x)
            res = self.process(out)
            y = self.channel_att(res)
            out = y * res + out
            return self.conv_last(out)

        elif self.output_type == 'qkv_channel_attention_output':
            # QKV-based channel attention output
            b, c, h, w = x.shape

            # Apply QKV channel attention
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

            # Project and add residual
            out = self.project_out(out) + x

            # Final output projection
            return self.output_conv(out)

        else:
            # Simple output
            return self.output(x)


class Net(nn.Module):
    """M5: M4 architecture with only output module replaced"""

    def __init__(self, args=None, num_channels=4, base_filter=None):
        super(Net, self).__init__()

        # Parse arguments with defaults (identical to M4)
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

        # Configuration parameters (identical to M4)
        self.num_channels = getattr(args, 'num_channels', num_channels) if hasattr(args, 'num_channels') else args.get('num_channels', num_channels)
        self.downsample_levels = getattr(args, 'downsample_levels', 0) if hasattr(args, 'downsample_levels') else args.get('downsample_levels', 0)
        self.fusion_type = getattr(args, 'fusion_type', 'conv') if hasattr(args, 'fusion_type') else args.get('fusion_type', 'conv')  # Changed default to conv
        self.output_type = getattr(args, 'output_type', 'enhanced_conv') if hasattr(args, 'output_type') else args.get('output_type', 'enhanced_conv')
        self.model_scale = getattr(args, 'model_scale', 'normal') if hasattr(args, 'model_scale') else args.get('model_scale', 'normal')  # New: model scaling
        self.base_filter = base_filter  # Store base_filter

        # Model configuration (identical to M4)
        # Keep base_filter=64 for controlled variables, only support DS0 and DS2
        if self.downsample_levels not in [0, 2]:
            raise ValueError(f"Only downsample_levels 0 and 2 are supported, got {self.downsample_levels}")

        # Model scaling configuration
        if self.model_scale == 'large_120k':
            # Large model targeting ~120K parameters
            if self.downsample_levels == 0:
                dim = 24  # Moderately increased dimension
                num_blocks = [4]  # More blocks
                num_heads = [3]  # More heads
            else:  # self.downsample_levels == 2
                dim = 18  # Moderately increased dimension for DS2
                num_blocks = [3, 3]  # More blocks
                num_heads = [2, 3]  # More heads
        else:
            # Normal model (identical to M4)
            if self.downsample_levels == 0:
                # No downsampling version - identical to M4 DS0
                dim = 20  # Fixed dimension that works with num_heads=2
                num_blocks = [3]  # More blocks for better performance
                num_heads = [2]
            else:  # self.downsample_levels == 2
                # 2-level downsampling - identical to M4 DS2
                dim = 14  # Smaller fixed dimension for DS2
                num_blocks = [2, 2]  # Moderate blocks for better performance
                num_heads = [2, 2]  # Smaller heads to control parameters

        num_refinement_blocks = 1  # Keep minimal
        ffn_expansion_factor = 2.0  # Reduced from 2.66
        bias = False
        LayerNorm_type = 'WithBias'

        # Dual Input projections (MS and PAN separately) - identical to M4
        self.ms_patch_embed = OverlapPatchEmbed(self.num_channels, dim, bias)
        self.pan_patch_embed = OverlapPatchEmbed(1, dim, bias)

        # MS Encoder - identical to M4
        self.ms_encoder_layers = nn.ModuleList([])
        self.ms_down_layers = nn.ModuleList([])

        # PAN Encoder - identical to M4
        self.pan_encoder_layers = nn.ModuleList([])
        self.pan_down_layers = nn.ModuleList([])

        dim_level = dim
        for i in range(self.downsample_levels):
            # MS encoder layer
            ms_encoder_layer = nn.Sequential(*[
                TransformerBlock(dim=dim_level, num_heads=num_heads[i], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
                for _ in range(num_blocks[i])
            ])
            self.ms_encoder_layers.append(ms_encoder_layer)

            # PAN encoder layer
            pan_encoder_layer = nn.Sequential(*[
                TransformerBlock(dim=dim_level, num_heads=num_heads[i], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
                for _ in range(num_blocks[i])
            ])
            self.pan_encoder_layers.append(pan_encoder_layer)

            if i < self.downsample_levels - 1:
                self.ms_down_layers.append(Downsample(dim_level))
                self.pan_down_layers.append(Downsample(dim_level))
                dim_level = int(dim_level * 2)

        # Bottleneck (latent processing) - identical to M4
        if self.downsample_levels > 0:
            self.latent = nn.Sequential(*[
                TransformerBlock(dim=dim_level * 2, num_heads=num_heads[-1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
                for _ in range(num_blocks[-1])
            ])
        else:
            self.latent = nn.Sequential(*[
                TransformerBlock(dim=dim_level * 2, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
                for _ in range(num_blocks[0])
            ])

        # Dimension reduction after latent processing - identical to M4
        self.latent_fusion = nn.Conv2d(dim_level * 2, dim_level, kernel_size=1, bias=bias)

        # Decoder - identical to M4
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

        # Refinement - identical to M4
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])

        # Fusion module - updated naming and options
        if self.fusion_type == 'conv':
            # Renamed from channel_attention to conv (same implementation)
            self.fusion_module = SimpleFusionModule(dim, 'channel_attention')
        elif self.fusion_type == 'deepfusion_5layers':
            self.fusion_module = SimpleFusionModule(dim, 'deepfusion_5layers')
        else:  # fallback to conv
            self.fusion_module = SimpleFusionModule(dim, 'channel_attention')

        # ONLY DIFFERENCE: Advanced output module instead of simple 3x3 conv
        if self.output_type == 'simple_conv':
            # M4's original output (for comparison)
            self.output_module = nn.Conv2d(dim, self.num_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        else:
            # Advanced output modules
            self.output_module = AdvancedOutputModule(dim, self.num_channels, self.output_type)

    def forward(self, ms, bms, pan, **kwargs):
        """
        Forward pass with dual encoders (identical to M4)
        Args:
            ms: multispectral image (low resolution) - not used directly
            bms: bicubic upsampled multispectral image [B, C, H, W]
            pan: panchromatic image [B, 1, H, W]
        """
        # Separate input embeddings - identical to M4
        ms_feat = self.ms_patch_embed(bms)
        pan_feat = self.pan_patch_embed(pan)

        # Store encoder outputs for skip connections (use MS encoder outputs) - identical to M4
        enc_outputs = []

        # Dual encoding - identical to M4
        for i in range(self.downsample_levels):
            ms_feat = self.ms_encoder_layers[i](ms_feat)
            pan_feat = self.pan_encoder_layers[i](pan_feat)

            # Store MS encoder output for skip connections
            enc_outputs.append(ms_feat)

            if i < self.downsample_levels - 1:
                ms_feat = self.ms_down_layers[i](ms_feat)
                pan_feat = self.pan_down_layers[i](pan_feat)

        # Concat fusion before latent processing - identical to M4
        concat_feat = torch.cat([ms_feat, pan_feat], dim=1)

        # Bottleneck (latent processing) with concat features - identical to M4
        out_enc = self.latent(concat_feat)

        # Reduce dimension after latent processing - identical to M4
        out_enc = self.latent_fusion(out_enc)

        # Decoder with skip connections - identical to M4
        out_dec = out_enc
        for i in range(self.downsample_levels):
            if i > 0:
                out_dec = self.up_layers[i - 1](out_dec)
                out_dec = self.reduce_chan_levels[i - 1](out_dec)
                out_dec = out_dec + enc_outputs[self.downsample_levels - 1 - i]

            out_dec = self.decoder_layers[i](out_dec)

        # Refinement - identical to M4
        out_dec = self.refinement(out_dec)

        # Fusion with PAN - identical to M4
        if hasattr(self, 'fusion_module'):
            # Resize PAN to match feature map size if needed
            _, _, h, w = out_dec.shape
            pan_resized = F.interpolate(pan, size=(h, w), mode='bilinear', align_corners=False)
            out_dec = self.fusion_module(out_dec, pan_resized)

        # ONLY DIFFERENCE: Advanced output module instead of simple 3x3 conv
        if isinstance(self.output_module, nn.Conv2d):
            # Simple conv output (M4 style)
            out_dec = self.output_module(out_dec) + bms
        else:
            # Advanced output module
            out_dec = self.output_module(out_dec) + bms

        return out_dec

    def get_model_info(self):
        """Get model configuration information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'name': f'M5_{self.fusion_type}_{self.output_type}',
            'num_channels': self.num_channels,
            'fusion_type': self.fusion_type,
            'output_type': self.output_type,
            'base_filter': self.base_filter,
            'total_params': total_params,
            'architecture': 'Dual encoder with advanced output module (DS0 only)'
        }


if __name__ == "__main__":
    # Test final M5 configurations for experiment15

    print("M5 Final Configurations for Experiment15:")
    print("=" * 60)

    # Test configurations for experiment15
    test_configs = [
        # Baseline comparison
        {'fusion_type': 'conv', 'output_type': 'enhanced_conv', 'model_scale': 'normal'},

        # Output optimization experiments
        {'fusion_type': 'conv', 'output_type': 'channel_attention_output', 'model_scale': 'normal'},
        {'fusion_type': 'conv', 'output_type': 'qkv_channel_attention_output', 'model_scale': 'normal'},

        # Model scaling experiment
        {'fusion_type': 'conv', 'output_type': 'enhanced_conv', 'model_scale': 'large_120k'},

        # Deep fusion reference
        {'fusion_type': 'deepfusion_5layers', 'output_type': 'enhanced_conv', 'model_scale': 'normal'},
    ]

    for i, config in enumerate(test_configs):
        try:
            args = {'model': config}
            model = Net(args)
            info = model.get_model_info()

            config_name = f"{config['fusion_type']}_{config['output_type']}_{config['model_scale']}"
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

    print("\nM5 Final Design Summary:")
    print("- 4+1 experiments: 1 baseline + 2 output variants + 1 large model + 1 deepfusion")
    print("- Fusion renamed: channel_attention -> conv (same implementation)")
    print("- New QKV channel attention output module")
    print("- Model scaling support for 120K parameter experiments")
    print("- All based on M4 architecture with only specified changes")
