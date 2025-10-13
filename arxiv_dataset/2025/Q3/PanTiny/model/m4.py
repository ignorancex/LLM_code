#!/usr/bin/env python
# coding=utf-8
"""
M4: Dual-Encoder with Enhanced Multi-layer Fusion
@Description: Dual encoder architecture with enhanced fusion layers, optimized for 40-80K parameters
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


class EnhancedFusionModule(nn.Module):
    """Enhanced fusion module with advanced architectures from XRestormer"""
    def __init__(self, dim, fusion_type='gated_dconv'):
        super(EnhancedFusionModule, self).__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'gated_dconv':
            # Gated-Dconv Feed-Forward Network inspired fusion
            hidden_features = int(dim * 2.0)  # Smaller expansion factor
            self.project_in = nn.Conv2d(dim + 1, hidden_features * 2, kernel_size=1, bias=False)
            self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=False)
            self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

        elif fusion_type == 'channel_attention':
            # Channel attention inspired by MDTA
            self.num_heads = 2
            self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
            self.qkv = nn.Conv2d(dim + 1, dim * 3, kernel_size=1, bias=False)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        elif fusion_type == 'spatial_gating':
            # Spatial gating mechanism
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(dim + 1, dim // 4, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, 1, kernel_size=7, padding=3, bias=False),
                nn.Sigmoid()
            )
            self.channel_proj = nn.Conv2d(dim + 1, dim, kernel_size=1, bias=False)

        else:
            # Simple 1x1 convolution fusion (baseline)
            self.fusion_layers = nn.Conv2d(dim + 1, dim, kernel_size=1, bias=False)

    def forward(self, ms_feat, pan):
        """
        Args:
            ms_feat: MS features [B, C, H, W]
            pan: PAN image [B, 1, H, W]
        """
        # Concatenate MS features with PAN
        fused_input = torch.cat([ms_feat, pan], dim=1)

        if self.fusion_type == 'gated_dconv':
            # Gated-Dconv fusion
            x = self.project_in(fused_input)
            x1, x2 = self.dwconv(x).chunk(2, dim=1)
            x = F.gelu(x1) * x2
            x = self.project_out(x)
            return x

        elif self.fusion_type == 'channel_attention':
            # Channel attention fusion
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

        elif self.fusion_type == 'spatial_gating':
            # Spatial gating fusion
            spatial_gate = self.spatial_gate(fused_input)
            channel_feat = self.channel_proj(fused_input)
            return channel_feat * spatial_gate + ms_feat * (1 - spatial_gate)

        else:
            return self.fusion_layers(fused_input)


class Net(nn.Module):
    """M4: Dual-Encoder with Enhanced Multi-layer Fusion"""

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
        self.fusion_type = getattr(args, 'fusion_type', 'gated_dconv') if hasattr(args, 'fusion_type') else args.get('fusion_type', 'gated_dconv')
        self.base_filter = base_filter  # Store base_filter

        # Model configuration optimized for parameter control
        # Keep base_filter=64 for controlled variables, only support DS0 and DS2
        if self.downsample_levels not in [0, 2]:
            raise ValueError(f"Only downsample_levels 0 and 2 are supported, got {self.downsample_levels}")

        if self.downsample_levels == 0:
            # No downsampling version - optimized for ~80K params max
            dim = 20  # Fixed dimension that works with num_heads=2
            num_blocks = [3]  # More blocks for better performance
            num_heads = [2]
        else:  # self.downsample_levels == 2
            # 2-level downsampling - can be larger, up to 200K params
            dim = 14  # Smaller fixed dimension for DS2
            num_blocks = [2, 2]  # Moderate blocks for better performance
            num_heads = [2, 2]  # Smaller heads to control parameters

        num_refinement_blocks = 1  # Keep minimal
        ffn_expansion_factor = 2.0  # Reduced from 2.66
        bias = False
        LayerNorm_type = 'WithBias'

        # Dual Input projections (MS and PAN separately) - smaller dimensions
        self.ms_patch_embed = OverlapPatchEmbed(self.num_channels, dim, bias)
        self.pan_patch_embed = OverlapPatchEmbed(1, dim, bias)

        # MS Encoder - lightweight
        self.ms_encoder_layers = nn.ModuleList([])
        self.ms_down_layers = nn.ModuleList([])

        # PAN Encoder - lightweight
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

        # Bottleneck (latent processing) - process concat features directly
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

        # Dimension reduction after latent processing
        self.latent_fusion = nn.Conv2d(dim_level * 2, dim_level, kernel_size=1, bias=bias)

        # Decoder - identical to PanRestormer
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

        # Refinement - identical to PanRestormer
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])

        # Enhanced Fusion module - key improvement
        self.fusion_module = EnhancedFusionModule(dim, self.fusion_type)

        # Output projection - identical to PanRestormer
        self.output = nn.Conv2d(dim, self.num_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, ms, bms, pan, **kwargs):
        """
        Forward pass with dual encoders
        Args:
            ms: multispectral image (low resolution) - not used directly
            bms: bicubic upsampled multispectral image [B, C, H, W]
            pan: panchromatic image [B, 1, H, W]
        """
        # Separate input embeddings
        ms_feat = self.ms_patch_embed(bms)
        pan_feat = self.pan_patch_embed(pan)

        # Store encoder outputs for skip connections (use MS encoder outputs)
        enc_outputs = []

        # Dual encoding
        for i in range(self.downsample_levels):
            ms_feat = self.ms_encoder_layers[i](ms_feat)
            pan_feat = self.pan_encoder_layers[i](pan_feat)

            # Store MS encoder output for skip connections
            enc_outputs.append(ms_feat)

            if i < self.downsample_levels - 1:
                ms_feat = self.ms_down_layers[i](ms_feat)
                pan_feat = self.pan_down_layers[i](pan_feat)

        # Concat fusion before latent processing (no dimension reduction)
        concat_feat = torch.cat([ms_feat, pan_feat], dim=1)

        # Bottleneck (latent processing) with concat features
        out_enc = self.latent(concat_feat)

        # Reduce dimension after latent processing
        out_enc = self.latent_fusion(out_enc)

        # Decoder with skip connections - identical to PanRestormer
        out_dec = out_enc
        for i in range(self.downsample_levels):
            if i > 0:
                out_dec = self.up_layers[i - 1](out_dec)
                out_dec = self.reduce_chan_levels[i - 1](out_dec)
                out_dec = out_dec + enc_outputs[self.downsample_levels - 1 - i]

            out_dec = self.decoder_layers[i](out_dec)

        # Refinement - identical to PanRestormer
        out_dec = self.refinement(out_dec)

        # Enhanced fusion with PAN - key improvement
        if hasattr(self, 'fusion_module'):
            # Resize PAN to match feature map size if needed
            _, _, h, w = out_dec.shape
            pan_resized = F.interpolate(pan, size=(h, w), mode='bilinear', align_corners=False)
            out_dec = self.fusion_module(out_dec, pan_resized)

        # Output projection - identical to PanRestormer
        out_dec = self.output(out_dec) + bms

        return out_dec

    def get_model_info(self):
        """Get model configuration information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'name': f'M4_DualEncoder_EnhancedFusion_DS{self.downsample_levels}',
            'num_channels': self.num_channels,
            'downsample_levels': self.downsample_levels,
            'fusion_type': self.fusion_type,
            'base_filter': self.base_filter,
            'total_params': total_params,
            'architecture': 'Dual independent encoders with enhanced multi-layer fusion'
        }


if __name__ == "__main__":
    # Test different configurations (only DS0 and DS2 supported)

    # Test no downsampling with different fusion types
    fusion_types = ['gated_dconv', 'channel_attention', 'spatial_gating', 'conv']

    print("DS0 (No downsampling) configurations:")
    for fusion_type in fusion_types:
        try:
            args = {'model': {'num_channels': 4, 'downsample_levels': 0, 'fusion_type': fusion_type}}
            model = Net(args)
            info = model.get_model_info()
            print(f"  {fusion_type}: {info['total_params']/1000:.1f}K params")
        except Exception as e:
            print(f"  {fusion_type}: Failed - {e}")

    print("\nDS2 (2-level downsampling) configurations:")
    for fusion_type in fusion_types:
        try:
            args = {'model': {'num_channels': 4, 'downsample_levels': 2, 'fusion_type': fusion_type}}
            model = Net(args)
            info = model.get_model_info()
            print(f"  {fusion_type}: {info['total_params']/1000:.1f}K params")
        except Exception as e:
            print(f"  {fusion_type}: Failed - {e}")

    # Test with dummy data
    print("\nForward pass test:")
    ms = torch.randn(1, 4, 64, 64)
    bms = torch.randn(1, 4, 128, 128)
    pan = torch.randn(1, 1, 128, 128)

    test_configs = [
        {'downsample_levels': 0, 'fusion_type': 'gated_dconv'},
        {'downsample_levels': 2, 'fusion_type': 'channel_attention'},
    ]

    for config in test_configs:
        try:
            args = {'model': config}
            model = Net(args)
            model.eval()

            with torch.no_grad():
                output = model(ms, bms, pan)

            config_str = f"DS{config['downsample_levels']}_{config['fusion_type']}"
            print(f"  {config_str}: Input {bms.shape} -> Output {output.shape} ✓")

        except Exception as e:
            config_str = f"DS{config['downsample_levels']}_{config['fusion_type']}"
            print(f"  {config_str}: Failed - {e}")
