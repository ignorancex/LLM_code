#!/usr/bin/env python
# coding=utf-8
"""
PanTiny: Lightweight Pan-sharpening Transformer with Efficient Fusion
@Description: Compact yet effective model with multiple fusion strategies for ablation studies
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


class GatedConvFusion(nn.Module):
    """Gated convolution fusion - proven effective"""
    def __init__(self, dim, bias=False):
        super(GatedConvFusion, self).__init__()
        self.project_in = nn.Conv2d(dim + 1, dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms_feat, pan):
        _, _, h, w = ms_feat.shape
        pan_resized = F.interpolate(pan, size=(h, w), mode='bilinear', align_corners=False)
        fused_input = torch.cat([ms_feat, pan_resized], dim=1)
        x = self.project_in(fused_input)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention for deepfusion"""
    def __init__(self, dim, num_heads, bias):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.kv_proj = nn.Conv2d(1, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, ms_feat, pan):
        b, c, h, w = ms_feat.shape
        q = self.q_dwconv(self.q_proj(ms_feat))
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


class RefinementModule(nn.Module):
    """True refinement module with multiple strategies"""
    def __init__(self, dim, refine_type='enhanced_conv', bias=False):
        super(RefinementModule, self).__init__()
        self.refine_type = refine_type
        
        if refine_type == 'enhanced_conv':
            # Enhanced convolution refinement
            self.refine = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
            )
        elif refine_type == 'channel_attention':
            # Channel attention refinement
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
        elif refine_type == 'residual_blocks':
            # Residual blocks refinement
            self.conv_in = nn.Conv2d(dim, dim, 3, stride=1, padding=1)
            self.residual_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim, dim, 3, stride=1, padding=1)
                ) for _ in range(3)
            ])
        else:
            # Simple convolution
            self.refine = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
    
    def forward(self, x):
        if self.refine_type == 'enhanced_conv':
            return self.refine(x)
        elif self.refine_type == 'channel_attention':
            out = self.conv_in(x)
            res = self.process(out)
            y = self.channel_att(res)
            out = y * res + out
            return out
        elif self.refine_type == 'residual_blocks':
            out = self.conv_in(x)
            for block in self.residual_blocks:
                res = block(out)
                out = out + res
            return out
        else:
            return self.refine(x)


class PanTinyFusionModule(nn.Module):
    """PanTiny Fusion Module with multiple strategies for ablation"""
    def __init__(self, dim, fusion_type='enhanced_conv', bias=False):
        super(PanTinyFusionModule, self).__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'gated_conv':
            # Gated convolution fusion (proven effective)
            self.fusion = GatedConvFusion(dim, bias)
        elif fusion_type == 'channel_attention':
            # Channel attention fusion (M4/M6 proven)
            self.num_heads = 2
            self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
            self.qkv = nn.Conv2d(dim + 1, dim * 3, kernel_size=1, bias=bias)
            self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
            self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        elif fusion_type == 'deepfusion_5layers':
            # 5-layer deep fusion (M5/M6 proven)
            self.cross_attentions = nn.ModuleList([
                CrossModalAttention(dim, 2, bias) for _ in range(5)
            ])
            self.norms = nn.ModuleList([
                LayerNorm(dim, 'WithBias') for _ in range(5)
            ])
        elif fusion_type == 'deepfusion_v2':
            # Deepfusion v2 with gated conv in each layer
            self.gated_fusions = nn.ModuleList([
                GatedConvFusion(dim, bias) for _ in range(5)
            ])
            self.norms = nn.ModuleList([
                LayerNorm(dim, 'WithBias') for _ in range(5)
            ])
        elif fusion_type == 'enhanced_conv':
            # Enhanced convolution fusion (dual-layer conv)
            self.fusion = nn.Sequential(
                nn.Conv2d(dim + 1, dim, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
            )
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
        # pan_resized = pan # the above line is not needed, but I keep it to avoid potential bugs

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
            # Deepfusion v2 with gated conv
            result = ms_feat
            for i in range(5):
                gated_feat = self.gated_fusions[i](result, pan_resized)
                result = result + self.norms[i](gated_feat)
            return result

        elif self.fusion_type == 'enhanced_conv':
            # Enhanced convolution fusion
            fused_input = torch.cat([ms_feat, pan_resized], dim=1)
            return self.fusion(fused_input)

        else:
            # Simple fusion
            fused_input = torch.cat([ms_feat, pan_resized], dim=1)
            return self.fusion(fused_input)


class Net(nn.Module):
    """PanTiny: Lightweight Pan-sharpening Transformer with Efficient Fusion"""

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
        self.downsample_levels = getattr(args, 'downsample_levels', 0) if hasattr(args, 'downsample_levels') else args.get('downsample_levels', 0)
        self.fusion_type = getattr(args, 'fusion_type', 'enhanced_conv') if hasattr(args, 'fusion_type') else args.get('fusion_type', 'enhanced_conv')
        self.refine_type = getattr(args, 'refine_type', 'simple') if hasattr(args, 'refine_type') else args.get('refine_type', 'simple')
        self.model_scale = getattr(args, 'model_scale', 'normal') if hasattr(args, 'model_scale') else args.get('model_scale', 'normal')
        self.base_filter = base_filter  # Store base_filter

        # Model configuration aligned with PanRestormer
        # Only support DS0 (no downsampling) based on experiment results
        if self.downsample_levels != 0:
            raise ValueError(f"PanTiny only supports downsample_levels=0, got {self.downsample_levels}")

        # Model scaling configuration
        if self.model_scale == 'large':
            # Large model targeting ~200K parameters
            dim = 40  # Must be divisible by num_heads (40 % 4 = 0)
            num_blocks = [4]  # Moderate blocks, just for encoder
            num_heads = [4]  # Moderate heads
            num_refinement_blocks = 4 # the number of blocks in latent module
        else:
            # Normal model targeting ~80K parameters
            dim = 28  # Adjusted for ~80K
            num_blocks = [3]  # Moderate blocks
            num_heads = [2]  # Moderate heads
            num_refinement_blocks = 3

        ffn_expansion_factor = 2.0  # Efficient expansion
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

        # PanTiny Fusion module - key innovation
        self.fusion_module = PanTinyFusionModule(dim, self.fusion_type, bias)

        # The Latent module. Name is misleading, not change this if you want to load pretrained weights
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads[0], ffn_expansion_factor=ffn_expansion_factor,
                           bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])

        # True refinement module in ablation
        self.advanced_refine = RefinementModule(dim, self.refine_type, bias)

        # Output projection - identical to PanRestormer
        self.output = nn.Conv2d(dim, self.num_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, ms, bms, pan, **kwargs):
        """
        Forward pass with PanTiny fusion strategies
        Args:
            ms: multispectral image (low resolution) - not used directly
            bms: bicubic upsampled multispectral image [B, C, H, W]
            pan: panchromatic image [B, 1, H, W]
        """
        # Input embedding
        inp_enc = self.patch_embed(bms)

        # Encoder
        out_enc = self.encoder(inp_enc)

        # PanTiny Fusion with PAN - key innovation
        out_enc = self.fusion_module(out_enc, pan)

        # Latent as the body module. The name is misleading, do not change this if you want to load pretrained weights
        out_dec = self.refinement(out_enc)

        # True refinement (in ablation). 
        # Due to the `refinement` name is used, I write `advanced_refine`
        out_dec = self.advanced_refine(out_dec)

        # Output projection
        out_dec = self.output(out_dec) + bms

        return out_dec

    def get_model_info(self):
        """Get model configuration information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'name': f'PanTiny_{self.fusion_type}_{self.refine_type}_{self.model_scale}',
            'num_channels': self.num_channels,
            'fusion_type': self.fusion_type,
            'refine_type': self.refine_type,
            'model_scale': self.model_scale,
            'base_filter': self.base_filter,
            'total_params': total_params,
            'architecture': 'Lightweight Transformer + Advanced Fusion (DS0 only)'
        }


if __name__ == "__main__":
    # Test PanTiny configurations

    print("PanTiny Model Parameter Analysis:")
    print("=" * 60)

    # Test configurations for ablation studies
    test_configs = [
        # Fusion ablation
        {'fusion_type': 'conv', 'refine_type': 'enhanced_conv', 'model_scale': 'normal'},
        {'fusion_type': 'gated_conv', 'refine_type': 'enhanced_conv', 'model_scale': 'normal'},
        {'fusion_type': 'channel_attention', 'refine_type': 'enhanced_conv', 'model_scale': 'normal'},
        {'fusion_type': 'deepfusion_v2', 'refine_type': 'enhanced_conv', 'model_scale': 'normal'},

        # Refinement ablation
        {'fusion_type': 'gated_conv', 'refine_type': 'channel_attention', 'model_scale': 'normal'},
        {'fusion_type': 'gated_conv', 'refine_type': 'residual_blocks', 'model_scale': 'normal'},

        # Scale ablation
        {'fusion_type': 'gated_conv', 'refine_type': 'enhanced_conv', 'model_scale': 'large'},
    ]

    config_names = [
        'conv_baseline',
        'gated_conv',
        'channel_attention',
        'deepfusion_v2',
        'refine_channel_att',
        'refine_residual',
        'large_scale'
    ]

    for i, (name, config) in enumerate(zip(config_names, test_configs)):
        try:
            args = {'model': config}
            model = Net(args)
            info = model.get_model_info()

            print(f"  {name}: {info['total_params']/1000:.1f}K params")

        except Exception as e:
            print(f"  {name}: Failed - {e}")

    # Test forward pass
    print("\nForward pass test:")
    ms = torch.randn(1, 4, 64, 64)
    bms = torch.randn(1, 4, 128, 128)
    pan = torch.randn(1, 1, 128, 128)

    for i, (name, config) in enumerate(zip(config_names[:3], test_configs[:3])):
        try:
            args = {'model': config}
            model = Net(args)
            model.eval()

            with torch.no_grad():
                output = model(ms, bms, pan)

            print(f"✓ {name}: Input {bms.shape} -> Output {output.shape}")

        except Exception as e:
            print(f"✗ {name}: Failed - {e}")

    print("\nPanTiny Design Summary:")
    print("- Lightweight yet effective architecture")
    print("- Multiple fusion strategies: conv, gated_conv, channel_attention, deepfusion_v2")
    print("- Advanced refinement modules: enhanced_conv, channel_attention, residual_blocks")
    print("- Scalable: normal (~60K) and large (~200K) variants")
    print("- Perfect for ablation studies and paper comparisons")
    print("- Suitable for paper title: 'PanTiny: Lightweight Pan-sharpening...'")
