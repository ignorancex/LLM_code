import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

# Try to import mamba_ssm, provide fallback if not available
try:
    from .mamba_simple import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("Warning: mamba_ssm not available, using fallback implementation")
    MAMBA_AVAILABLE = False
    # Fallback Mamba implementation
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
            super().__init__()
            self.d_model = d_model
            self.linear = nn.Linear(d_model, d_model)
            
        def forward(self, hidden_states, inference_params=None):
            return self.linear(hidden_states)

# Try to import refine module, create dummy if not available
try:
    from .refine import Refine
except ImportError:
    print("Warning: refine module not available, using dummy implementation")
    class Refine(nn.Module):
        def __init__(self, n_feat, out_channels=4):
            super().__init__()
            self.conv = nn.Conv2d(n_feat, out_channels, 3, 1, 1)
            
        def forward(self, x):
            return self.conv(x)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class CrossAttention(nn.Module):
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

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm_cro1= LayerNorm(dim, LayerNorm_type)
        self.norm_cro2 = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.cro = CrossAttention(dim,num_heads,bias)
        self.proj = nn.Conv2d(dim,dim,1,1,0)
    def forward(self, ms,pan):
        ms = ms+self.cro(self.norm_cro1(ms),self.norm_cro2(pan))
        ms = ms + self.ffn(self.norm2(ms))
        return ms


class BiasFree_LayerNorm(nn.Module):
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
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
class PatchUnEmbed(nn.Module):
    def __init__(self,basefilter) -> None:
        super().__init__()
        self.nc = basefilter
    def forward(self, x,x_size):
        B,HW,C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self,patch_size=4, stride=4,in_chans=36, embed_dim=32*32*32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm(embed_dim,'BiasFree')

    def forward(self, x):
        #（b,c,h,w)->(b,c*s*p,h//s,w//s)
        #(b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x
class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        # Use supported bimamba_type or None for standard mamba
        self.encoder = Mamba(dim, bimamba_type="none")
        self.norm = LayerNorm(dim, 'WithBias')  # Fix LayerNorm_type
        
    def forward(self, ipt):
        x, residual = ipt
        residual = x + residual
        x = self.norm(residual)
        return (self.encoder(x), residual)
class TokenSwapMamba(nn.Module):
    def __init__(self, dim):
        super(TokenSwapMamba, self).__init__()
        self.msencoder = Mamba(dim, bimamba_type="none")
        self.panencoder = Mamba(dim, bimamba_type="none")
        self.norm1 = LayerNorm(dim, 'WithBias')  # Fix LayerNorm_type
        self.norm2 = LayerNorm(dim, 'WithBias')  # Fix LayerNorm_type
        
    def forward(self, ms, pan, ms_residual, pan_residual):
        # ms (B,N,C)
        # pan (B,N,C)
        ms_residual = ms + ms_residual
        pan_residual = pan + pan_residual
        ms = self.norm1(ms_residual)
        pan = self.norm2(pan_residual)
        B, N, C = ms.shape
        ms_first_half = ms[:, :, :C//2]
        pan_first_half = pan[:, :, :C//2]
        ms_swap = torch.cat([pan_first_half, ms[:, :, C//2:]], dim=2)
        pan_swap = torch.cat([ms_first_half, pan[:, :, C//2:]], dim=2)
        ms_swap = self.msencoder(ms_swap)
        pan_swap = self.panencoder(pan_swap)
        return ms_swap, pan_swap, ms_residual, pan_residual
class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        # Use supported bimamba_type - try "v2" instead of "v3"
        try:
            self.cross_mamba = Mamba(dim, bimamba_type="v2")
        except:
            # Fallback to standard mamba if bimamba not supported
            self.cross_mamba = Mamba(dim, bimamba_type="none")
            
        self.norm1 = LayerNorm(dim, 'WithBias')  # Fix LayerNorm_type
        self.norm2 = LayerNorm(dim, 'WithBias')  # Fix LayerNorm_type
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        # Since Mamba doesn't support extra_emb, we need to fuse features differently
        self.fusion_proj = nn.Linear(dim * 2, dim)  # Project concatenated features
        
    def forward(self, ms, ms_resi, pan):
        ms_resi = ms + ms_resi
        ms = self.norm1(ms_resi)
        pan = self.norm2(pan)
        
        # Concatenate ms and pan features for cross-attention effect
        B, N, C = ms.shape
        fused_input = torch.cat([ms, pan], dim=-1)  # (B, N, 2*C)
        fused_input = self.fusion_proj(fused_input)  # (B, N, C)
        
        global_f = self.cross_mamba(fused_input)
        B, HW, C = global_f.shape
        ms = global_f.transpose(1, 2).view(B, C, 128, 128)
        ms = (self.dwconv(ms) + ms).flatten(2).transpose(1, 2)
        return ms, ms_resi
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x+resi
class Net(nn.Module):
    def __init__(self, num_channels=None, base_filter=None, args=None):
        super(Net, self).__init__()
        
        # Handle different argument formats for compatibility
        if args is not None:
            if hasattr(args, 'base_filter'):
                base_filter = args.base_filter
            elif hasattr(args, 'get'):
                base_filter = args.get('base_filter', 32)
        
        if base_filter is None:
            base_filter = 32
            
        if num_channels is None:
            num_channels = 4
            
        self.base_filter = base_filter
        self.num_channels = num_channels
        self.stride = 1
        self.patch_size = 1
        
        # Encoders
        self.pan_encoder = nn.Sequential(
            nn.Conv2d(1, base_filter, 3, 1, 1),
            HinResBlock(base_filter, base_filter),
            HinResBlock(base_filter, base_filter),
            HinResBlock(base_filter, base_filter)
        )
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(num_channels, base_filter, 3, 1, 1),
            HinResBlock(base_filter, base_filter),
            HinResBlock(base_filter, base_filter),
            HinResBlock(base_filter, base_filter)
        )
        
        self.embed_dim = base_filter * self.stride * self.patch_size
        self.shallow_fusion1 = nn.Conv2d(base_filter * 2, base_filter, 3, 1, 1)
        self.shallow_fusion2 = nn.Conv2d(base_filter * 2, base_filter, 3, 1, 1)
        
        # Patch embedding
        self.ms_to_token = PatchEmbed(
            in_chans=base_filter, embed_dim=self.embed_dim, 
            patch_size=self.patch_size, stride=self.stride
        )
        self.pan_to_token = PatchEmbed(
            in_chans=base_filter, embed_dim=self.embed_dim,
            patch_size=self.patch_size, stride=self.stride
        )
        
        # Deep fusion layers
        self.deep_fusion1 = CrossMamba(self.embed_dim)
        self.deep_fusion2 = CrossMamba(self.embed_dim)
        self.deep_fusion3 = CrossMamba(self.embed_dim)
        self.deep_fusion4 = CrossMamba(self.embed_dim)
        self.deep_fusion5 = CrossMamba(self.embed_dim)

        # Feature extraction
        self.pan_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
        self.ms_feature_extraction = nn.Sequential(*[SingleMambaBlock(self.embed_dim) for i in range(8)])
        self.swap_mamba1 = TokenSwapMamba(self.embed_dim)
        self.swap_mamba2 = TokenSwapMamba(self.embed_dim)
        
        # Output layers
        self.patchunembe = PatchUnEmbed(base_filter)
        self.output = Refine(base_filter, num_channels)
    
    def forward(self, ms, _, pan):
        ms_bic = F.interpolate(ms, scale_factor=4)
        ms_f = self.ms_encoder(ms_bic)
        b, c, h, w = ms_f.shape
        pan_f = self.pan_encoder(pan)
        
        # Convert to tokens
        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        residual_ms_f = 0
        residual_pan_f = 0
        
        # Feature extraction
        ms_f, residual_ms_f = self.ms_feature_extraction([ms_f, residual_ms_f])
        pan_f, residual_pan_f = self.pan_feature_extraction([pan_f, residual_pan_f])
        
        # Token swapping
        ms_f, pan_f, residual_ms_f, residual_pan_f = self.swap_mamba1(ms_f, pan_f, residual_ms_f, residual_pan_f)
        ms_f, pan_f, residual_ms_f, residual_pan_f = self.swap_mamba2(ms_f, pan_f, residual_ms_f, residual_pan_f)
        
        # Shallow fusion
        ms_f = self.patchunembe(ms_f, (h, w))
        pan_f = self.patchunembe(pan_f, (h, w))
        ms_f = self.shallow_fusion1(torch.cat([ms_f, pan_f], dim=1)) + ms_f
        pan_f = self.shallow_fusion2(torch.cat([pan_f, ms_f], dim=1)) + pan_f
        
        # Deep fusion
        ms_f = self.ms_to_token(ms_f)
        pan_f = self.pan_to_token(pan_f)
        residual_ms_f = 0
        ms_f, residual_ms_f = self.deep_fusion1(ms_f, residual_ms_f, pan_f)
        ms_f, residual_ms_f = self.deep_fusion2(ms_f, residual_ms_f, pan_f)
        ms_f, residual_ms_f = self.deep_fusion3(ms_f, residual_ms_f, pan_f)
        ms_f, residual_ms_f = self.deep_fusion4(ms_f, residual_ms_f, pan_f)
        ms_f, residual_ms_f = self.deep_fusion5(ms_f, residual_ms_f, pan_f)
        
        # Output
        ms_f = self.patchunembe(ms_f, (h, w))
        hrms = self.output(ms_f) + ms_bic
        return hrms

    def get_model_info(self):
        """Get model configuration information"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'name': f'PanMamba_{self.base_filter}',
            'base_filter': self.base_filter,
            'num_channels': self.num_channels,
            'total_params': total_params,
            'mamba_available': MAMBA_AVAILABLE
        }

if __name__ == "__main__":
    # Test the model with different configurations
    print("Testing PanMamba model...")
    
    # Test basic configuration
    args_basic = {'base_filter': 32, 'num_channels': 4}
    model_basic = Net(args=args_basic)
    print(f"Basic model info: {model_basic.get_model_info()}")
    
    # Test with dummy data
    ms = torch.randn(1, 4, 64, 64)
    pan = torch.randn(1, 1, 256, 256)
    
    try:
        with torch.no_grad():
            out = model_basic(ms, None, pan)
            print(f"Forward pass successful! Output shape: {out.shape}")
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        
    print(f"MAMBA_AVAILABLE: {MAMBA_AVAILABLE}")


