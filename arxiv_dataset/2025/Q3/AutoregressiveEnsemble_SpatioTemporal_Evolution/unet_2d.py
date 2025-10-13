## Code adapted from DiTTO paper by Ovadia et al.
## Ovadia, Oded, et al. "Real-time inference and extrapolation via a diffusion-inspired 
## temporal transformer operator (DiTTO)." arXiv preprint arXiv:2307.09072 (2023).


import math
from functools import partial

import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange


####  helper functions   #####

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

## normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

## Residual connection

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

## upsample layer
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

## Downsample layer
def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn, attn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)
        self.attn = attn

    def forward(self, x):
        x = self.norm(x)
        ##return x itself if not using attention block
        if self.attn:
            return self.fn(x)
        else:
            return x
        
        
####  Specific blocks ########
## sinusoidal positional embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        #half_dim = self.dim // 2
        half_dim = self.dim  
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
       # emb = x[:, None] * emb[None, :]
        emb = x * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


## Building block CNN

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        # self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


## ResNet
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

## Attention module for intermediate layers (temporal)
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        if heads is not None:
            self.scale = dim_head ** -0.5
            hidden_dim = dim_head * heads
            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

            self.to_out = nn.Sequential(
                nn.Conv2d(hidden_dim, dim, 1),
                LayerNorm(dim)
            )

    def forward(self, x):
        if self.heads is None:
            return x

        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), 
                      qkv)

        # q = q.softmax(dim = -2)
        # k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)
        
        ##temporal. attention matrix (nch x nch)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v) 

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)


## Attention module for bottleneck layer        (Spatial)
class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        if heads is not None:
            self.scale = dim_head ** -0.5
            hidden_dim = dim_head * heads

            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
            self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        if self.heads is None:
            return x
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), 
                      qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        ##spatial. attention matrix (ngrid x ngrid)
        attn = sim#.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

##Gate network to choose between expert networks (simple MLP takes in strain value history)
class moe_gate_network(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
    ##MoE gate network MLP embedding
        self.moe_var_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.GELU(),
                nn.Linear(32, n_experts),
                nn.Softmax(dim=-1)
        )
        
        #self.conv_block = Block(2, 1, groups = 1)
        #self.conv_block = nn.Conv2d(720, 360, 1)  ##for lf model
        self.conv_block = nn.Conv2d(2, 1, 1)       ##for lb model
    def forward(self,x):
        return self.conv_block(x)
            

#########  Full UNet model ##############3
class Unet2D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = None,
        ip_time_dim = None,
        attn = None,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attention_heads = 4,
        dim_head = 32
    ):
        super().__init__()

        ## determine dimensions
        self.ip_time_dim = ip_time_dim
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        ## time embedding
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
           # fourier_dim = learned_sinusoidal_dim + 1
        else:
            ##Temporal input assumed to be concatenatation of {time variable, time ID}
            dataset = 'micro'
            if dataset == 'micro':
                ip_time_var = int(self.ip_time_dim/2)            ## micro case
                ip_timeid_dim = int(self.ip_time_dim/2)        ##micro case
            elif dataset == 'grayscott':
                ip_time_var = 2                                    ##gray scott
                ip_timeid_dim = self.ip_time_dim - ip_time_var   ## if giving Time ID + time var as input - gray scott
            else:
                ip_timeid_dim = self.ip_time_dim                  ##if only giving time ID as input - surface temp
            sinu_pos_emb = SinusoidalPosEmb(ip_timeid_dim)
            
           # sinu_pos_emb = SinusoidalPosEmb(self.ip_time_dim)
           # fourier_dim = dim
        
        ## Time ID sinusoidal embedding 
        self.time_emb_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(int(2*ip_timeid_dim), int(time_dim/2)),
            nn.GELU(),
            nn.Linear(int(time_dim/2), int(time_dim/2))    ##if giving time ID + time var as input
           # nn.Linear(int(time_dim/2), int(time_dim))     ##if only giving time ID as input
        )
       

        ## TIme variable MLP embedding
        if dataset == 'micro' or  dataset == 'grayscott':
            self.time_var_mlp = nn.Sequential(
                nn.Linear(ip_time_var, int(time_dim/2)),
                nn.GELU(),
                nn.Linear(int(time_dim/2), int(time_dim/2))
            )
        
        ## Define intermediate layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in, heads=attention_heads, dim_head=dim_head), attn)),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, heads=attention_heads), attn))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out, heads=attention_heads, dim_head=dim_head), attn)),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        
        ## Define output layer
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
    
    
    def get_grid(self, shape, device='cuda'):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    ## Forward pass for UNet
    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()
        
        ##MLP embedding
        mlp = True           
        if mlp:
            dataset = 'micro'
            if dataset  == 'micro':
                time_var_ip = time[:,:int(self.ip_time_dim/2)]        ## micro
                time_emb_ip = time[:,int(self.ip_time_dim/2):]        ## micro
            elif dataset == 'grayscott':
                time_var_ip = time[:,:2]                              ##Gray scott
                time_emb_ip = time[:,2:]                              ##Gray Scott
            else:
                t = self.time_emb_mlp(time) if time is not None else None          ## surface temp
            if dataset  == 'grayscott' or dataset == 'micro':    
                t1 = self.time_var_mlp(time_var_ip) if time is not None else None   ##micro/Gray scott
                t2 = self.time_emb_mlp(time_emb_ip) if time is not None else None    ## micro/Gray scott
                t= torch.cat((t1,t2),dim=-1)                                        ## micro/Gray scott
        else:
            #LSTM embedding
            t, (_ ,_) =self.lstm(time.reshape(time.shape[0],int(time.shape[1]/2),2))
            t=t.reshape(t.shape[0],-1)

        h = []
        
        ## Forward pass
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


## Check on dummy data
if __name__ == "__main__":
    model = Unet2D(dim=16, out_dim = 1, dim_mults=(1, 2, 4, 8), channels = 3, ip_time_dim = 6)
    batchsize = 40
    pred = model(torch.rand((batchsize, 3, 72, 144)), time=torch.rand((batchsize,6)))
    print('OK')
    
    
    
    
