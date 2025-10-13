from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from .modules_conv import checkpoint
from .modules_attention import FeedForward, CrossAttention, Normalize, zero_module


def _to_seq(x):
    if x.ndim == 5:
        out = rearrange(x, "b c t h w -> b (t h w) c")
    elif x.ndim == 4:
        out = rearrange(x, "b c h w -> b (h w) c")
    else:
        raise
    return out


def _from_seq(x, data_shape):
    out = rearrange(x, "b s c -> b c s")
    out = out.reshape(out.shape[:2] + data_shape)
    return out


class AdditiveCondition(nn.Module):
    def __init__(self, query_dim, context_dim, axis_to_align_x, axis_to_align_c, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.axis_x = axis_to_align_x
        self.axis_c = axis_to_align_c

        # self.proj_c = nn.Linear(context_dim, query_dim, bias=False)
        self.norm = Normalize(query_dim)
        self.proj_c = FeedForward(dim=context_dim, dim_out=query_dim, mult=4, glu=True)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        # the shape of the input should be either (b,c,t,h,w) or (b,c,h,w)
        h = self.heads

        context_shape = context.shape[2:]
        context = _from_seq(self.proj_c(_to_seq(context)), context_shape)
        dim_reduce = tuple(set(range(context.ndim)) - set([0, 1, self.axis_c]))
        context = context.mean(dim=dim_reduce) # (b,c,t) or (b,c,h)
        context = self.norm(context)
        x_shape = x.shape
        if len(x_shape) == 5:
            x = rearrange(x, "b c t h w -> b c t (h w)")
        n = x.shape[list(set(range(len(x.shape))) - set([self.axis_x]))[-1]]
        context = context.unsqueeze(dim=-1).tile(n)
        context = F.interpolate(context, size=x.shape[-2:])
        x = (x + context).reshape(x_shape)
        if len(x_shape) == 5:
            x = rearrange(x, "b c t h w -> (b t) c h w")

        data_shape = x.shape[2:]
        x = _to_seq(x)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        out = self.to_out(out)
        out = _from_seq(out, data_shape)
        if len(x_shape) == 5:
            out = rearrange(out, "(b t) c h w -> b c t h w", t=x_shape[2])

        return out


class AdditiveConditionBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, axis_to_align_x, axis_to_align_c, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn_c = AdditiveCondition(query_dim=dim, context_dim=context_dim, axis_to_align_x=axis_to_align_x, axis_to_align_c=axis_to_align_c, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn_s = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context):
        # the shape of the input should be either (b,c,t,h,w) or (b,c,h,w)
        ## Type A: additive condition -> self-attention
        data_shape = x.shape[2:]
        # additive condition
        x = _from_seq(self.norm1(_to_seq(x)), data_shape)
        x = self.attn_c(x, context) + x
        # self-attention
        if len(data_shape) == 3:
            x = rearrange(x, "b c t h w -> (b t) c h w")
        x = _to_seq(x)
        x = self.norm2(x)
        x = self.attn_s(x) + x
        # feed-forward
        x = self.norm3(x)
        x = self.ff(x) + x
        x = _from_seq(x, data_shape[-2:])
        if len(data_shape) == 3:
            x = rearrange(x, "(b t) c h w -> b c t h w", t=data_shape[0])

        return x
    

class AdditiveConditionTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, axis_to_align_x, axis_to_align_c, 
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [AdditiveConditionBlock(inner_dim, n_heads, d_head, axis_to_align_x, axis_to_align_c, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context):
        x_in = x
        data_shape = x.shape[2:]
        x = _to_seq(x).permute(0, 2, 1).unsqueeze(-1)
        x = self.norm(x)
        x = self.proj_in(x)
        x = _from_seq(x.permute(0, 2, 1, 3).squeeze(-1), data_shape)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = _to_seq(x).permute(0, 2, 1).unsqueeze(-1)
        x = self.proj_out(x)
        x = _from_seq(x.permute(0, 2, 1, 3).squeeze(-1), data_shape)
        return x + x_in
    