"""
Based on the official implementation of CoDi (MIT license)
https://github.com/microsoft/i-Code/tree/main/i-Code-V3
"""

from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import random
import numpy as np
import torch as th
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from .modules_conv import \
    checkpoint, conv_nd, linear, avg_pool_nd, \
    zero_module, normalization, timestep_embedding
from .modules_attention import SpatialTransformer
from .modules_video import SpatioTemporalAttention
from .modules_original import AdditiveConditionTransformer


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class VideoSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    pass
    
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, context=None, x_0=None):
        is_video = (x.ndim == 5)
        if is_video:
            num_frames = x.shape[2]

        for layer in self:
            if isinstance(layer, TimestepBlock):
                if is_video and emb is not None:
                    emb = emb.unsqueeze(1).repeat(1, num_frames, 1)
                    emb = rearrange(emb, 'b t c -> (b t) c')
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                if is_video:
                    x = rearrange(x, 'b c t h w -> (b t) c h w ')
                    if context is not None:
                        context_vid = context.unsqueeze(1).repeat(1, num_frames, 1, 1)
                        context_vid = rearrange(context_vid, 'b t n c -> (b t) n c')
                    x = layer(x, context_vid)
                    x = rearrange(x, '(b t) c h w -> b c t h w', t=num_frames)
                else:
                    x = layer(x, context)
            elif isinstance(layer, SpatioTemporalAttention):
                x = layer(x, x_0)
            elif isinstance(layer, VideoSequential) or isinstance(layer, nn.ModuleList):
                if is_video and emb is not None:
                    emb = emb.unsqueeze(1).repeat(1, num_frames, 1)
                    emb = rearrange(emb, 'b t c -> (b t) c')
                x = layer[0](x, emb)    
                x = layer[1](x, x_0)    
            elif isinstance(layer, AdditiveConditionTransformer):
                x = layer(x, context)
            else:
                if is_video:
                    x = rearrange(x, 'b c t h w -> (b t) c h w ')
                x = layer(x)
                if is_video:
                    x = rearrange(x, '(b t) c h w -> b c t h w', t=num_frames)
        return x


class UpsampleDeterministic(nn.Module):
    def __init__(self, upscale=2):
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale
        
    def forward(self, x):
        return x[:,:,:, None, :, None]\
        .expand(-1,-1,-1, self.upscale, -1 , self.upscale)\
        .reshape(x. size(0), x.size(1), x.size(2)*self.upscale, x.size(3)*self.upscale)
    
    
class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)
        self.upsample = UpsampleDeterministic(2)
        
    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.upsample(x)
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

    
class ConnectorOut(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout=0,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        use_temporal_attention=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )
        self.use_temporal_attention = use_temporal_attention
        if use_temporal_attention:
            self.temporal_attention = SpatioTemporalAttention(
                                dim = self.out_channels,
                                dim_head = self.out_channels // 4,
                                heads = 8,
                                use_resnet = False,
                              )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, ), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        is_video = x.ndim == 5
        if is_video:
            num_frames = x.shape[2]
            if self.use_temporal_attention:
                x = self.temporal_attention(x)
            x = rearrange(x, 'b c t h w -> (b t) c h w ')
                    
        h = self.in_layers(x)
        h = self.out_layers(h)
        out = self.skip_connection(x) + h
        if is_video:
            out = rearrange(out, '(b t) c h w -> b c t h w', t=num_frames)
            out = out.mean(2)
        return out.mean([2, 3]).unsqueeze(1)

    
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        is_video = x.ndim == 5
        if is_video:
            num_frames = x.shape[2]
            x = rearrange(x, 'b c t h w -> (b t) c h w ')
                    
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
            
        out = self.skip_connection(x) + h
        if is_video:
            out = rearrange(out, '(b t) c h w -> b c t h w', t=num_frames)
        return out


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight, dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight, dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class FCBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 1, padding=0),)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, self.out_channels,),)
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 1, padding=0)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1, padding=0)

    def forward(self, x, emb):
        if len(x.shape) == 2:
            x = x[:, :, None, None]
        elif len(x.shape) == 4:
            pass
        else:
            raise ValueError
        y = checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint)
        if len(x.shape) == 2:
            return y[:, :, 0, 0]
        elif len(x.shape) == 4:
            return y

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class Linear_MultiDim(nn.Linear):
    def __init__(self, in_features, out_features, *args, **kwargs):
        
        in_features = [in_features] if isinstance(in_features, int) else list(in_features)
        out_features = [out_features] if isinstance(out_features, int) else list(out_features)
        self.in_features_multidim = in_features
        self.out_features_multidim = out_features
        super().__init__(
            np.array(in_features).prod(), 
            np.array(out_features).prod(), 
            *args, **kwargs)

    def forward(self, x):
        shape = x.shape
        n = len(self.in_features_multidim)
        x = x.reshape(*shape[0:-n], self.in_features)
        y = super().forward(x)
        y = y.view(*shape[0:-n], *self.out_features_multidim)
        return y

    
class FCBlock_MultiDim(FCBlock):
    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_checkpoint=False,):
        channels = [channels] if isinstance(channels, int) else list(channels)
        channels_all = np.array(channels).prod()
        self.channels_multidim = channels

        if out_channels is not None:
            out_channels = [out_channels] if isinstance(out_channels, int) else list(out_channels)
            out_channels_all = np.array(out_channels).prod()
            self.out_channels_multidim = out_channels
        else:
            out_channels_all = channels_all
            self.out_channels_multidim = self.channels_multidim

        self.channels = channels
        super().__init__(
            channels = channels_all,
            emb_channels = emb_channels,
            dropout = dropout,
            out_channels = out_channels_all,
            use_checkpoint = use_checkpoint,)

    def forward(self, x, emb):
        shape = x.shape
        n = len(self.channels_multidim)
        x = x.reshape(*shape[0:-n], self.channels, 1, 1)
        x = x.view(-1, self.channels, 1, 1)
        y = checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint)
        y = y.view(*shape[0:-n], -1)
        y = y.view(*shape[0:-n], *self.out_channels_multidim)
        return y

    