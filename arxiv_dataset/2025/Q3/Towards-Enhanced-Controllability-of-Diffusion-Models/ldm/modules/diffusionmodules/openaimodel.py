from abc import abstractmethod
from functools import partial
import math
from pickle import NONE
from time import time
from typing import Iterable
from unittest.mock import NonCallableMock

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class TimestepAndCondBlock(nn.Module):
    """
    Any module where forward() takes 
    1. timestep embeddings as a second argument
    2. condition embeddings as a third argument
    """

    @abstractmethod
    def forward(self, x, t_emb, c_emb, c_emb2=None, style_mask=None, content_mask=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepAndCondEmbedSequential(nn.Sequential, TimestepAndCondBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, t_emb, c_emb, c_emb2=None, style_mask=None, content_mask=None):
        for layer in self:
            if isinstance(layer, TimestepAndCondBlock):
                if c_emb2 is None:
                    x = layer(x, t_emb, c_emb)
                else:
                    x = layer(x, t_emb, c_emb, c_emb2, style_mask, content_mask)
            elif isinstance(layer, SpatialTransformer):
                if c_emb2 is None:
                    x = layer(x, c_emb)
                else:
                    x = layer(x, c_emb, c_emb2)
            else:
                x = layer(x)
        return x


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

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
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

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
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
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
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
        return self.skip_connection(x) + h


class DisentanglingResBlock(TimestepAndCondBlock): # (TimestepBlock)
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
        style_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        use_content_feature=False,
        is_cond2_style_enc=False,
        time_for_content=False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.style_channels = style_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_content_feature = use_content_feature
        self.is_cond2_style_enc = is_cond2_style_enc
        self.time_for_content = time_for_content

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


        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        
        if time_for_content:
            self.t_emb_layers_content = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    1 if not is_cond2_style_enc else self.out_channels,
                ),
            )

        
        self.style_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                style_channels,
                self.out_channels,
            ),
        )

        if self.is_cond2_style_enc:
            self.style2_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    style_channels,
                    self.out_channels,
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
        if self.use_content_feature:
            if not self.is_cond2_style_enc and not self.time_for_content:
                self.beta = nn.Parameter(th.ones(1,1,1,1))

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, t_emb, c_emb, c_emb2, style_mask, content_mask):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, t_emb, c_emb, c_emb2, style_mask, content_mask), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, t_emb, style, content, style_mask, content_mask):
        
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        t_emb_out = self.t_emb_layers(t_emb).type(h.dtype)
        if self.time_for_content:
            t_emb_content_out = self.t_emb_layers_content(t_emb).type(h.dtype)

        style_out = self.style_layers(style).type(h.dtype)

        if content is not None:
            if not self.is_cond2_style_enc: 
            # content_out = F.adaptive_avg_pool2d(content, h.size(2))
                content_out = F.interpolate(content, h.size(2), mode='bilinear')
            else: # if style encoder is used for the second condition
                content_out = self.style2_layers(content).type(h.dtype)
                

        while len(t_emb_out.shape) < len(h.shape):
            t_emb_out = t_emb_out[..., None]
        while len(style_out.shape) < len(h.shape):
            style_out = style_out[..., None]
        if self.is_cond2_style_enc: 
            while len(content_out.shape) < len(h.shape):
                content_out = content_out[..., None]
        if self.time_for_content:
            while len(t_emb_content_out.shape) < len(h.shape):
                t_emb_content_out = t_emb_content_out[..., None]

        style_out = style_out * style_mask # [1,0]
        content_out = content_out * content_mask

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(t_emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            # h = out_norm(h) * scale + shift
            
            if content is not None: #1 16 16
                # h = self.beta * content_out * h + h
                if not self.is_cond2_style_enc: 
                    if self.time_for_content:
                        # scale, shift = th.chunk(t_emb_content_out, 2, dim=1)
                        # h = h * (1 + (scale * content_out + shift))
                        h = h * (1 + (t_emb_content_out * content_out))
                    else:
                        h = h * (1 + self.beta * content_out) 
                    
                else: # if style encoder is used for the second condition
                    if self.time_for_content:
                        # scale, shift = th.chunk(t_emb_content_out, 2, dim=1)
                        # h = h * (1 + (scale * content_out + shift))
                        h = h * (1 + (t_emb_content_out * content_out))
                    else:
                        h = h * (1 + content_out)
            h = h * (1 + style_out)
            h = out_rest(h)

        # else:
        #     out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        #     # scale, shift = th.chunk(style_out, 2, dim=1)
        #     h = out_norm(h) + t_emb_out
        #     if content is not None: 
        #         if not self.is_cond2_style_enc: 
        #             if self.time_for_content:
        #                 h = t_emb_content_out*(content_out * h) + h
        #             else:
        #                 h = self.beta * content_out * h + h
        #         else: # if style encoder is used for the second condition
        #             if self.time_for_content:
        #                 h = t_emb_content_out*(content_out[:] * h) + h
        #             else:   
        #                 h = content_out * h + h
        #     h = style_out * h # + shift
        #     h = out_rest(h)
        return self.skip_connection(x) + h
    
       
        

class EncResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    total layers:
        in_layers
        - norm
        - act
        - conv
        out_layers
        - norm
        - (modulation)
        - act
        - conv
    """    
    def __init__(
        self,
        channels,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.out_channels = out_channels or channels
        #############################
        # IN LAYERS
        #############################
        layers = [
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1)
        ]
        self.in_layers = nn.Sequential(*layers)

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        #############################
        # SKIP LAYERS
        #############################
        if self.out_channels == channels:
            # cannot be used with gatedconv, also gatedconv is alsways used as the first block
            self.skip_connection = nn.Identity()
        else:
            if use_conv:
                kernel_size = 3
                padding = 1
            else:
                kernel_size = 1
                padding = 0

            self.skip_connection = conv_nd(dims,
                                           channels,
                                           self.out_channels,
                                           kernel_size,
                                           padding=padding)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        Args:
            x: input
            lateral: lateral connection from the encoder
        """
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        return self.skip_connection(x) + h



class DecResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    total layers:
        in_layers
        - norm
        - act
        - conv
        out_layers
        - norm
        - (modulation)
        - act
        - conv
    """    
    def __init__(
        self,
        channels,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.out_channels = out_channels or channels
        #############################
        # IN LAYERS
        #############################
        layers = [
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1)
        ]
        self.in_layers = nn.Sequential(*layers)

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        #############################
        # SKIP LAYERS
        #############################
        if self.out_channels == channels:
            # cannot be used with gatedconv, also gatedconv is alsways used as the first block
            self.skip_connection = nn.Identity()
        else:
            if use_conv:
                kernel_size = 3
                padding = 1
            else:
                kernel_size = 1
                padding = 0

            self.skip_connection = conv_nd(dims,
                                           channels,
                                           self.out_channels,
                                           kernel_size,
                                           padding=padding)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        Args:
            x: input
            lateral: lateral connection from the encoder
        """
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        return self.skip_connection(x) + h



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
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
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
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append( 
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class DisentanglingUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        style_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        use_content_feature=False,
        is_cond2_style_enc=False,
        time_for_content=False,
        disentangling_scheduler=None, # one of [sigmoid, linear, exclusive]
        disentangling_timestep=None,
        sigmoid_coefficient=None, # used if disentangling_scheduler is set to be sigmoid
        content_scheduling=True,
        style_scheduling=True,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.style_channels = style_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.use_content_feature = use_content_feature
        # self.disentangling_timestep = disentangling_timestep
        self.is_cond2_style_enc = is_cond2_style_enc
        self.time_for_content = time_for_content
        
        # sigmoid
        if disentangling_scheduler == "sigmoid":
            if style_scheduling:
                self.scheduling_style_mask = lambda t: 1 / (1 + th.exp(-sigmoid_coefficient * (-(t + 1) + disentangling_timestep)))
            else:
                self.scheduling_style_mask = lambda t: (t == t).float()

            if content_scheduling:
                self.scheduling_content_mask = lambda t: 1 / (1 + th.exp(-sigmoid_coefficient * ((t + 1) - disentangling_timestep)))
            else:
                self.scheduling_content_mask = lambda t: (t == t).float()
                
        elif disentangling_scheduler == "linear":
            if style_scheduling:
                self.scheduling_style_mask = lambda t: (1001-t) / 1000
            else:
                self.scheduling_style_mask = lambda t: (t == t).float()

            if content_scheduling:
                self.scheduling_content_mask = lambda t: (t) / 1000
            else:
                self.scheduling_content_mask = lambda t: (t == t).float()
                
        elif disentangling_scheduler == "exclusive":
            # self.scheduling_style_mask = lambda t: (t < self.disentangling_timestep).float()
            if style_scheduling:
                self.scheduling_style_mask = lambda t: (t < disentangling_timestep).float()
                # self.scheduling_style_mask = lambda t: ((t < self.disentangling_timestep).float() * (t >= self.disentangling_timestep - exclusive_interval).float())
            else:
                self.scheduling_style_mask = lambda t: (t == t).float()
            
            if content_scheduling:
                self.scheduling_content_mask = lambda t: (t >= disentangling_timestep).float()
            else:
                self.scheduling_content_mask = lambda t: (t == t).float()
        
        elif disentangling_scheduler is None:
            self.scheduling_style_mask = lambda t: (t == t).float()
            self.scheduling_content_mask = lambda t: (t == t).float()
            print("disentangling_scheduler is set to be None")
        else:
            raise ValueError("disentangling_scheduler needs to be set one of ['sigmoid', 'linear', 'exclusive'].")
        # TODO: add the options to the config file.

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepAndCondEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    DisentanglingResBlock(
                        ch,
                        time_embed_dim,
                        self.style_channels,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_content_feature=self.use_content_feature,
                        is_cond2_style_enc=self.is_cond2_style_enc,
                        time_for_content=time_for_content,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append( 
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepAndCondEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepAndCondEmbedSequential(
                        DisentanglingResBlock(
                            ch,
                            time_embed_dim,
                            self.style_channels,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            use_content_feature=self.use_content_feature,
                            down=True,
                            is_cond2_style_enc=self.is_cond2_style_enc,
                            time_for_content=time_for_content,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepAndCondEmbedSequential(
            DisentanglingResBlock(
                ch,
                time_embed_dim,
                self.style_channels,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_content_feature=self.use_content_feature,
                is_cond2_style_enc=self.is_cond2_style_enc,
                time_for_content=time_for_content,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            DisentanglingResBlock(
                ch,
                time_embed_dim,
                self.style_channels,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_content_feature=self.use_content_feature,
                is_cond2_style_enc=self.is_cond2_style_enc,
                time_for_content=time_for_content,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    DisentanglingResBlock(
                        ch + ich,
                        time_embed_dim,
                        self.style_channels,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_content_feature=self.use_content_feature,
                        is_cond2_style_enc=self.is_cond2_style_enc,
                        time_for_content=time_for_content,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        DisentanglingResBlock(
                            ch,
                            time_embed_dim,
                            self.style_channels,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            use_content_feature=self.use_content_feature,
                            up=True,
                            is_cond2_style_enc=self.is_cond2_style_enc,
                            time_for_content=time_for_content,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepAndCondEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, style=None, content=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        if kwargs['validation']:
            scheduling_style_mask = lambda t: 1 / (1 + th.exp(-0.025 * (-(t + 1) + 550)))
            scheduling_content_mask = lambda t: 1 / (1 + th.exp(-0.025 * ((t + 1) - 550)))
            style_mask = scheduling_style_mask(timesteps).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            content_mask = scheduling_content_mask(timesteps).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        else:
            style_mask = self.scheduling_style_mask(timesteps).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            content_mask = self.scheduling_content_mask(timesteps).unsqueeze(1).unsqueeze(2).unsqueeze(3)

        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, style, content, style_mask, content_mask)
            hs.append(h)
        h = self.middle_block(h, emb, style, content, style_mask, content_mask)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, style, content, style_mask, content_mask)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        *args,
        **kwargs
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)


class StyleEncoderModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """
    def __init__(
        self,
        image_size, # 256
        in_channels, # 3
        model_channels, # 128
        out_channels, # 512
        num_res_blocks, # 2
        attention_resolutions, # (16,)
        dropout=0, # 0.1
        channel_mult=(1, 2, 4, 8), # (1,1,2,2,4,4,4)
        conv_resample=True, # True
        dims=2, # 2
        use_checkpoint=False, # False
        num_heads=1, # 1
        num_head_channels=-1, # -1
        # num_heads_upsample=-1, # 
        # use_scale_shift_norm=False,
        resblock_updown=False, # True
        use_new_attention_order=False, # False
        pool="adaptive", # 'adaptivenonzero'
        *args,
        **kwargs
    ):

        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order
        self.pool = pool

        # self.num_heads_upsample = num_heads_upsample
        ch = int(self.channel_mult[0] * self.model_channels)
        
        self.input_blocks = nn.ModuleList([
            nn.Sequential(
                conv_nd(self.dims, self.in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        
        resolution = self.image_size
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    EncResBlock(
                            ch,
                            out_channels=int(mult * self.model_channels),
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                    )
                ]
                ch = int(mult * self.model_channels)
                
                if resolution in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        ))
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                resolution //= 2 # ??
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        EncResBlock(
                            ch,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            down=True
                        ) if (
                            self.resblock_updown
                        ) else Downsample(ch,
                                          self.conv_resample,
                                          dims=self.dims,
                                          out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)

                self._feature_size += ch

        self.middle_block = nn.Sequential(
            EncResBlock(
                ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            EncResBlock(
                ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
            ),
        )
        self._feature_size += ch
        if self.pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(self.dims, ch, self.out_channels, 1),
                nn.Flatten(),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                # nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(self.dims, ch, self.out_channels, 1),
                # nn.Flatten(),
            )
            pass
            # raise NotImplementedError(f"Unexpected {self.pool} pooling")

    def forward(self, x, return_2d_feature=False):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        # :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
        else:
            h = h.type(x.dtype)

        h_2d = h
        h = self.out(h)

        if return_2d_feature:
            return h, h_2d
        else:
            return h

    def forward_flatten(self, x):
        """
        transform the last 2d feature into a flatten vector
        """
        h = self.out(x)
        return h


class ContentEncoderModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """
    def __init__(
        self,
        image_size, # 256
        in_channels, # 3
        model_channels, # 128
        out_channels, # 512
        num_res_blocks, # 2
        attention_resolutions, # (16,)
        dropout=0, # 0.1
        enc_channel_mult=(1, 2, 4, 8), # (1,1,2,2,4,4,4)
        dec_channel_mult=(1, 2, 4), # (1,1,2,2,4,4,4)
        conv_resample=True, # True
        dims=2, # 2
        use_checkpoint=False, # False
        num_heads=1, # 1
        num_head_channels=-1, # -1
        # num_heads_upsample=-1, # 
        # use_scale_shift_norm=False,
        resblock_updown=False, # True
        use_new_attention_order=False, # False
        pool="adaptive", # 'adaptivenonzero'
        legacy=True,
        use_spatial_transformer=False,
        num_heads_upsample=-1,
        transformer_depth=1,
        context_dim=None,
        lowpass_filter=True,
        *args,
        **kwargs
    ):

        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.enc_channel_mult = enc_channel_mult
        self.dec_channel_mult = dec_channel_mult
        self.conv_resample = conv_resample
        self.dims = dims
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order
        self.pool = pool

        # self.num_heads_upsample = num_heads_upsample
        ch = int(self.enc_channel_mult[0] * self.model_channels)
        
        self.input_blocks = nn.ModuleList([
            nn.Sequential(
                conv_nd(self.dims, self.in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        
        resolution = self.image_size
        for level, mult in enumerate(self.enc_channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    EncResBlock(
                            ch,
                            out_channels=int(mult * self.model_channels),
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                    )
                ]
                ch = int(mult * self.model_channels)
                
                if resolution in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=self.use_checkpoint,
                            num_heads=self.num_heads,
                            num_head_channels=self.num_head_channels,
                            use_new_attention_order=self.use_new_attention_order,
                        ))
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.enc_channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        EncResBlock(
                            ch,
                            out_channels=out_ch,
                            dims=self.dims,
                            use_checkpoint=self.use_checkpoint,
                            down=True
                        ) if (
                            self.resblock_updown
                        ) else Downsample(ch,
                                          self.conv_resample,
                                          dims=self.dims,
                                          out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)

                self._feature_size += ch
        
        # resolution //= 2
        # self.input_blocks.append(nn.AdaptiveAvgPool2d(1))

        # self.input_blocks.append(
        #     DecResBlock(
        #                     ch,
        #                     out_channels=out_ch,
        #                     dims=dims,
        #                     use_checkpoint=use_checkpoint,
        #                     up=True,
        #                 )
        # )
        # resolution *= 2
        # self.input_blocks.append(
        #     DecResBlock(
        #                     ch,
        #                     out_channels=out_ch,
        #                     dims=dims,
        #                     use_checkpoint=use_checkpoint,
        #                     up=True,
        #                 )
        # )
        # resolution *= 2
        
        # self.input_blocks.append(EqualConvTranspose2d(in_channels, out_channels, 4, 1, 0))
        # self.input_blocks.append(EqualConvTranspose2d(in_channels, out_channels, 4, 1, 0))
        # self.input_blocks.append(EqualConvTranspose2d(in_channels, out_channels, 4, 1, 0))
        # layers_content.append(PixelNorm())
        # layers_content.append(nn.LeakyReLU(0.1))


        # self.middle_block = nn.Sequential(nn.Identity())
        self.middle_block = nn.Sequential(
            EncResBlock(
                ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            EncResBlock(
                ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
            ),
        )
        self._feature_size += ch
       
        self.output_blocks = nn.ModuleList([])
    
        # self.output_blocks.append(nn.Sequential(nn.Identity()))
        for level, mult in list(enumerate(self.dec_channel_mult))[::-1]:
            for i in range(num_res_blocks):
                layers = [
                    DecResBlock(
                        ch,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = model_channels * mult
                if resolution in attention_resolutions:
                    print(resolution)
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                print("lowpass_filter", lowpass_filter)
                if lowpass_filter:
                    if level and i == num_res_blocks-1 and resolution == 4:
                        print('upsample')
                    #  if level and i == num_res_blocks-1: => for 16 x 16 content mask
                        out_ch = ch
                        layers.append(
                            DecResBlock(
                                ch,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                up=True,
                            )
                            if resblock_updown
                            else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                        resolution *= 2
                    
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch


        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, model_channels, 1, 1),
            nn.Sigmoid()
        )
        

                
        # if self.pool == "adaptive":
        #     self.out = nn.Sequential(
        #         normalization(ch),
        #         nn.SiLU(),
        #         nn.AdaptiveAvgPool2d((1, 1)),
        #         conv_nd(self.dims, ch, self.out_channels, 1),
        #         nn.Flatten(),
        #     )
        # else:
        #     self.out = nn.Sequential(
        #         normalization(ch),
        #         nn.SiLU(),
        #         # nn.AdaptiveAvgPool2d((1, 1)),
        #         conv_nd(self.dims, ch, self.out_channels, 1),
        #         nn.Sigmoid()
        #         # nn.Flatten(),
        #     )
        #     pass
            # raise NotImplementedError(f"Unexpected {self.pool} pooling")

    def forward(self, x, return_flattened_feature=False):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        # :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
        else:
            h = h.type(x.dtype)
        
        h_2d = h

        for module in self.output_blocks:
            h = module(h)
        
        h = h.type(x.dtype)
        h = self.out(h)
        
        assert h.size() == (h.size(0), 1, 8, 8), f"Content feature size is wrong --current size: {h.size()}"

        if return_flattened_feature:
            return h, h_2d
        else:
            return h

    def forward_flatten(self, x):
        """
        transform the last 2d feature into a flatten vector
        """
        h = self.out(x)
        return h