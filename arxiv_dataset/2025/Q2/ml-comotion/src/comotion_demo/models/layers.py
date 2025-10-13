# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import einops as eo
import torch
import transformers
from timm.layers import LayerNorm2d
from timm.models.convnext import ConvNeXtBlock
from torch import nn
from torch.functional import F
from transformers.models.llama import modeling_llama

from ..utils import helper


class PosEmbed(nn.Module):
    def __init__(self, embed_dim=16):
        super().__init__()
        self.register_buffer(
            "freq_bands", 2 ** torch.linspace(0, embed_dim - 1, embed_dim)
        )
        self.out_dim = 2 * (1 + len(self.freq_bands))

    def forward(self, x):
        # Input: ... x 2
        emb = torch.sin(x.unsqueeze(-1) * self.freq_bands)
        emb = eo.rearrange(emb, "... d0 d1 -> ... (d0 d1)")

        return torch.cat([x, emb], -1)


class RotaryEmbed(nn.Module):
    def __init__(self, dim=32, rescale=1000, base=10000):
        super().__init__()
        self.dim = dim
        self.rescale = rescale
        self.base = base

        inv_freq = rescale / (base ** (torch.arange(0, dim).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.out_dim = dim

    def forward(self, x):
        """Apply frequency band and return cos and sin terms."""
        emb = x.unsqueeze(-1) * self.inv_freq
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x, cs, sn):
    half_channels = x.shape[1] // 2
    x0 = x[:, :half_channels]
    x1 = x[:, half_channels:]
    x_ = torch.cat((-x1, x0), dim=1)
    return cs * x + sn * x_


def _gru_update(h, z, q):
    return (1 - z) * h + z * q


class GRU(nn.Module):
    def __init__(self, hdim=128, f_in=128):
        super().__init__()
        self.fz = nn.Linear(hdim + f_in, hdim)
        self.fr = nn.Linear(hdim + f_in, hdim)
        self.fq = nn.Linear(hdim + f_in, hdim)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=-1)

        z = torch.sigmoid(self.fz(hx))
        r = torch.sigmoid(self.fr(hx))
        q = torch.tanh(self.fq(torch.cat([r * h, x], dim=-1)))

        h = _gru_update(h, z, q)
        return h


class ResidualMLP(nn.Module):
    """Residual Multilayer Perceptron."""

    def __init__(self, dim, out_dim=None, hidden_dim=None, pre_ln=True):
        super().__init__()

        if out_dim is None:
            out_dim = dim
        if hidden_dim is None:
            hidden_dim = 2 * dim
        self.is_residual = dim == out_dim
        self.pre_ln = pre_ln

        if pre_ln:
            self.ln0 = nn.LayerNorm(dim)
        self.l1 = nn.Linear(dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        inp = x
        if self.pre_ln:
            x = self.ln0(x)
        x = self.l1(x)
        x = self.act(self.ln1(x))
        x = self.l2(x)

        if self.is_residual:
            return inp + x
        else:
            return x


class DownsampleConvNextBlock(nn.Module):
    """Module to downsample 2x and apply ConvNext layers.

    Note we assume the input has already had normalization applied,
    and apply LayerNorm as the last operation.
    """

    def __init__(self, input_dim, output_dim=None, dropout=0.0):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 2, 2, 0),  # Downsample
            ConvNeXtBlock(output_dim),
            LayerNorm2d(output_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.layers(x)


@helper.fixed_dim_op(d=3)
def _call_llama(x, tf):
    kargs = {
        "position_embeddings": (
            torch.ones_like(x[..., :1]),
            torch.zeros_like(x[..., :1]),
        )
    }
    for tf_ in tf:
        x = tf_(x, **kargs)[0]
    return x


class DecodeFromTokens(nn.Module):
    """Decode token into hidden update."""

    def __init__(self, hidden_dim):
        super().__init__()

        self.ln = nn.LayerNorm(hidden_dim)
        self.token_weight = nn.Linear(hidden_dim, hidden_dim)
        self.token_to_value = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = ResidualMLP(hidden_dim)

    def forward(self, x):
        x = self.ln(x)
        v = self.token_to_value(x)
        w = torch.sigmoid(self.token_weight(x))
        v = v * w
        x = v.mean(-2)
        x = self.decoder(x)

        return x


class CrossAttention(nn.Module):
    """Cross attention module."""

    def __init__(
        self,
        num_tokens,
        num_heads,
        hidden_dim,
        token_dim,
        num_layers=2,
        drop_rate=0.1,
        use_global_attention=2,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.use_global_attention = use_global_attention

        self.token_to_query = nn.Sequential(
            ResidualMLP(self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.token_dim),
        )

        self.post_attention = nn.Sequential(
            ResidualMLP(self.token_dim),
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.hidden_dim),
        )

        cfg = transformers.LlamaConfig()
        cfg.hidden_size = self.hidden_dim
        cfg.intermediate_size = self.hidden_dim * 2
        cfg.num_attention_heads = 16
        cfg.num_key_value_heads = 16
        cfg.attention_dropout = self.drop_rate

        if self.use_global_attention > 0:
            self.global_attention = nn.ModuleList(
                [
                    modeling_llama.LlamaDecoderLayer(cfg, i)
                    for i in range(self.num_layers)
                ]
            )

        self.indiv_attention = nn.ModuleList(
            [modeling_llama.LlamaDecoderLayer(cfg, i) for i in range(self.num_layers)]
        )

    def forward(self, image_key, image_value, tokens):
        q = self.token_to_query(tokens)
        q = eo.rearrange(
            q, "b n d0 (h d1) -> b h (n d0) d1", h=self.num_heads
        ).contiguous()
        px_feedback = F.scaled_dot_product_attention(q, image_key, image_value)
        px_feedback = eo.rearrange(
            px_feedback, "b h (n d0) d1 -> b n d0 (h d1)", d0=self.num_tokens
        )
        tokens = tokens + self.post_attention(px_feedback)

        # Attention across all people
        if self.use_global_attention == 1:
            # Concatenate all tokens together
            tokens = eo.rearrange(tokens, "b n d0 d1 -> b (n d0) d1")
            tokens = _call_llama(tokens, self.global_attention)
            tokens = eo.rearrange(
                tokens, "b (n d0) d1 -> b n d0 d1", d0=self.num_tokens
            )
        elif self.use_global_attention == 2:
            # Do attention over people separately per-token
            tokens = eo.rearrange(tokens, "b n d0 d1 -> b d0 n d1")
            tokens = _call_llama(tokens, self.global_attention)
            tokens = eo.rearrange(tokens, "b d0 n d1 -> b n d0 d1")

        # Separate attention update per-person
        tokens = _call_llama(tokens, self.indiv_attention)

        return tokens


class FusePyramid(nn.Module):
    """Module for fusing the feature pyramid."""

    def __init__(self, in_dim=256, hidden_dim=512):
        """Initialize FusePyramid module."""
        super().__init__()

        self.dc0 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, 2)
        self.proj1 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.dc1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, 2)
        self.ln1 = LayerNorm2d(hidden_dim)
        self.proj2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.dc2 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, 2)
        self.ln2 = LayerNorm2d(hidden_dim)
        self.dc3 = nn.ConvTranspose2d(hidden_dim, in_dim, 4, 4)
        self.proj3 = nn.Conv2d(in_dim, in_dim, 1)
        self.ln3 = LayerNorm2d(in_dim)

    @helper.fixed_dim_op(nargs=4, is_class_fn=True)
    def forward(self, f64, f16, f8, f4):
        """Aggregate features across feature pyramid.

        Args:
        ----
            f64: features of shape (B, 3, 64, 64), assuming input res is 512x512.
            f16: features of shape (B, 3, 16, 16)
            f8: features of shape (B, 3, 8, 8)
            f4: features of shape (B, 3, 4, 4)

        Return:
        ------
            output features of shape (B, in_dim, 64, 64) with the same res as f64.

        """
        x = self.dc0(f4) + self.proj1(f8)
        x = self.ln1(F.gelu(x))
        x = self.dc1(x) + self.proj2(f16)
        x = self.ln2(F.gelu(x))
        x = self.dc3(x) + self.proj3(f64)
        x = self.ln3(x)

        return x
