from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()), #QuickGELU()
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, need_weights=False, no_softmax=False):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if not need_weights:
            output, attention_weights = self.attn(x, x, x, need_weights=need_weights, attn_mask=self.attn_mask)
        else:
            output, attention_weights = self.attn(x, x, x, need_weights=need_weights, attn_mask=self.attn_mask, no_softmax=no_softmax)
        return output, attention_weights
        # return self.attn(x, x, x, need_weights=need_weights, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, need_weights=False, no_softmax=False):
        out, attn = self.attention(self.ln_1(x), need_weights=need_weights, no_softmax=no_softmax)
        x = x + out
        x = x + self.mlp(self.ln_2(x))
        return x, attn
        # x = x + self.attention(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
        # return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, need_weights=False, no_softmax=False):
        attn_list = []
        for i in range(len(self.resblocks)):
            x, attn = self.resblocks[i](x, need_weights=need_weights, no_softmax=no_softmax)
            if need_weights:
                attn_list.append(attn)
        return x, attn_list
        # return self.resblocks(x)



class TextTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers

        encoder_layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=width * 4,
            dropout=0.0,
            activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)


    def get_extended_attention_mask(self, attention_mask, input_shape):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            d1, d2, d3 = attention_mask.shape
            extended_attention_mask = attention_mask.reshape((d1, 1, d2, d3))
        elif attention_mask.dim() == 2:
            d1, d2 = attention_mask.shape
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask.reshape((d1, 1, 1, d2))
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        return extended_attention_mask

    def forward(self, embedding_output: torch.Tensor, attention_mask: torch.Tensor):
        input_shape = embedding_output.shape
        # extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        # print(extended_attention_mask)
        encoder_outputs = self.encoder(embedding_output, src_key_padding_mask=attention_mask)

        return encoder_outputs

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, n_class_tokens: int = 1):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.n_class_tokens = n_class_tokens
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(n_class_tokens, width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + n_class_tokens, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, return_attn=False, need_weights=False, no_softmax=False, with_proj=True, with_ln=True):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn_list = self.transformer(x, need_weights=need_weights, no_softmax=no_softmax)
        if need_weights:
            return attn_list
        # x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD'

        x = self.ln_post(x[:, :self.n_class_tokens, :])

        if with_proj:
            if self.proj is not None:
                x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 output_dim: int, 
                 n_class_tokens: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length + n_class_tokens - 1

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=output_dim,
            n_class_tokens=n_class_tokens
        )

        self.transformer = TextTransformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_embedding = LayerNorm(transformer_width)
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, output_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_attn=False, need_weights=False, no_softmax=False, with_proj=True, with_ln=True):
        return self.visual(image.type(self.dtype), return_attn=False, need_weights=need_weights, no_softmax=no_softmax, with_proj=with_proj, with_ln=with_ln)

    def encode_text(self, text, attn):

        attn = torch.logical_not(attn.type(torch.bool))
        
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = self.ln_embedding(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).type(self.dtype)
        x = x[:, :1]
        x = x @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(*text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = torch.einsum('abd,cbd->ac', image_features, text_features)
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
