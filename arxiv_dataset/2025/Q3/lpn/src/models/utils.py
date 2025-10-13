"""Adapted from https://github.com/google/flax/blob/main/examples/nlp_seq/models.py"""

from functools import partial
from typing import Any, Optional

import chex
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass, field


@dataclass
class TransformerLayerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    num_heads: int = 8
    emb_dim_per_head: int = 16
    mlp_dim_factor: float = 4.0
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    use_bias: bool = False
    activation: str = "silu"
    dtype: Any = jnp.float32
    emb_dim: int = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "emb_dim", self.num_heads * self.emb_dim_per_head)


@dataclass
class EncoderTransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    transformer_layer: TransformerLayerConfig = TransformerLayerConfig()
    vocab_size: int = 10
    output_vocab_size: int = 10
    num_layers: int = 2
    latent_dim: int = 32
    variational: bool = False
    max_rows: int = 30
    max_cols: int = 30
    latent_projection_bias: bool = False
    scaled_position_embeddings: bool = False
    dtype: jnp.dtype = field(default=None)
    emb_dim: int = field(default=None)
    max_len: int = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "dtype", self.transformer_layer.dtype)
        object.__setattr__(self, "emb_dim", self.transformer_layer.emb_dim)
        object.__setattr__(self, "max_len", self.max_rows * self.max_cols)


@dataclass
class DecoderTransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    transformer_layer: TransformerLayerConfig = TransformerLayerConfig()
    vocab_size: int = 10
    output_vocab_size: int = 10
    num_layers: int = 2
    max_rows: int = 30
    max_cols: int = 30
    scaled_position_embeddings: bool = False
    next_position_embeddings: bool = True
    next_position_embeddings_new_input_embeds: bool = False
    logits_projection_bias: bool = False
    dtype: jnp.dtype = field(default=None)
    emb_dim: int = field(default=None)
    max_len: int = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "dtype", self.transformer_layer.dtype)
        object.__setattr__(self, "emb_dim", self.transformer_layer.emb_dim)
        object.__setattr__(self, "max_len", self.max_rows * self.max_cols)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
        config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerLayerConfig

    def setup(self) -> None:
        if self.config.activation == "relu":
            self.activation = nn.relu
        elif self.config.activation == "silu":
            self.activation = nn.silu
        else:
            raise ValueError(f"Unsupported activation: {self.config.activation}")

    @nn.compact
    def __call__(self, inputs: chex.Array, dropout_eval: bool) -> chex.Array:
        """Applies Transformer MlpBlock module."""
        config = self.config
        x = inputs
        x = nn.Dense(int(config.mlp_dim_factor * config.emb_dim), config.use_bias, config.dtype)(x)
        x = self.activation(x)
        x = nn.Dense(inputs.shape[-1], config.use_bias, config.dtype)(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=dropout_eval)
        return x


class TransformerLayer(nn.Module):
    """Transformer encoder layer.

    Attributes:
        config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerLayerConfig

    @nn.compact
    def __call__(
        self,
        embeddings: chex.Array,
        dropout_eval: bool,
        pad_mask: Optional[chex.Array] = None,
    ) -> chex.Array:
        """Applies TransformerLayer module.

        Args:
            embeddings: input embeddings.
            dropout_eval: if false dropout is applied otherwise it is not.
            pad_mask: mask to apply on the inputs to avoid attending to padding tokens.

        Returns:
            output after transformer encoder layer.
        """
        config = self.config

        # Attention block.
        assert embeddings.ndim >= 3
        x = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias, use_scale=False)(embeddings)

        x = nn.MultiHeadAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            dropout_rate=config.attention_dropout_rate,
            use_bias=config.use_bias,
            attention_fn=partial(nn.dot_product_attention, force_fp32_for_softmax=True),
        )(inputs_q=x, mask=pad_mask, deterministic=dropout_eval)
        residuals = nn.Dropout(rate=config.dropout_rate)(x, deterministic=dropout_eval)
        embeddings += residuals

        # MLP block.
        x = nn.LayerNorm(dtype=config.dtype, use_bias=config.use_bias, use_scale=False)(embeddings)
        residuals = MlpBlock(config=config)(x, dropout_eval=dropout_eval)
        embeddings += residuals
        return embeddings
