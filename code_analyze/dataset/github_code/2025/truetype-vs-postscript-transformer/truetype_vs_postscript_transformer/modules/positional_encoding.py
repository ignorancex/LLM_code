"""Provides classes for adding positional encoding to embeddings."""

import math

import torch
from torch import Tensor, nn


class SinusoidPositionalEncoding(nn.Module):
    """Applies sinusoidal positional encoding to token embeddings."""

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000) -> None:
        """Initialize positional encoding with embedding size, dropout, and max length.

        This method sets up positional encoding using sine and cosine functions.
        """
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        """Add positional encoding to the input token embeddings."""
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :],
        )


class LearnedPositionalEncoding(nn.Module):
    """Applies learned positional encoding to token embeddings."""

    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000) -> None:
        """Initialize learnable positional encoding."""
        super().__init__()
        self.pos_embedding = nn.Embedding(maxlen, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding: Tensor) -> Tensor:
        """Add learned positional encoding to the input token embeddings."""
        seq_len = token_embedding.size(1)
        positions = torch.arange(seq_len, device=token_embedding.device).unsqueeze(0)
        return self.dropout(token_embedding + self.pos_embedding(positions))
