"""Pooling modules for aggregating sequence representations."""

from torch import Tensor, nn


class AttentionPooling(nn.Module):
    """Self-Attention pooling mechanism to aggregate sequence representations."""

    def __init__(self, emb_size: int, num_heads: int, dropout: float = 0.1) -> None:
        """Initialize the AttentionPooling module."""
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query = nn.Embedding(1, emb_size)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """Perform attention pooling.

        Args:
            x: Input sequence tensor of shape (batch_size, seq_len, emb_size).
            key_padding_mask: Padding mask of shape (batch_size, seq_len).

        Returns:
            Tensor: Pooled representation of shape (batch_size, emb_size).

        """
        batch_size = x.size(0)
        query = self.query.weight.unsqueeze(0).expand(batch_size, -1, -1)
        attn_output, _ = self.attention(query, x, x, key_padding_mask=key_padding_mask)
        return attn_output.squeeze(1)
