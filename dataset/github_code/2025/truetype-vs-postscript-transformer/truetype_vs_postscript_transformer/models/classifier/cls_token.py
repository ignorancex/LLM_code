"""Font classifier model using CLS token."""

from typing import Literal

import torch
from torch import Tensor, nn

from truetype_vs_postscript_transformer.modules.embedding import SegmentEmbedding
from truetype_vs_postscript_transformer.modules.positional_encoding import (
    LearnedPositionalEncoding,
)


class ClsTokenFontClassifier(nn.Module):
    """Font classifier using dynamically added CLS token."""

    def __init__(
        self,
        num_layers: int,
        emb_size: int,
        nhead: int,
        num_classes: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        outline_format: Literal["truetype", "postscript"] = "postscript",
    ) -> None:
        """Initialize the FontClassifier model."""
        super().__init__()
        self.emb_size = emb_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))  # CLS token
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_size,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.src_tok_emb = SegmentEmbedding(
            emb_size,
            dropout=dropout,
            outline_format=outline_format,
        )
        self.positional_encoding = LearnedPositionalEncoding(emb_size, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )

    def forward(
        self,
        *,
        src: tuple[Tensor, Tensor],
        src_mask: Tensor | None = None,
        src_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass for the FontClassifierWithCLS model.

        Args:
            src: Input tensor as a tuple (command tokens, positional tokens).
            src_mask: Source mask tensor.
            src_padding_mask: Source padding mask tensor.

        Returns:
            Tensor: Class probabilities.

        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))

        batch_size = src_emb.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src_emb = torch.cat([cls_tokens, src_emb], dim=1)

        if src_mask is not None:
            cls_mask = torch.ones(
                1,
                src_mask.size(1),
                device=src_mask.device,
                dtype=src_mask.dtype,
            )
            src_mask = torch.cat([cls_mask, src_mask], dim=0)
        if src_padding_mask is not None:
            cls_padding_mask = torch.zeros(
                batch_size,
                1,
                device=src_padding_mask.device,
                dtype=src_padding_mask.dtype,
            )
            src_padding_mask = torch.cat([cls_padding_mask, src_padding_mask], dim=1)

        memory = self.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask,
        )

        cls_representation = memory[:, 0]
        return self.classifier(cls_representation)
