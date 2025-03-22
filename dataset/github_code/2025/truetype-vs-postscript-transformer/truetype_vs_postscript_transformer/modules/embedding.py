"""Provides a class for token embedding in the transformer model."""

from __future__ import annotations

import math
from typing import Literal

from torch import Tensor, nn

from truetype_vs_postscript_transformer.torchfont.io.font import (
    POSTSCRIPT_COMMAND_TYPE_TO_NUM,
    POSTSCRIPT_MAX_ARGS_LEN,
    TRUETYPE_COMMAND_TYPE_TO_NUM,
    TRUETYPE_MAX_ARGS_LEN,
)


class SegmentEmbedding(nn.Module):
    """Embedding layer for the transformer model."""

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
        outline_format: Literal["truetype", "postscript"] = "postscript",
    ) -> None:
        """Initialize the embedding layer with the specified embedding dimension."""
        super().__init__()

        if outline_format == "truetype":
            self.command_type_to_num = TRUETYPE_COMMAND_TYPE_TO_NUM
            self.max_args_len = TRUETYPE_MAX_ARGS_LEN
        elif outline_format == "postscript":
            self.command_type_to_num = POSTSCRIPT_COMMAND_TYPE_TO_NUM
            self.max_args_len = POSTSCRIPT_MAX_ARGS_LEN

        self.command_embedding = nn.Embedding(
            len(self.command_type_to_num),
            embedding_dim,
        )
        self.position_embedding = nn.Linear(self.max_args_len, embedding_dim)
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, glyphs_batch: tuple[Tensor, Tensor]) -> Tensor:
        """Forward pass for the embedding layer."""
        command_indices_tensor, positions_tensor = glyphs_batch
        embedded_commands = self.command_embedding(command_indices_tensor)
        embedded_positions = self.position_embedding(positions_tensor)
        combined_embeddings = (embedded_commands + embedded_positions) * math.sqrt(
            self.embedding_dim,
        )
        return self.dropout(combined_embeddings)


class SegmentUnembedding(nn.Module):
    """Output layer for the transformer model with Classifier."""

    def __init__(
        self,
        embedding_dim: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        outline_format: Literal["truetype", "postscript"] = "postscript",
    ) -> None:
        """Initialize the output layer with the specified embedding dimension."""
        super().__init__()

        if outline_format == "truetype":
            self.command_type_to_num = TRUETYPE_COMMAND_TYPE_TO_NUM
            self.max_args_len = TRUETYPE_MAX_ARGS_LEN
        elif outline_format == "postscript":
            self.command_type_to_num = POSTSCRIPT_COMMAND_TYPE_TO_NUM
            self.max_args_len = POSTSCRIPT_MAX_ARGS_LEN

        self.command_classifier = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, len(self.command_type_to_num)),
        )
        self.position_output_layer = nn.Linear(embedding_dim, self.max_args_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, combined_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for the output layer."""
        dropped_embeddings = self.dropout(combined_embeddings)
        command_logits = self.command_classifier(dropped_embeddings)
        position_output = self.position_output_layer(dropped_embeddings)

        return command_logits, position_output


class PointEmbedding(nn.Module):
    """Embedding layer for the transformer model, specialized for Point inputs."""

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
        maxlen: int = 2500,
    ) -> None:
        """Initialize the embedding layer with the specified embedding dimension."""
        super().__init__()
        self.contour_embedding = nn.Embedding(maxlen, embedding_dim, padding_idx=0)
        self.point_embedding = nn.Embedding(maxlen, embedding_dim, padding_idx=0)
        self.location_embedding = nn.Linear(2, embedding_dim)
        self.on_curve_embedding = nn.Embedding(3, embedding_dim, padding_idx=0)

        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, point_batch: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """Forward pass for the embedding layer."""
        contour_tensor, point_tensor, location_tensor, on_curve_tensor = point_batch

        offset = 1
        contour_tensor = contour_tensor + offset
        point_tensor = point_tensor + offset
        on_curve_tensor = on_curve_tensor + offset

        embedded_contours = self.contour_embedding(contour_tensor)
        embedded_points = self.point_embedding(point_tensor)
        embedded_locations = self.location_embedding(location_tensor)
        embedded_on_curves = self.on_curve_embedding(on_curve_tensor)

        combined_embeddings = (
            embedded_contours
            + embedded_points
            + embedded_locations
            + embedded_on_curves
        ) * math.sqrt(self.embedding_dim)

        return self.dropout(combined_embeddings)


class PointUnembedding(nn.Module):
    """Output layer for the transformer model, specialized for Point outputs."""

    def __init__(
        self,
        embedding_dim: int,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the output layer with the specified embedding dimension."""
        super().__init__()
        self.location_output_layer = nn.Linear(embedding_dim, 2)
        self.on_curve_classifier = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 2),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, combined_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for the output layer."""
        dropped_embeddings = self.dropout(combined_embeddings)
        location_output = self.location_output_layer(dropped_embeddings)
        on_curve_logits = self.on_curve_classifier(dropped_embeddings)

        return location_output, on_curve_logits
