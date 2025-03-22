"""Masking utilities for the transformer model."""

from __future__ import annotations

import torch
from torch import Tensor

from truetype_vs_postscript_transformer.torchfont.io.font import (
    POSTSCRIPT_COMMAND_TYPE_TO_NUM,
)


def create_mask(
    glyph_tensor: tuple[Tensor, Tensor],
    *,
    causal: bool,
    padding_token: int,
) -> tuple[Tensor | None, Tensor]:
    """Create both the mask and padding mask for a sequence.

    Args:
        glyph_tensor (tuple[Tensor, Tensor]): Input tuple (commands, coords).
        causal (bool): Whether the mask should be causal (for autoregressive tasks).
        padding_token (int): Padding token value.

    Returns:
        tuple[Tensor | None, Tensor]: (mask, padding_mask).

    """
    commands = glyph_tensor[0]
    seq_len = commands.size(1)
    mask = (
        None
        if not causal
        else torch.nn.Transformer.generate_square_subsequent_mask(
            seq_len,
            dtype=torch.bool,
            device=commands.device,
        )
    )
    padding_mask = commands == padding_token
    return mask, padding_mask


def autoregressive_translation_mask(
    src: tuple[Tensor, Tensor],
    tgt: tuple[Tensor, Tensor],
) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
    """Create masks for the source and target sequences for autoregressive tasks.

    Args:
        src (tuple[Tensor, Tensor]): Source tuple (commands, coords).
        tgt (tuple[Tensor, Tensor]): Target tuple (commands, coords).

    Returns:
        tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
            (src_mask, tgt_mask, src_padding_mask, tgt_padding_mask).

    """
    src_mask, src_padding_mask = create_mask(
        glyph_tensor=src,
        causal=False,
        padding_token=POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
    )
    tgt_mask, tgt_padding_mask = create_mask(
        glyph_tensor=tgt,
        causal=True,
        padding_token=POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
    )

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def feedforward_translation_mask(
    src: tuple[Tensor, Tensor],
    tgt: tuple[Tensor, Tensor],
) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
    """Create masks for the source and target sequences for feedforward tasks.

    Args:
        src (tuple[Tensor, Tensor]): Source tuple (commands, coords).
        tgt (tuple[Tensor, Tensor]): Target tuple (commands, coords).

    Returns:
        tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
            (src_mask, tgt_mask, src_padding_mask, tgt_padding_mask).

    """
    src_mask, src_padding_mask = create_mask(
        glyph_tensor=src,
        causal=False,
        padding_token=POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
    )
    tgt_mask, tgt_padding_mask = create_mask(
        glyph_tensor=tgt,
        causal=False,
        padding_token=-100,
    )

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def classification_mask(
    src: tuple[Tensor, Tensor],
) -> tuple[Tensor | None, Tensor]:
    """Create masks for the source sequence for classification tasks.

    Args:
        src (tuple[Tensor, Tensor]): Source tuple (commands, coords).

    Returns:
        tuple[Tensor | None, Tensor]: (src_mask, src_padding_mask).

    """
    src_mask, src_padding_mask = create_mask(
        glyph_tensor=src,
        causal=False,
        padding_token=POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
    )

    return src_mask, src_padding_mask


def autoregressive_autoencoder_masks(
    src: tuple[Tensor, Tensor],
    tgt: tuple[Tensor, Tensor],
) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
    """Create masks for a TransformerDecoder-based Autoencoder.

    Args:
        src (tuple[Tensor, Tensor]): Source tuple (commands, coords).
        tgt (tuple[Tensor, Tensor]): Target tuple (commands, coords).

    Returns:
        tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
            (src_mask, tgt_mask, src_padding_mask, tgt_padding_mask).

    """
    src_mask, src_padding_mask = create_mask(
        glyph_tensor=src,
        causal=False,
        padding_token=POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
    )

    tgt_mask, tgt_padding_mask = create_mask(
        glyph_tensor=tgt,
        causal=True,
        padding_token=POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
    )

    return (
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
    )
