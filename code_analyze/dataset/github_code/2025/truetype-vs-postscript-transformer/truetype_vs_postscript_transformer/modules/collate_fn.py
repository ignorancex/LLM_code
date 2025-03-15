"""Collate function for dataloaders."""

import torch
from torch import Tensor

from truetype_vs_postscript_transformer.torchfont.io.font import (
    POSTSCRIPT_COMMAND_TYPE_TO_NUM,
)


class FontPairPostScriptCollate:
    """Collate function wrapper for style transfer dataloaders."""

    def __init__(self, pad_size: int | None = None) -> None:
        """Initialize the collate function with optional pad size."""
        self.pad_size = pad_size

    def __call__(
        self,
        batch: list[tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]],
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Collate function for dataloaders that pads the batch to the same size."""
        src, target = zip(*batch, strict=True)
        target = _add_special_tokens_postscript_outline_batch(target)
        src_padded = _pad_postscript_outline(src, self.pad_size)
        target_padded = _pad_postscript_outline(target, self.pad_size)
        return src_padded, target_padded


class MultiFontPostScriptCollate:
    """Collate function wrapper for classification dataloaders."""

    def __init__(self, pad_size: int | None = None) -> None:
        """Initialize the collate function with optional pad size."""
        self.pad_size = pad_size

    def __call__(
        self,
        batch: list[tuple[tuple[Tensor, Tensor], int, int]],
    ) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
        """Collate function for dataloaders that pads the batch to the same size."""
        glyph, codepoint, font_index = zip(*batch, strict=True)
        glyph_padded = _pad_postscript_outline(glyph, self.pad_size)
        return glyph_padded, torch.tensor(codepoint), torch.tensor(font_index)


class MultiFontPathPostScriptCollate:
    """Collate function wrapper for classification dataloaders."""

    def __init__(self, pad_size: int | None = None) -> None:
        """Initialize the collate function with optional pad size."""
        self.pad_size = pad_size

    def __call__(
        self,
        batch: list[tuple[list[tuple[Tensor, Tensor]], int, int]],
    ) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
        """Collate function for dataloaders that pads the batch to the same size."""
        path, codepoint, font_index = zip(
            *[
                (path, codepoint, font_index)
                for glyphs, codepoint, font_index in batch
                for path in glyphs
            ],
            strict=True,
        )

        path_padded = _pad_postscript_outline(path, self.pad_size)

        return path_padded, torch.tensor(codepoint), torch.tensor(font_index)


def _add_special_tokens_postscript_outline_batch(
    batch: list[tuple[Tensor, Tensor]],
) -> list[tuple[Tensor, Tensor]]:
    """Add <bos> and <eos> tokens to each sequence in the batch."""
    return [
        _add_special_tokens_postscript_outline(commands, coordinates)
        for commands, coordinates in batch
    ]


def _add_special_tokens_postscript_outline(
    commands: Tensor,
    coordinates: Tensor,
) -> tuple[Tensor, Tensor]:
    """Add <bos> and <eos> tokens to a single sequence."""
    bos_command = torch.tensor(
        [POSTSCRIPT_COMMAND_TYPE_TO_NUM["<bos>"]],
        dtype=commands.dtype,
        device=commands.device,
    )
    eos_command = torch.tensor(
        [POSTSCRIPT_COMMAND_TYPE_TO_NUM["<eos>"]],
        dtype=commands.dtype,
        device=commands.device,
    )
    bos_coordinates = torch.zeros(
        (1, coordinates.size(1)),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    eos_coordinates = torch.zeros(
        (1, coordinates.size(1)),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )

    commands = torch.cat([bos_command, commands, eos_command], dim=0)
    coordinates = torch.cat([bos_coordinates, coordinates, eos_coordinates], dim=0)

    return commands, coordinates


def _pad_postscript_outline(
    data: list[tuple[Tensor, Tensor]],
    pad_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Pad commands and coordinates to the same length."""
    commands, coordinates = zip(*data, strict=True)

    if pad_size is None:
        pad_size = max(cmd.size(0) for cmd in commands)

    padded_commands = torch.stack(
        [
            torch.nn.functional.pad(
                cmd,
                (0, pad_size - cmd.size(0)),
                value=POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
            )
            for cmd in commands
        ],
    )

    padded_coordinates = torch.stack(
        [
            torch.nn.functional.pad(
                coord,
                (0, 0, 0, pad_size - coord.size(0)),
                value=0.0,
            )
            for coord in coordinates
        ],
    )

    return padded_commands, padded_coordinates


class FontPairTrueTypeCollate:
    """Collate function wrapper for style transfer dataloaders (TrueType outlines)."""

    def __init__(self, pad_size: int | None = None) -> None:
        """Initialize the collate function with optional pad size."""
        self.pad_size = pad_size

    def __call__(
        self,
        batch: list[
            tuple[
                tuple[Tensor, Tensor, Tensor, Tensor],
                tuple[Tensor, Tensor, Tensor, Tensor],
            ]
        ],
    ) -> tuple[
        tuple[Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor],
    ]:
        """Collate function for dataloaders that pads the batch to the same size."""
        src, target = zip(*batch, strict=True)

        src_padded = _pad_truetype_outline(src, self.pad_size)
        target_padded = _pad_truetype_outline(target, self.pad_size)

        return src_padded, target_padded


class MultiFontTrueTypeCollate:
    """Collate function wrapper for classification dataloaders (TrueType outlines)."""

    def __init__(self, pad_size: int | None = None) -> None:
        """Initialize the collate function with optional pad size."""
        self.pad_size = pad_size

    def __call__(
        self,
        batch: list[tuple[tuple[Tensor, Tensor, Tensor, Tensor], int, int]],
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], Tensor, Tensor]:
        """Collate function for dataloaders that pads the batch to the same size."""
        glyph, codepoint, font_index = zip(*batch, strict=True)

        glyph_padded = _pad_truetype_outline(glyph, self.pad_size)

        return glyph_padded, torch.tensor(codepoint), torch.tensor(font_index)


def _pad_truetype_outline(
    data: list[tuple[Tensor, Tensor, Tensor, Tensor]],
    pad_size: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad Contour IDs, Point IDs, Locations, and On-Curve flags to the same length."""
    contour_ids, point_ids, locations, on_curve_flags = zip(*data, strict=True)

    if pad_size is None:
        pad_size = max(c.size(0) for c in contour_ids)

    padded_contour_ids = torch.stack(
        [
            torch.nn.functional.pad(
                c,
                (0, pad_size - c.size(0)),
                value=-1,
            )
            for c in contour_ids
        ],
    )

    padded_point_ids = torch.stack(
        [
            torch.nn.functional.pad(
                p,
                (0, pad_size - p.size(0)),
                value=-1,
            )
            for p in point_ids
        ],
    )

    padded_locations = torch.stack(
        [
            torch.nn.functional.pad(
                loc,
                (0, 0, 0, pad_size - loc.size(0)),
                value=0.0,
            )
            for loc in locations
        ],
    )

    padded_on_curve_flags = torch.stack(
        [
            torch.nn.functional.pad(
                flag,
                (0, pad_size - flag.size(0)),
                value=-1,
            )
            for flag in on_curve_flags
        ],
    )

    return padded_contour_ids, padded_point_ids, padded_locations, padded_on_curve_flags
