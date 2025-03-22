"""Font pair dataset module."""

import random
from collections.abc import Callable
from typing import Any, Literal

from fontTools.ttLib import TTFont
from torch.utils.data import Dataset

from truetype_vs_postscript_transformer.torchfont.datasets.single_font import (
    SingleFontDataset,
)
from truetype_vs_postscript_transformer.torchfont.io.font import SegmentOutline


class FontPairDataset(Dataset):
    """Dataset for a pair of fonts."""

    def __init__(
        self,
        src_font: TTFont,
        target_font: TTFont,
        *,
        outline_mode: Literal["segment", "point"] = "segment",
        codepoints: list[int] | None = None,
        split: Literal["train", "valid", "test"] | None = None,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int | None = None,
        transform: Callable[[SegmentOutline, TTFont], Any] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            src_font: Source font file as a TTFont object.
            target_font: Target font file as a TTFont object.
            outline_mode: Outline mode to use ("segment" or "point").
            codepoints: Optional list of codepoints to filter.
            split: Subset of the dataset to load ("train", "valid", "test").
            split_ratios: Ratios for splitting the dataset (train, valid, test).
            seed: Random seed for reproducible splits.
            transform: Optional transformation function applied to each glyph.

        """
        src_cmap = src_font.getBestCmap()
        target_cmap = target_font.getBestCmap()
        src_codepoints = set(src_cmap.keys())
        target_codepoints = set(target_cmap.keys())

        common_codepoints = list(src_codepoints & target_codepoints)
        if codepoints is not None:
            common_codepoints = list(set(common_codepoints) & set(codepoints))

        if seed is None:
            seed = random.randint(0, 2**32 - 1)  # noqa: S311

        self.src_dataset = SingleFontDataset(
            font=src_font,
            outline_mode=outline_mode,
            codepoints=common_codepoints,
            split=split,
            split_ratios=split_ratios,
            seed=seed,
            transform=transform,
        )
        self.target_dataset = SingleFontDataset(
            font=target_font,
            outline_mode=outline_mode,
            codepoints=common_codepoints,
            split=split,
            split_ratios=split_ratios,
            seed=seed,
            transform=transform,
        )

        self.common_codepoints = self.src_dataset.codepoints

    def __len__(self) -> int:
        """Get the number of common codepoints."""
        return len(self.common_codepoints)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Get a pair of source and target glyphs."""
        _, src_glyph = self.src_dataset[idx]
        _, target_glyph = self.target_dataset[idx]

        return src_glyph, target_glyph
