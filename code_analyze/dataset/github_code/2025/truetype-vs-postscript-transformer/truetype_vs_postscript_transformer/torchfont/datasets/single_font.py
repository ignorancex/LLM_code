"""Single font dataset module."""

import random
from collections.abc import Callable
from typing import Any, Literal

from fontTools.ttLib import TTFont
from torch.utils.data import Dataset

from truetype_vs_postscript_transformer.torchfont.io.font import (
    SegmentOutline,
    extract_point_outline,
    extract_segment_outline,
)


class SingleFontDataset(Dataset):
    """Dataset for a single font."""

    def __init__(
        self,
        font: TTFont,
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
            font: Font file as a TTFont object.
            outline_mode: Outline mode to use ("segment" or "point").
            codepoints: Optional list of codepoints to filter.
            split: Subset of the dataset to load ("train", "valid", "test").
            split_ratios: Ratios for splitting the dataset (train, valid, test).
            seed: Random seed for reproducible splits.
            transform: Optional transformation function applied to each glyph.

        """
        self.font = font
        self.outline_mode = outline_mode
        self.transform = transform

        cmap = self.font.getBestCmap()
        all_codepoints = list(cmap.keys())

        if codepoints is not None:
            self.codepoints = list(set(all_codepoints) & set(codepoints))
        else:
            self.codepoints = all_codepoints

        split_ratios_sum = sum(split_ratios)
        split_ratios = (
            split_ratios[0] / split_ratios_sum,
            split_ratios[1] / split_ratios_sum,
            split_ratios[2] / split_ratios_sum,
        )

        if seed is not None:
            random.seed(seed)

        shuffled_codepoints = random.sample(
            self.codepoints,
            len(self.codepoints),
        )
        train_end = int(split_ratios[0] * len(shuffled_codepoints))
        valid_end = train_end + int(split_ratios[1] * len(shuffled_codepoints))

        self.splits = {
            "train": shuffled_codepoints[:train_end],
            "valid": shuffled_codepoints[train_end:valid_end],
            "test": shuffled_codepoints[valid_end:],
        }

        if split is not None:
            self.codepoints = self.splits[split]

    def __len__(self) -> int:
        """Get the number of codepoints."""
        return len(self.codepoints)

    def __getitem__(self, idx: int) -> tuple[int, Any]:
        """Get the glyph path and its corresponding codepoint."""
        codepoint = self.codepoints[idx]

        if self.outline_mode == "segment":
            glyph = extract_segment_outline(self.font, codepoint)
        else:
            glyph = extract_point_outline(self.font, codepoint)

        if self.transform is not None and glyph is not None:
            glyph = self.transform(glyph, self.font)

        return codepoint, glyph
