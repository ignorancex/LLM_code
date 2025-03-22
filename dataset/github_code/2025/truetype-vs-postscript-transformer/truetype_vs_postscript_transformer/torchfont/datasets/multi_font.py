"""Multi font dataset module."""

import random
from collections.abc import Callable
from typing import Any, Literal

from fontTools.ttLib import TTFont
from torch.utils.data import Dataset

from truetype_vs_postscript_transformer.torchfont.datasets.single_font import (
    SingleFontDataset,
)
from truetype_vs_postscript_transformer.torchfont.io.font import SegmentOutline


class MultiFontDataset(Dataset):
    """Dataset for multiple fonts, using the union of available codepoints."""

    def __init__(
        self,
        fonts: list[TTFont],
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
            fonts: List of TTFont objects.
            outline_mode: Outline mode to use ("segment" or "point").
            codepoints: Optional list of codepoints to filter.
            split: Subset of the dataset to load ("train", "valid", "test").
            split_ratios: Ratios for splitting the dataset (train, valid, test).
            seed: Random seed for reproducible splits.
            transform: Optional transformation function applied to each glyph.

        """
        self.fonts = fonts
        self.font_datasets = []
        self.indices = []
        self.transform = transform

        all_codepoints_sets = [set(font.getBestCmap().keys()) for font in fonts]
        valid_codepoints = set.union(*all_codepoints_sets)

        if codepoints is not None:
            valid_codepoints = valid_codepoints & set(codepoints)

        self.valid_codepoints = list(valid_codepoints)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)  # noqa: S311

        global_rng = random.Random(seed)  # noqa: S311
        font_seeds = [global_rng.randint(0, 2**32 - 1) for _ in range(len(fonts))]

        for font_index, (font, font_seed) in enumerate(
            zip(fonts, font_seeds, strict=True),
        ):
            font_dataset = SingleFontDataset(
                font=font,
                outline_mode=outline_mode,
                codepoints=self.valid_codepoints,
                split=split,
                split_ratios=split_ratios,
                seed=font_seed,
                transform=transform,
            )
            self.font_datasets.append(font_dataset)

            self.indices.extend([(font_index, i) for i in range(len(font_dataset))])

    def __len__(self) -> int:
        """Get the total number of samples across all fonts."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Any, int, int]:
        """Get the glyph, its codepoint, and the font index."""
        font_index, glyph_idx = self.indices[idx]
        codepoint, glyph = self.font_datasets[font_index][glyph_idx]
        codepoint_index = self.valid_codepoints.index(codepoint)

        return glyph, codepoint_index, font_index
