"""DataModule for managing FontCollection datasets."""

from typing import Literal

from fontTools.ttLib import TTFont
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from truetype_vs_postscript_transformer.modules.collate_fn import (
    MultiFontPostScriptCollate,
    MultiFontTrueTypeCollate,
)
from truetype_vs_postscript_transformer.torchfont.datasets.multi_font import (
    MultiFontDataset,
)
from truetype_vs_postscript_transformer.torchfont.transforms import (
    Compose,
    DecomposeSegment,
    NormalizeSegment,
    PostScriptSegmentToTensor,
    QuadToCubic,
    SegmentToContourPoint,
    TrueTypeSegmentToTensor,
)
from truetype_vs_postscript_transformer.torchfont.transforms.transforms import (
    ContourPointToTensor,
    NormalizeContourPoint,
    ToContourPoint,
)


class MultiFontLDM(LightningDataModule):
    """DataModule for managing FontCollection datasets."""

    def __init__(
        self,
        fonts: list[TTFont],
        *,
        outline_mode: Literal[
            "postscript_segment",
            "truetype_segment",
            "truetype_atomic_point",
            "truetype_point",
        ] = "postscript_segment",
        batch_size: int = 32,
        num_workers: int = 4,
        codepoints: list[int] | None = None,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int | None = None,
        pad_size: int | None = None,
    ) -> None:
        """Initialize the data module.

        Args:
            fonts: List of fonts to include in the dataset.
            outline_mode: Outline mode to use ("segment" or "point").
            batch_size: Batch size for the dataloaders.
            num_workers: Number of workers for data loading.
            codepoints: List of codepoints to include in the dataset.
            split_ratios: Ratios for splitting the dataset (train, valid, test).
            seed: Random seed for reproducible splits.
            pad_size: Optional padding size for batching.

        """
        super().__init__()
        self.fonts = fonts
        self.outline_mode: Literal[
            "postscript_segment",
            "truetype_segment",
            "truetype_atomic_point",
            "truetype_point",
        ] = outline_mode
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.codepoints = codepoints
        self.split_ratios = split_ratios
        self.seed = seed
        self.pad_size = pad_size

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Set up datasets for train, val, and test splits."""
        if self.outline_mode == "postscript_segment":
            transform = Compose(
                [
                    DecomposeSegment(),
                    NormalizeSegment(),
                    QuadToCubic(),
                    PostScriptSegmentToTensor("zeros"),
                ],
            )
            dataset_outline = "segment"
            collate_outline = "segment"
        elif self.outline_mode == "truetype_segment":
            transform = Compose(
                [
                    DecomposeSegment(),
                    NormalizeSegment(),
                    TrueTypeSegmentToTensor("zeros"),
                ],
            )
            dataset_outline = "segment"
            collate_outline = "segment"
        elif self.outline_mode == "truetype_atomic_point":
            transform = Compose(
                [
                    DecomposeSegment(),
                    SegmentToContourPoint(),
                    NormalizeContourPoint(),
                    ContourPointToTensor(),
                ],
            )
            dataset_outline = "segment"
            collate_outline = "point"
        else:
            transform = Compose(
                [
                    ToContourPoint(),
                    NormalizeContourPoint(),
                    ContourPointToTensor(),
                ],
            )
            dataset_outline = "point"
            collate_outline = "point"

        self.train_dataset = MultiFontDataset(
            fonts=self.fonts,
            outline_mode=dataset_outline,
            codepoints=self.codepoints,
            split="train",
            split_ratios=self.split_ratios,
            seed=self.seed,
            transform=transform,
        )
        self.val_dataset = MultiFontDataset(
            fonts=self.fonts,
            outline_mode=dataset_outline,
            codepoints=self.codepoints,
            split="valid",
            split_ratios=self.split_ratios,
            seed=self.seed,
            transform=transform,
        )
        self.test_dataset = MultiFontDataset(
            fonts=self.fonts,
            outline_mode=dataset_outline,
            codepoints=self.codepoints,
            split="test",
            split_ratios=self.split_ratios,
            seed=self.seed,
            transform=transform,
        )

        if collate_outline == "segment":
            self.collate_fn = MultiFontPostScriptCollate(pad_size=self.pad_size)
        else:
            self.collate_fn = MultiFontTrueTypeCollate(pad_size=self.pad_size)

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
