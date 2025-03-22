"""Transforms for processing glyphs."""

from collections.abc import Callable

from fontTools.ttLib import TTFont
from torch import Tensor

from truetype_vs_postscript_transformer.torchfont.io.font import (
    AtomicPostScriptOutline,
    AtomicSegmentOutline,
    AtomicTrueTypeOutline,
    PointOutline,
    SegmentOutline,
)
from truetype_vs_postscript_transformer.torchfont.transforms import functional as F


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms: list[Callable]) -> None:
        """Initialize the transform."""
        self.transforms = transforms

    def __call__(
        self,
        glyph: SegmentOutline,
        font: TTFont,
    ) -> SegmentOutline | tuple[Tensor, Tensor]:
        """Apply the transformations to the glyph."""
        for t in self.transforms:
            glyph = t(glyph, font)
        return glyph

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class DecomposeSegment:
    """Decompose complex Bezier segments in glyphs."""

    def __call__(self, glyph: SegmentOutline, _: TTFont) -> AtomicSegmentOutline:
        """Decompose the glyph into simpler segments."""
        return F.decompose_segment(glyph)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class QuadToCubic:
    """Convert quadratic Bézier curves to cubic Bézier curves."""

    def __call__(
        self,
        glyph: AtomicSegmentOutline,
        _: TTFont,
    ) -> AtomicPostScriptOutline:
        """Convert all `qCurvedTo` commands to `curveTo` commands."""
        return F.quad_to_cubic(glyph)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class NormalizeSegment:
    """Normalize the glyph path to fit within a standard coordinate range."""

    def __call__(self, glyph: SegmentOutline, font: TTFont) -> SegmentOutline:
        """Normalize the glyph using the font's units per em."""
        return F.normalize_segment(glyph, font)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class TrueTypeSegmentToTensor:
    """Convert a glyph path to a PyTorch tensor."""

    def __init__(self, method: F.PadMethod) -> None:
        """Initialize the transform."""
        self.method: F.PadMethod = method

    def __call__(
        self,
        glyph: AtomicTrueTypeOutline,
        _: TTFont,
    ) -> tuple[Tensor, Tensor]:
        """Convert the glyph to separate tensors for commands and arguments."""
        return F.truetype_segment_to_tensor(glyph, self.method)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return f"{self.__class__.__name__}(method='{self.method}')"


class PostScriptSegmentToTensor:
    """Convert a glyph path to a PyTorch tensor."""

    def __init__(self, method: F.PadMethod) -> None:
        """Initialize the transform."""
        self.method: F.PadMethod = method

    def __call__(
        self,
        glyph: AtomicPostScriptOutline,
        _: TTFont,
    ) -> tuple[Tensor, Tensor]:
        """Convert the glyph to separate tensors for commands and arguments."""
        return F.postscript_segment_to_tensor(glyph, self.method)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return f"{self.__class__.__name__}(method='{self.method}')"


class SplitGlyphToPaths:
    """Split a glyph's tensor into paths based on the 'closePath' command."""

    def __call__(
        self,
        glyph_tensor: tuple[Tensor, Tensor],
        _: TTFont,
    ) -> list[tuple[Tensor, Tensor]]:
        """Split the glyph's tensor into paths based on the 'closePath' command."""
        return F.split_glyph_to_paths(glyph_tensor)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class TensorToSegment:
    """Convert a PyTorch tensor back to a glyph path."""

    def __call__(
        self,
        tensor: tuple[Tensor, Tensor],
        _: TTFont,
    ) -> AtomicPostScriptOutline:
        """Convert the tensors back to a glyph."""
        return F.tensor_to_segment(tensor)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class ToContourPoint:
    """Convert a list of operations to a structured contour format."""

    def __call__(self, data: list, _: TTFont) -> PointOutline:
        """Convert operations to contour points."""
        return F.to_contour_point(data)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class NormalizeContourPoint:
    """Normalize contour points using font's unitsPerEm value."""

    def __call__(self, outline: PointOutline, font: TTFont) -> PointOutline:
        """Normalize the contour points."""
        return F.normalize_contour_point(outline, font)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class ContourPointToTensor:
    """Convert contour points to separate tensors."""

    def __call__(
        self,
        outline: PointOutline,
        _: TTFont,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convert the contour points to tensors."""
        return F.contour_point_to_tensor(outline)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"


class SegmentToContourPoint:
    """Convert a glyph segment to a contour point."""

    def __call__(self, glyph: AtomicTrueTypeOutline, _: TTFont) -> PointOutline:
        """Convert the glyph segment to a contour point."""
        return F.segment_to_contour_point(glyph)

    def __repr__(self) -> str:
        """Get the string representation of the transform."""
        return self.__class__.__name__ + "()"
