"""Font I/O module."""

from typing import Literal

from fontTools.pens.recordingPen import (
    DecomposingRecordingPen,
    DecomposingRecordingPointPen,
)
from fontTools.ttLib import TTFont

Point = tuple[float, float]

CommandBOS = Literal["<bos>"]
CommandEOS = Literal["<eos>"]
CommandPAD = Literal["<pad>"]
CommandMoveTo = Literal["moveTo"]
CommandLineTo = Literal["lineTo"]
CommandQCurveTo = Literal["qCurveTo"]
CommandCurveTo = Literal["curveTo"]
CommandClosePath = Literal["closePath"]

Command = (
    CommandBOS
    | CommandEOS
    | CommandPAD
    | CommandMoveTo
    | CommandLineTo
    | CommandQCurveTo
    | CommandCurveTo
    | CommandClosePath
)

SegmentBOS = tuple[CommandBOS, tuple[()]]
SegmentEOS = tuple[CommandEOS, tuple[()]]
SegmentPAD = tuple[CommandPAD, tuple[()]]

SegmentMoveTo = tuple[CommandMoveTo, tuple[Point]]
SegmentLineTo = tuple[CommandLineTo, tuple[Point]]
SegmentQCurveTo = tuple[CommandQCurveTo, tuple[Point, ...]]
SegmentCurveTo = tuple[CommandCurveTo, tuple[Point, ...]]
SegmentClosePath = tuple[CommandClosePath, tuple[()]]

AtomicSegmentQCurveTo = tuple[CommandQCurveTo, tuple[Point, Point]]
AtomicSegmentCurveTo = tuple[CommandCurveTo, tuple[Point, Point, Point]]

Segment = (
    SegmentBOS
    | SegmentEOS
    | SegmentPAD
    | SegmentMoveTo
    | SegmentLineTo
    | SegmentQCurveTo
    | SegmentCurveTo
    | SegmentClosePath
)

TrueTypeSegment = (
    SegmentBOS
    | SegmentEOS
    | SegmentPAD
    | SegmentMoveTo
    | SegmentLineTo
    | SegmentQCurveTo
    | SegmentClosePath
)

PostScriptSegment = (
    SegmentBOS
    | SegmentEOS
    | SegmentPAD
    | SegmentMoveTo
    | SegmentLineTo
    | SegmentCurveTo
    | SegmentCurveTo
    | SegmentClosePath
)

AtomicSegment = (
    SegmentBOS
    | SegmentEOS
    | SegmentPAD
    | SegmentMoveTo
    | SegmentLineTo
    | AtomicSegmentQCurveTo
    | AtomicSegmentCurveTo
    | SegmentClosePath
)

AtomicTrueTypeSegment = (
    SegmentBOS
    | SegmentEOS
    | SegmentPAD
    | SegmentMoveTo
    | SegmentLineTo
    | AtomicSegmentQCurveTo
    | SegmentClosePath
)

AtomicPostScriptSegment = (
    SegmentBOS
    | SegmentEOS
    | SegmentPAD
    | SegmentMoveTo
    | SegmentLineTo
    | AtomicSegmentCurveTo
    | SegmentClosePath
)

SegmentOutline = list[Segment]
TrueTypeOutline = list[TrueTypeSegment]
PostScriptOutline = list[PostScriptSegment]
AtomicSegmentOutline = list[AtomicSegment]
AtomicTrueTypeOutline = list[AtomicTrueTypeSegment]
AtomicPostScriptOutline = list[AtomicPostScriptSegment]

ContourPoint = tuple[int, int, Point, bool]
PointOutline = list[ContourPoint]

TRUETYPE_COMMAND_TYPE_TO_NUM = {
    "<bos>": 0,
    "<eos>": 1,
    "<pad>": 2,
    "moveTo": 3,
    "lineTo": 4,
    "qCurveTo": 5,
    "closePath": 6,
}

TRUETYPE_NUM_TO_COMMAND_TYPE = {v: k for k, v in TRUETYPE_COMMAND_TYPE_TO_NUM.items()}

POSTSCRIPT_COMMAND_TYPE_TO_NUM = {
    "<bos>": 0,
    "<eos>": 1,
    "<pad>": 2,
    "moveTo": 3,
    "lineTo": 4,
    "curveTo": 5,
    "closePath": 6,
}

POSTSCRIPT_NUM_TO_COMMAND_TYPE = {
    v: k for k, v in POSTSCRIPT_COMMAND_TYPE_TO_NUM.items()
}

M_L_ARGS_LEN = 1
Q_ARGS_LEN = 2
C_ARGS_LEN = 3
Z_ARGS_LEN = 0

TRUETYPE_MAX_ARGS_LEN = max(M_L_ARGS_LEN, Q_ARGS_LEN, Z_ARGS_LEN) * 2
POSTSCRIPT_MAX_ARGS_LEN = max(M_L_ARGS_LEN, Q_ARGS_LEN, C_ARGS_LEN, Z_ARGS_LEN) * 2


def extract_segment_outline(
    font: TTFont,
    codepoint: int,
) -> SegmentOutline | None:
    """Extract path data for a glyph given its Unicode codepoint.

    Args:
        font: A `TTFont` object representing the font.
        codepoint: A Unicode codepoint representing the desired glyph.

    Returns:
        A list of commands (`Glyph`) representing the glyph path, or `None` if the
        glyph is not available in the font.

    """
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    glyph_name = cmap.get(codepoint)
    if glyph_name is None or glyph_name not in glyph_set:
        return None

    glyph = glyph_set[glyph_name]
    pen = DecomposingRecordingPen(glyph_set)
    glyph.draw(pen)
    return pen.value


def extract_point_outline(
    font: TTFont,
    codepoint: int,
) -> list | None:
    """Extract TrueType outline data for a glyph given its Unicode codepoint.

    This function uses DecomposingRecordingPointPen to decompose composite glyphs
    and record TrueType outline data as a sequence of points and segment types.

    Args:
        font: A `TTFont` object representing the font.
        codepoint: A Unicode codepoint representing the desired glyph.

    Returns:
        A list of recorded point operations representing the glyph outline,
        or `None` if the glyph is not available in the font.

    """
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    glyph_name = cmap.get(codepoint)
    if glyph_name is None or glyph_name not in glyph_set:
        return None

    glyph = glyph_set[glyph_name]
    pen = DecomposingRecordingPointPen(glyph_set)
    glyph.drawPoints(pen)
    return pen.value
