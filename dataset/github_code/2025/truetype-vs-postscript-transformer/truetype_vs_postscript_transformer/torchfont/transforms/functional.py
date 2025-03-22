"""Functional transformations for glyphs."""

from typing import Literal

import torch
from fontTools.pens.basePen import (
    decomposeQuadraticSegment,
    decomposeSuperBezierSegment,
)
from fontTools.ttLib import TTFont
from torch import Tensor

from truetype_vs_postscript_transformer.torchfont.io.font import (
    C_ARGS_LEN,
    M_L_ARGS_LEN,
    POSTSCRIPT_COMMAND_TYPE_TO_NUM,
    POSTSCRIPT_NUM_TO_COMMAND_TYPE,
    Q_ARGS_LEN,
    TRUETYPE_COMMAND_TYPE_TO_NUM,
    AtomicPostScriptOutline,
    AtomicSegmentOutline,
    AtomicTrueTypeOutline,
    Command,
    Point,
    PointOutline,
    SegmentOutline,
)

PadMethod = Literal["trajectory", "zeros"]

ZERO_POINT = (-1.0, -1.0)


def _handle_curve_to(points: tuple[Point, ...]) -> AtomicSegmentOutline:
    """Handle decomposition for 'curveTo' commands."""
    n = len(points)

    if n < M_L_ARGS_LEN:
        return []
    if n == M_L_ARGS_LEN:
        return [("lineTo", (points[0],))]
    if n == Q_ARGS_LEN:
        return [("qCurveTo", (points[0], points[1]))]
    if n == C_ARGS_LEN:
        return [("curveTo", (points[0], points[1], points[2]))]

    decomposed_segments = decomposeSuperBezierSegment(points)
    return [("curveTo", segment) for segment in decomposed_segments]


def _handle_qcurve_to(points: tuple[Point, ...]) -> AtomicSegmentOutline:
    """Handle decomposition for 'qCurveTo' commands."""
    path = []

    if points[-1] is None:
        last_off_curve = points[-2]
        first_off_curve = points[0]
        implicit_on_curve = (
            (last_off_curve[0] + first_off_curve[0]) / 2,
            (last_off_curve[1] + first_off_curve[1]) / 2,
        )

        path.append(("moveTo", (implicit_on_curve,)))

        points = points[:-1] + (implicit_on_curve,)

    n = len(points)

    if n < M_L_ARGS_LEN:
        return path
    if n == M_L_ARGS_LEN:
        path.append(("lineTo", points))
        return path
    if n == Q_ARGS_LEN:
        path.append(("qCurveTo", points))
        return path

    decomposed_segments = decomposeQuadraticSegment(points)
    path.extend([("qCurveTo", segment) for segment in decomposed_segments])

    return path


def decompose_segment(glyph: SegmentOutline) -> AtomicSegmentOutline:
    """Decompose complex Bezier segments in a glyph into simpler segments.

    Args:
        glyph: A list of commands and their points,
        e.g., [('curveTo', [(x1, y1), (x2, y2), (x, y)])].

    Returns:
        A list of decomposed commands where all 'curveTo' and 'qCurveTo' segments
        are split into their atomic components.

    """
    decomposed_glyph = []

    for command, points in glyph:
        if command == "curveTo":
            decomposed_glyph.extend(_handle_curve_to(points))
        elif command == "qCurveTo":
            decomposed_glyph.extend(_handle_qcurve_to(points))
        else:
            decomposed_glyph.append((command, points))

    return decomposed_glyph


def quad_to_cubic(glyph: AtomicSegmentOutline) -> AtomicPostScriptOutline:
    """Convert quadratic B-spline curves (qCurveTo) to cubic Bézier curves (curveTo).

    Args:
        glyph: A list of commands representing the glyph path.

    Returns:
        A glyph where all `qCurveTo` commands are converted to `curveTo` commands.

    """
    converted_glyph = []
    current_point = ZERO_POINT
    path_start_point = ZERO_POINT

    for command, points in glyph:
        if command == "qCurveTo" and len(points) == Q_ARGS_LEN:
            control_point, end_point = points
            cp1 = (
                current_point[0] + 2 / 3 * (control_point[0] - current_point[0]),
                current_point[1] + 2 / 3 * (control_point[1] - current_point[1]),
            )
            cp2 = (
                end_point[0] + 2 / 3 * (control_point[0] - end_point[0]),
                end_point[1] + 2 / 3 * (control_point[1] - end_point[1]),
            )
            converted_glyph.append(("curveTo", (cp1, cp2, end_point)))
            current_point = end_point
        elif command == "moveTo" and len(points) == M_L_ARGS_LEN:
            path_start_point = points[-1]
            current_point = path_start_point
            converted_glyph.append((command, points))
        elif command == "closePath":
            converted_glyph.append((command, ()))
            current_point = path_start_point
        else:
            converted_glyph.append((command, points))
            if points:
                current_point = points[-1]

    return converted_glyph


def normalize_segment(glyph: SegmentOutline, font: TTFont) -> SegmentOutline:
    """Normalize the glyph path based on the font's unitsPerEm value.

    Args:
        glyph: A list of commands representing the glyph path.
        font: A `TTFont` object representing the font.

    Returns:
        A normalized glyph where all coordinates are divided by unitsPerEm.

    """
    normalized_glyph = []
    upem = font["head"].unitsPerEm  # type: ignore[attr-defined]

    for command, points in glyph:
        normalized_points = tuple((x / upem, y / upem) for x, y in points)
        normalized_glyph.append((command, normalized_points))

    return normalized_glyph


def _pad_truetype_with_trajectory(
    command: Command,
    points: tuple[Point, ...],
    current_point: Point,
    start_point: Point,
) -> tuple[tuple[Point, Point], Point, Point]:
    """Pad points using trajectory-based interpolation."""
    if len(points) == M_L_ARGS_LEN:
        padded_points = (current_point, points[0])
        if command == "moveTo":
            start_point = points[0]
    elif len(points) == Q_ARGS_LEN:
        padded_points = (points[0], points[1])
    elif len(points) == Q_ARGS_LEN:
        padded_points = points
    else:
        padded_points = (current_point, start_point)
    return padded_points, padded_points[-1], start_point


def _pad_truetype_with_zeros(points: tuple[Point, ...]) -> tuple[Point, Point]:
    """Pad points using zero-based padding."""
    if len(points) == M_L_ARGS_LEN:
        return ZERO_POINT, *points
    if len(points) == Q_ARGS_LEN:
        return points
    return ZERO_POINT, ZERO_POINT


def truetype_segment_to_tensor(
    glyph: AtomicTrueTypeOutline,
    method: PadMethod,
) -> tuple[Tensor, Tensor]:
    """Pad the glyph path and convert it to tensors based on the specified method.

    Args:
        glyph: A list of commands representing the glyph path.
        method: The padding method to use. Can be "trajectory" or "zeros".

    Returns:
        A tuple of two tensors:
        - The first tensor contains the command types as integers.
        - The second tensor contains the command arguments as floats.

    """
    command_types = []
    args = []
    current_point = ZERO_POINT
    start_point = ZERO_POINT

    for command, points in glyph:
        if method == "trajectory":
            (
                padded_points,
                current_point,
                start_point,
            ) = _pad_truetype_with_trajectory(
                command,
                points,
                current_point,
                start_point,
            )
        elif method == "zeros":
            padded_points = _pad_truetype_with_zeros(points)

        command_types.append(
            TRUETYPE_COMMAND_TYPE_TO_NUM.get(
                command,
                TRUETYPE_COMMAND_TYPE_TO_NUM["<pad>"],
            ),
        )
        args.append([coord for point in padded_points for coord in point])

    command_type_tensor = torch.tensor(command_types, dtype=torch.int64)
    args_tensor = torch.tensor(args, dtype=torch.float32).view(-1, 4)

    return command_type_tensor, args_tensor


def _pad_postscript_with_trajectory(
    command: Command,
    points: tuple[Point, ...],
    current_point: Point,
    start_point: Point,
) -> tuple[tuple[Point, Point, Point], Point, Point]:
    """Pad points using trajectory-based interpolation."""
    if len(points) == M_L_ARGS_LEN:
        padded_points = (current_point, points[0], points[0])
        if command == "moveTo":
            start_point = points[0]
    elif len(points) == Q_ARGS_LEN:
        padded_points = (points[0], points[0], points[1])
    elif len(points) == C_ARGS_LEN:
        padded_points = points
    else:
        padded_points = (current_point, start_point, start_point)
    return padded_points, padded_points[-1], start_point


def _pad_postscript_with_zeros(points: tuple[Point, ...]) -> tuple[Point, Point, Point]:
    """Pad points using zero-based padding."""
    if len(points) == M_L_ARGS_LEN:
        return ZERO_POINT, ZERO_POINT, *points
    if len(points) == Q_ARGS_LEN:
        return ZERO_POINT, *points
    if len(points) == C_ARGS_LEN:
        return points
    return ZERO_POINT, ZERO_POINT, ZERO_POINT


def postscript_segment_to_tensor(
    glyph: AtomicPostScriptOutline,
    method: PadMethod,
) -> tuple[Tensor, Tensor]:
    """Pad the glyph path and convert it to tensors based on the specified method.

    Args:
        glyph: A list of commands representing the glyph path.
        method: The padding method to use. Can be "trajectory" or "zeros".

    Returns:
        A tuple of two tensors:
        - The first tensor contains the command types as integers.
        - The second tensor contains the command arguments as floats.

    """
    command_types = []
    args = []
    current_point = ZERO_POINT
    start_point = ZERO_POINT

    for command, points in glyph:
        if method == "trajectory":
            (
                padded_points,
                current_point,
                start_point,
            ) = _pad_postscript_with_trajectory(
                command,
                points,
                current_point,
                start_point,
            )
        elif method == "zeros":
            padded_points = _pad_postscript_with_zeros(points)

        command_types.append(
            POSTSCRIPT_COMMAND_TYPE_TO_NUM.get(
                command,
                POSTSCRIPT_COMMAND_TYPE_TO_NUM["<pad>"],
            ),
        )
        args.append([coord for point in padded_points for coord in point])

    command_type_tensor = torch.tensor(command_types, dtype=torch.int64)
    args_tensor = torch.tensor(args, dtype=torch.float32).view(-1, 6)

    return command_type_tensor, args_tensor


def split_glyph_to_paths(
    glyph_tensor: tuple[Tensor, Tensor],
) -> list[tuple[Tensor, Tensor]]:
    """Split a glyph's tensor into paths based on the 'closePath' command.

    Args:
        glyph_tensor: A tuple of two tensors:
        - The first tensor contains the command types as integers.
        - The second tensor contains the flattened coordinates as a 2D tensor of shape.

    Returns:
        A list of tuples, where each tuple represents a path:
        - The first element is a tensor of command types.
        - The second element is a tensor of coordinates for that path.

    """
    command_type_tensor, args_tensor = glyph_tensor
    paths = []
    current_path_commands = []
    current_path_args = []

    for command_type, args in zip(
        command_type_tensor.tolist(),
        args_tensor.tolist(),
        strict=True,
    ):
        current_path_commands.append(command_type)
        current_path_args.append(args)

        if command_type == POSTSCRIPT_COMMAND_TYPE_TO_NUM["closePath"]:
            path_commands_tensor = torch.tensor(
                current_path_commands,
                dtype=torch.int64,
            )
            path_args_tensor = torch.tensor(current_path_args, dtype=torch.float32)
            paths.append((path_commands_tensor, path_args_tensor))

            current_path_commands = []
            current_path_args = []

    return paths


def tensor_to_segment(tensor: tuple[Tensor, Tensor]) -> AtomicPostScriptOutline:
    """Convert separate tensors back to a glyph path.

    Args:
        tensor: A tuple of two tensors:
        - The first tensor contains the command types as integers.
        - The second tensor contains the flattened coordinates as a 2D tensor of shape.

    Returns:
        A list of commands representing the glyph path.

    """
    command_type_tensor, args_tensor = tensor
    glyph = []

    for command_type, coords in zip(
        command_type_tensor.tolist(),
        args_tensor.tolist(),
        strict=True,
    ):
        command = POSTSCRIPT_NUM_TO_COMMAND_TYPE.get(command_type)

        if command in ["moveTo", "lineTo"]:
            points = ((coords[4], coords[5]),)
        elif command == "curveTo":
            points = (
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
            )
        elif command == "closePath":
            points = ()
        else:
            continue

        glyph.append((command, tuple(points)))

    return glyph


def to_contour_point(data: list) -> PointOutline:
    """Convert a list of operations to a structured format.

    Args:
        data: List of operations, such as those recorded by RecordingPointPen.

    Returns:
        List of tuples (Contour ID, Point ID, Location, On/Off curve point).

    """
    result = []
    contour_id = -1
    point_id = 0

    for op, args, _ in data:
        if op == "beginPath":
            contour_id += 1
            point_id = 0
        elif op == "addPoint":
            location = args[0]
            segment_type = args[1]
            on_curve = segment_type is not None
            result.append((contour_id, point_id, location, on_curve))
            point_id += 1
        elif op == "endPath":
            pass

    return result


def normalize_contour_point(outline: PointOutline, font: TTFont) -> PointOutline:
    """Normalize the contour points based on the font's unitsPerEm value.

    Args:
        outline: A list of tuples (Contour ID, Point ID, Location, On/Off curve point).
        font: A `TTFont` object representing the font.

    Returns:
        A list of normalized tuples:
        (Contour ID, Point ID, Location, On/Off curve point),
        where Location is normalized by dividing coordinates by unitsPerEm.

    """
    normalized_outline = []
    upem = font["head"].unitsPerEm  # type: ignore[attr-defined]

    for contour_id, point_id, location, on_curve in outline:
        x, y = location
        normalized_location = (x / upem, y / upem)
        normalized_outline.append((contour_id, point_id, normalized_location, on_curve))

    return normalized_outline


def contour_point_to_tensor(
    outline: PointOutline,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert a normalized PointOutline to separate tensors for each attribute.

    Args:
        outline: A list of tuples (Contour ID, Point ID, Location, On/Off curve point).

    Returns:
        A tuple of four tensors:
        - Contour IDs (Tensor[int]): Tensor of contour IDs.
        - Point IDs (Tensor[int]): Tensor of point IDs.
        - Locations (Tensor[float]): Tensor of locations (x, y).
        - On/Off curve flags (Tensor[int]): Tensor of on-curve flags.

    """
    contour_ids = [contour_id for contour_id, _, _, _ in outline]
    point_ids = [point_id for _, point_id, _, _ in outline]
    locations = [location for _, _, location, _ in outline]
    on_curve_flags = [1 if on_curve else 0 for _, _, _, on_curve in outline]

    contour_tensor = torch.tensor(contour_ids, dtype=torch.int64)
    point_tensor = torch.tensor(point_ids, dtype=torch.int64)
    location_tensor = torch.tensor(locations, dtype=torch.float32).view(-1, 2)
    on_curve_tensor = torch.tensor(on_curve_flags, dtype=torch.int64)

    return contour_tensor, point_tensor, location_tensor, on_curve_tensor


def segment_to_contour_point(glyph: AtomicTrueTypeOutline) -> PointOutline:
    """Convert PostScript outline segments to contour points.

    Args:
        glyph: A list of commands representing the glyph path.

    Returns:
        A list of tuples representing contour points:
        - Contour ID (int)
        - Point ID (int)
        - Location (Tuple[float, float])
        - On-curve flag (bool)

    """
    contour_id = 0
    point_id = 0
    current_point = ZERO_POINT
    start_point = ZERO_POINT
    contour_points = []

    for command, points in glyph:
        if command == "moveTo" and len(points) == M_L_ARGS_LEN:
            start_point = points[0]
            current_point = start_point
            contour_points.append((contour_id, point_id, start_point, True))
            point_id += 1
        elif command == "lineTo" and len(points) == M_L_ARGS_LEN:
            current_point = points[0]
            contour_points.append((contour_id, point_id, current_point, True))
            point_id += 1
        elif command == "qCurveTo" and len(points) == Q_ARGS_LEN:
            control, end = points
            contour_points.append((contour_id, point_id, control, False))
            point_id += 1
            contour_points.append((contour_id, point_id, end, True))
            point_id += 1
            current_point = end
        elif command == "closePath":
            point_id = 0
            contour_id += 1

    return contour_points
