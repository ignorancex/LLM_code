"""Utility functions for plotting glyphs using matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches

from truetype_vs_postscript_transformer.torchfont.io.font import C_ARGS_LEN
from truetype_vs_postscript_transformer.torchfont.transforms.functional import (
    tensor_to_segment,
)

sns.set_theme(style="darkgrid")


if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from torch import Tensor


def _plot_glyph(  # noqa: C901, PLR0912
    ax: Axes,
    tensor: tuple[Tensor, Tensor],
) -> None:
    """Plot a single glyph on the given axes using tensors."""
    glyph = tensor_to_segment(tensor)
    if not glyph:
        return

    path = mpath.Path
    path_data_list = []
    start_point = None
    points_only = []

    try:
        for command, points in glyph:
            if command == "moveTo":
                if points:
                    start_point = points[0]
                    path_data_list.append((path.MOVETO, start_point))
                    points_only.append(start_point)
            elif command == "lineTo":
                for point in points:
                    path_data_list.append((path.LINETO, point))
                    points_only.append(point)
            elif command == "curveTo":
                if len(points) == C_ARGS_LEN:
                    path_data_list.extend((path.CURVE4, point) for point in points)
                    points_only.extend(points)
            elif command == "closePath" and start_point:
                path_data_list.append((path.CLOSEPOLY, start_point))

        if path_data_list:
            codes, vertices = zip(*path_data_list, strict=True)
            path = mpath.Path(vertices, codes)
            patch = patches.PathPatch(
                path,
                fill=True,
                edgecolor="#007AFF",
                facecolor="#5AC8FA",
                alpha=0.8,
            )
            ax.add_patch(patch)
        elif points_only:
            x_vals, y_vals = zip(*points_only, strict=True)
            ax.plot(x_vals, y_vals, "o", color="red")

    except (ValueError, TypeError):
        if points_only:
            x_vals, y_vals = zip(*points_only, strict=True)
            ax.plot(x_vals, y_vals, "o", color="red")


def _create_figure_and_axes(
    num_axes: int = 1,
    figsize: tuple[int, int] = (8, 8),
) -> tuple[Figure, list[Axes]]:
    """Create a figure and the specified number of axes."""
    fig, axes = plt.subplots(1, num_axes, figsize=figsize)
    fig.subplots_adjust(wspace=0.5)

    if num_axes == 1:
        axes = [axes]

    return fig, axes


def _draw_glyphs_on_axes(
    axes: Sequence[Axes],
    glyphs: Sequence[tuple[str, tuple[Tensor, Tensor]]],
) -> None:
    """Draw glyphs on the provided axes."""
    for ax, (title, tensor) in zip(axes, glyphs, strict=True):
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        _plot_glyph(ax, tensor)


def _save_figure(fig: Figure, save_dir: str | Path, file_name: str) -> None:
    """Save a figure to the specified directory with the given file name."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / f"{file_name}.pdf"
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)


def save_glyph_plot(
    tensor: tuple[Tensor, Tensor],
    save_dir: str | Path,
    file_name: str,
) -> None:
    """Draw the glyph and save the plot to a specified directory as a PNG file."""
    fig, axes = _create_figure_and_axes(num_axes=1)
    _draw_glyphs_on_axes(axes, [("Glyph", tensor)])
    _save_figure(fig, save_dir, file_name)


def save_combined_glyph_plot(
    input_tensor: tuple[Tensor, Tensor],
    target_tensor: tuple[Tensor, Tensor],
    output_tensor: tuple[Tensor, Tensor],
    save_dir: str | Path,
    file_name: str,
) -> None:
    """Save a plot of four glyphs to a PNG file."""
    glyphs = [
        ("Input", input_tensor),
        ("Target", target_tensor),
        ("Output", output_tensor),
    ]
    fig, axes = _create_figure_and_axes(num_axes=3, figsize=(12, 4))
    _draw_glyphs_on_axes(axes, glyphs)
    _save_figure(fig, save_dir, file_name)
