from __future__ import annotations
from typing import Any, Literal
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.lib.custom_vars import XAx, OriginEnum, Fig, Axs, Ifm, Acq


def imshow_enhanced(
    image: np.ndarray[tuple[int, int], Any],
    origin: OriginEnum = OriginEnum.LOWER,
    rectangle: np.ndarray[tuple[int, Literal[4]], np.float64] = None,
    points: np.ndarray[tuple[int, Literal[2]], np.float64] = None,
    edge_color: str | None = "g",
    center_color: str | None = "r",
    figure: tuple[Fig, Axs] = None,
    colormap: str = "viridis",
    limits: tuple[float, float] = None,
    text_color: str = "k",
    start_from_one: bool = True,
    font_size: float = None,
    marker: str = "+",
    point_style: dict | None = None,
    color_bar: bool = False,
    rectangle_style: dict | None = None,
) -> tuple[Fig, Axs]:
    """
    Visualizes an image with the following style characteristics.
    rectangle: a list of patches whose format is
               (top coordinate, left coordinate, height, width)
    points: a list of points to plot
    """
    fig, ax = plt.subplots() if figure is None else figure
    if limits is None:
        limits = (np.percentile(image, q=2), np.percentile(image, q=98))
    im = ax.imshow(
        image,
        origin=origin.value,
        vmin=limits[0],
        vmax=limits[1],
        interpolation="nearest",
        extent=(0, image.shape[1], 0, image.shape[0]),
        cmap=colormap,
    )
    if rectangle is not None:
        patches = []
        centers = rectangle[:, :2] + rectangle[:, 2:] / 2
        for ii in np.arange(rectangle.shape[0]):
            rect = mpl.patches.Rectangle(
                xy=(rectangle[ii, 1], rectangle[ii, 0]),
                width=rectangle[ii, 3],
                height=rectangle[ii, 2],
            )
            patches.append(rect)
            label = f"{ii + 1}" if start_from_one else f"{ii}"
            style = {} if font_size is None else {"fontsize": font_size}
            ax.annotate(
                text=label,
                xy=(
                    rectangle[ii, 1] + 0.1 * rectangle[ii, 3],
                    rectangle[ii, 0] + 0.9 * rectangle[ii, 2],
                ),
                horizontalalignment="left",
                verticalalignment="top",
                c=text_color,
                bbox={"boxstyle": "round", "facecolor": "white"},
                **style
            )
            if font_size is not None:
                ax.tick_params(axis='both', which='major',
                               labelsize=font_size - 6)
        if rectangle_style is None:
            rectangle_style = {"linewidth": 1, "edgecolor": edge_color}
        rectangle_style.update({"facecolor": "none"})
        ax.add_collection(PatchCollection(patches, **rectangle_style))
        if center_color is not None:
            ax.scatter(
                centers[:, 1], centers[:, 0], c=center_color, marker=marker
            )
    if points is not None:
        if point_style is None:
            point_style = {"marker": marker, "c": "b"}
        ax.scatter(points[:, 1], points[:, 0], **point_style)
    if color_bar:
        divider = make_axes_locatable(ax)
        color_ax = divider.append_axes("right", size="1%", pad=0.10)
        fig.colorbar(im, ax=color_ax)
        color_ax.set_axis_off()
    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((0, image.shape[0]))
    return fig, ax


def plot_1d(
    y: np.ndarray[tuple[XAx, Any], Any] | np.ndarray[XAx, Any],
    x: np.ndarray[tuple[XAx], Any] = None,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    legend: list[str] = None,
    marker: Any = None,
    line_style: Any = "-",
    font_size: float = 16,
    figure: tuple[Fig, Axs] = None,
    save_filepath: Path | str = None,
    color: str = None,
    x_limit: tuple[float, float] = None,
    reciprocal_axis: bool = False,
    reciprocal_label: str | None = None,
    scientific: bool = False,
) -> tuple[Fig, Axs]:
    y = y[np.newaxis, :] if y.ndim == 1 else y
    fig, ax = plt.subplots() if figure is None else figure
    apply_style(ax=ax)
    plot_options = {"marker": marker, "linestyle": line_style}
    if color is not None:
        plot_options.update({"color": color})
    for ii in np.arange(y.shape[0]):
        if x is not None:
            ax.plot(x, y[ii, :], **plot_options)
        else:
            ax.plot(y[ii, :], **plot_options)
    if x_limit is not None:
        ax.set_xlim(x_limit)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=font_size)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=font_size)
    if title is not None:
        ax.set_title(title, fontsize=font_size)
    if legend is not None:
        ax.legend(legend, fontsize=font_size, frameon=False)
    if scientific:
        ax.ticklabel_format(
            axis='x', style='sci', scilimits=(0, 0), useMathText=True
        )

    if reciprocal_axis:
        ax2 = ax.secondary_xaxis('top', functions=(wn_to_wl, wl_to_wn))
        if reciprocal_label is not None:
            ax2.set_xlabel(reciprocal_label, fontsize=font_size)
        ax2.xaxis.labelpad = 10
        if scientific:
            ax2.ticklabel_format(
                axis='x', style='sci', scilimits=(0, 0), useMathText=True
            )

    if save_filepath is not None:
        plt.savefig(save_filepath)
    return fig, ax


def wn_to_wl(v: np.ndarray):
    return np.divide(1e7, v, out=np.zeros_like(v), where=v != 0)


def wl_to_wn(v: np.ndarray):
    return np.divide(1e-7, v, out=np.zeros_like(v), where=v != 0)


def plot_shaded(
    y: np.ndarray[tuple[XAx, Any], Any],
    x: np.ndarray[tuple[XAx], Any] = None,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    legend: list[str] = None,
    font_size: float = 16,
    figure: tuple[Fig, Axs] = None,
    save_filepath: Path | str = None,
    x_limit: tuple[float, float] = None,
    scientific: bool = False,
    reciprocal_axis: bool = False,
    reciprocal_label: str | None = None,
) -> tuple[Fig, Axs]:
    """Plots a 1d curve with a shaded standard deviation"""
    fig, ax = plt.subplots() if figure is None else figure
    y_mean = np.mean(y, axis=1)
    y_std = np.std(y, axis=1)
    if x is None:
        ax.fill_between(y_mean - y_std, y_mean + y_std, alpha=0.2)
    else:
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)
    return plot_1d(
        y=y_mean,
        figure=(fig, ax),
        x=x,
        legend=legend,
        title=title,
        x_label=x_label,
        y_label=y_label,
        line_style="-",
        font_size=font_size,
        save_filepath=save_filepath,
        x_limit=x_limit,
        scientific=scientific,
        reciprocal_axis=reciprocal_axis,
        reciprocal_label=reciprocal_label,
    )


def plot_shift(
    subimage_centers: np.ndarray[tuple[Literal[2], Ifm], np.float64],
    subimage_size: tuple[int, int],
    reference_spots: np.ndarray[tuple[Literal[2], Acq], np.float64],
    shift: np.ndarray[tuple[Literal[2], Ifm, Acq], np.float64],
    intensity: np.ndarray[tuple[Ifm, Acq], np.float64],
    origin: OriginEnum,
    figure: tuple[Fig, Axs] = None,
    limits: tuple[float, float] = None,
    spot_area: float = 1,
    colormap: str = "viridis",
    title: str = None,
    mask: np.ndarray[tuple[Ifm, Acq], bool] = None,
) -> tuple[Fig, Axs]:
    size = np.array(subimage_size)[np.newaxis, :]
    corners = subimage_centers.T - size / 2
    arrays = (corners, np.broadcast_to(size, corners.shape))
    rectangles = np.concatenate(arrays, axis=1)
    image_size = np.max(subimage_centers, axis=1) + size[0, :] / 2
    image = np.uint8(np.ones(shape=np.array(image_size, dtype=int)) * 255)
    fig, ax = imshow_enhanced(
        image=image,
        rectangle=rectangles,
        center_color=None,
        figure=figure,
        origin=origin,
        colormap="gray",
        limits=(0, 1),
    )
    subimage_corners_v = np.moveaxis(rectangles[:, np.newaxis, :2], 2, 0)
    reference_spots_v = reference_spots[:, np.newaxis, :]
    spots = np.reshape(subimage_corners_v + reference_spots_v, (2, -1))
    if mask is None:
        mask = np.ones_like(intensity, dtype=bool)
    mask_v = mask.flatten()
    intensity_v = intensity.flatten()
    if limits is None:
        limits = (np.min(intensity_v), np.max(intensity_v))
    color_map = plt.get_cmap(colormap)
    im = ax.scatter(
        x=spots[1, mask_v],
        y=spots[0, mask_v],
        c=intensity_v[mask_v],
        marker="o",
        vmin=limits[0],
        vmax=limits[1],
        s=spot_area,
        cmap=color_map,
    )
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="2%", pad=0.05)
    colorbar_ax.axis("off")
    fig.colorbar(im, ax=colorbar_ax)
    if title is not None:
        ax.set_title(label=title)
    return fig, ax


def apply_style(ax: Axs) -> None:
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_facecolor("#EAEAF2")
    ax.grid(color="w")
