from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass
class PlotOptions:
    pass


@dataclass
class RcParamsOptions:
    fontsize: int = 10


@dataclass
class SubplotsOptions:
    nrows: int = 1
    ncols: int = 1
    squeeze: bool = False
    figsize: tuple = (6.4, 4.8)
    tight_layout: bool = True


def plot_custom(
        axs: Axes,
        x_array: np.ndarray | list,
        array: np.ndarray,
        linestyle: str = "-",
        label: str = None,
        color: str = "C0",
        linewidth: float = 1.5,
        marker: str = "",
        markevery: int = 1,
        title: str = None,
        xlabel: str = None,
        xlim: list = None,
        ylabel: str = None,
        ylim: list = None,
        x_ticklabel_format: bool = None,
):
    axs.plot(
        x_array,
        array,
        linestyle=linestyle,
        marker=marker,
        markevery=markevery,
        label=label,
        color=color,
        linewidth=linewidth,
    )

    if xlim is not None:
        axs.set_xlim(xlim)
    if ylim is not None:
        axs.set_ylim(ylim)

    if x_ticklabel_format is not None:
        axs.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    axs.set_title(title)
    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)
    axs.legend()
    axs.grid(visible=True)


def imshow_custom(
        fig: Figure,
        axs,
        image: np.ndarray[tuple[int, int], np.dtype[np.float_]],
        x_variable: np.ndarray[tuple[int], np.dtype[np.float_]],
        y_variable: np.ndarray[tuple[int], np.dtype[np.float_]],
        title: str = None,
        aspect: str | float = 1.,
        vmin: float = None,
        vmax: float = None,
        is_colorbar: bool = True,
        x_ticks_num: int = 6,
        x_ticks_decimals: int = 2,
        x_label: str = "",
        y_ticks_num: int = 6,
        y_ticks_decimals: int = 2,
        y_label: str = "",
        interpolation: str = "antialiased",
):
    imshow = axs.imshow(
        image,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
    )
    if is_colorbar:
        fig.colorbar(imshow, ax=axs)

    y_ticks = np.linspace(start=0, stop=y_variable.size - 1, num=y_ticks_num, dtype=int)
    y_ticks_labels = np.around(a=y_variable[y_ticks], decimals=y_ticks_decimals)
    axs.set_yticks(ticks=y_ticks, labels=y_ticks_labels)

    x_ticks = np.linspace(start=0, stop=x_variable.size - 1, num=x_ticks_num, dtype=int)
    x_ticks_labels = np.around(a=x_variable[x_ticks], decimals=x_ticks_decimals)
    axs.set_xticks(ticks=x_ticks, labels=x_ticks_labels)

    axs.set_title(title)
    axs.set_aspect(aspect)
    axs.set_ylabel(y_label)
    axs.set_xlabel(x_label)


def savefig_dir_list(
        fig: Figure,
        filename: str,
        directories_list: list[Path],
        subdirectory: str = "",
        fmt: str = "pdf",
        bbox_inches: str = "tight",
):
    for figures_dir in directories_list:
        save_dir = figures_dir / subdirectory
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname=save_dir / filename, format=fmt, bbox_inches=bbox_inches)
    plt.close(fig=fig)


def visualize_and_save(
        visualize_rc_params: dict,
        visualize_subplots_options: dict,
        visualize_func: Callable,
        visualize_func_options: dict,
        save_filename: str,
        save_directory_list: list[Path],
        save_subdirectory: str = "",
        save_fmt: str = "pdf",
        save_bbox_inches: str = "tight",
):
    plt.rcParams['font.size'] = str(visualize_rc_params["fontsize"])
    fig, axes = plt.subplots(**visualize_subplots_options)
    visualize_func(axs=axes[0, 0], **visualize_func_options)
    savefig_dir_list(
        fig=fig,
        filename=save_filename,
        directories_list=save_directory_list,
        subdirectory=save_subdirectory,
        fmt=save_fmt,
        bbox_inches=save_bbox_inches,
    )
    plt.close(fig)
