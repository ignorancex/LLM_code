"""
plotting functions
"""


import os.path
from tqdm import trange
from typing import Union, List, Tuple, Dict, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ecombine.eprocesses import (
    eprocess_exch_universal,
    eprocess_exch_conformal,
)


# globals
PLOT_DEFAULT_KWARGS = dict(
    alpha=0.7,
    linewidth=2,
    height=4,
    aspect=2,
)
PLOT_DEFAULT_COLORS = sns.color_palette("colorblind")


def set_theme(style="whitegrid", palette="colorblind", font="Avenir", font_scale=1.5):
    """Set the default plotting themes for the package.

    A wrapper around `seaborn.set_theme()` with custom defaults.
    """
    return sns.set_theme(
        style=style,
        palette=palette,
        font=font,
        font_scale=font_scale,
    )


def plot_eprocess_from_df(
        df: pd.DataFrame,
        threshold: float = 0,
        title: str = "E-process",
        **plot_kwargs
) -> sns.FacetGrid:
    """Plot e-processes from a "tall" pandas data frame."""

    # overwrite default plotting args if necessary
    kwargs = PLOT_DEFAULT_KWARGS.copy()
    for k, v in plot_kwargs.items():
        kwargs[k] = v

    fg = sns.relplot(
        x="Time",
        y="Value",
        hue="E-process",
        style="E-process",
        kind="line",
        data=df,
        **kwargs
    )
    fg.ax.set(
        title=title,
        ylabel="E-Process (log-scale)",
        yscale="log",
    )

    fg.ax.axhline(
            y=1, color="black", alpha=PLOT_DEFAULT_KWARGS["alpha"] * 0.5,
            linestyle="dashed", linewidth=PLOT_DEFAULT_KWARGS["linewidth"],
        )
    if threshold > 0:
        fg.ax.axhline(
            y=threshold, color="gray", alpha=PLOT_DEFAULT_KWARGS["alpha"] * 0.5,
            linestyle="dashed", linewidth=PLOT_DEFAULT_KWARGS["linewidth"],
        )
    return fg


def plot_eprocess(
        e: Union[np.ndarray, List[np.ndarray]],
        e_label: Union[str, List[str]] = None,
        time_index: np.ndarray = None,
        threshold: float = 0,
        title: str = "E-process",
        **plot_kwargs
) -> sns.FacetGrid:
    """Plot e-processes provided as a list of numpy arrays
    and return the FacetGrid object for further modifications."""

    # each row is an e-process
    e = np.array(e)
    if len(e.shape) == 1:
        e = e[np.newaxis, :]
        e_label = [e_label]
    T = e.shape[1]

    # construct a "tall" data frame
    if time_index is None:
        time_index = np.arange(1, T + 1)
    e_dfs = []
    for e_proc, e_lab in zip(e, e_label):
        e_dfs.append(pd.DataFrame({
            "Time": time_index,
            "E-process": e_lab,
            "Value": e_proc,
        }))
    df = pd.concat(e_dfs)
    return plot_eprocess_from_df(df, threshold=threshold, title=title, **plot_kwargs)


def plot_eprocesses_exch(
        x: np.ndarray,
        jumps: Tuple[float] = (0.1, 0.01, 0.001),
        jumper_weights: Tuple[float] = (1/3, 1/3, 1/3),
        title: str = "E-processes for testing exchangeability",
        n_repeats: int = 1,
        rng: np.random.Generator = None,  # for conformal
        **plot_kwargs
):
    """Plot various e-processes (UI & simple jumper)
    for testing exchangeability."""
    eprocesses, names = [], []

    repeat_iter = trange(n_repeats, desc="repeated runs") if n_repeats > 1 else range(n_repeats)
    for _ in repeat_iter:
        # UI
        e_ui = eprocess_exch_universal(x)
        eprocesses.append(e_ui)
        names.append("UI")
        # Conformal variants
        e_confs = [
            eprocess_exch_conformal(
                x,
                jump=jump,
                jumper_weights=jumper_weights,
                rng=rng,
            )
            for jump in jumps
        ]
        eprocesses.extend(e_confs)
        names.extend([f"Conformal-j{jump:g}" for jump in jumps])
    fg = plot_eprocess(eprocesses, names, title=title, **plot_kwargs)
    return fg, eprocesses


def plot_stopped_e_values(
        dfs: Dict[str, pd.DataFrame],
        estop_dfs: Dict[str, pd.DataFrame],
        hue_order: List[str],
        alpha: float = PLOT_DEFAULT_KWARGS["alpha"],
        xlim_hist: Tuple = (0, 2.5),
        ylim_sample: Tuple[float] = (0.005, 25),  # log-scale
        n_samples: int = 100,
        no_title: bool = False,
        plots_dir: str = "./plots",
        plots_ext: str = ".pdf",
        xlabel: str = "Time",  # for fig. 2
        ylabel: str = "E-Process (log-scale)",  # for fig. 2
        vertical: bool = True,
) -> Tuple[sns.FacetGrid, sns.FacetGrid]:
    """Plot a histogram of stopped e-values.

    Input is a pair of list of data frames obtained by calling
    `ecombine.diagnostics.compute_e_and_stopping()`.
    """
    os.makedirs(plots_dir, exist_ok=True)

    # define a consistent palette across plots
    palette = [PLOT_DEFAULT_COLORS[hue_order.index(method)] for method in dfs]
    linestyles = ["solid", "dashed", "dotted", "dashdot"]

    # 1. Histogram
    combined_edf = pd.concat(
        [estop_df for estop_df in estop_dfs.values()],
        ignore_index=True,
    )
    fg_hist = sns.displot(
        x="e_stopped",
        kind="hist",
        hue="E-process",
        palette=palette,
        binwidth=0.25,
        alpha=alpha,
        data=combined_edf,
        height=5,      # 6,
        aspect=1.6,    # 1.5,
        facet_kws=dict(legend_out=False)
    )
    for i, (method, estop_df) in enumerate(estop_dfs.items()):
        fg_hist.ax.axvline(
            x=estop_df.e_stopped.mean(),
            c=palette[i],
            linestyle=linestyles[i],
            linewidth=3,
        )
    fg_hist.ax.axvline(x=1, c="gray", linestyle="dotted", linewidth=2)
    fg_hist.ax.set(
        xlabel="Stopped e-value",
        xlim=xlim_hist,
    )
    if not no_title:
        fg_hist.ax.set_title(
            "Histogram of stopped e-values at " 
            r"$\tau^{\mathbb{F}}$"
        )
    fg_hist.tight_layout()
    fg_hist.savefig(os.path.join(plots_dir, "stopped_e_histogram" + plots_ext), 
                    dpi=350, bbox_inches="tight")

    # 2. Sample e-processes and stopping times
    combined_df = pd.concat(
        [df for df in dfs.values()],
        ignore_index=True,
    )
    vertical_args = (
        {"row": "E-process", "height": 3, "aspect": 2} 
        if vertical 
        else {"col": "E-process", "height": 4, "aspect": 1.5}
    )
    fg_e = sns.relplot(
        x="Time",
        y="e",
        units="id",
        estimator=None,
        hue="E-process",
        style="E-process",
        kind="line",
        linewidth=PLOT_DEFAULT_KWARGS["linewidth"],
        palette=palette,
        alpha=0.5,
        data=combined_df.loc[combined_df.id < n_samples].loc[~combined_df.stopped],
        legend=False,
        **vertical_args,
    )
    # sns.move_legend(fg_e, loc="upper right", bbox_to_anchor=(1, 1))
    # highlight stopped values
    for i, (method, estop_df) in enumerate(estop_dfs.items()):
        ax = fg_e.axes.flatten()[i]
        sns.scatterplot(
            x="tau",
            y="e_stopped",
            hue="E-process",
            palette=[palette[i]],
            data=estop_df.loc[estop_df.id < n_samples],
            ax=ax,
            legend=False,
        )
        ax.set_title(f"{method}")
        ax.axhline(y=1, color="gray", linestyle="dotted")
        ax.set(
            ylim=ylim_sample,
            yscale="log",
            xlabel=xlabel,
            ylabel=ylabel,
        )
    fg_e.tight_layout()
    fg_e.savefig(os.path.join(plots_dir, "stopped_e_samples" + plots_ext), 
                 dpi=350, bbox_inches="tight")

    return fg_hist, fg_e
