# plots_paper.py
"""Plots used in the paper for the SEIRD numerical example."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mplpatches

import step4_plot as step4


# Font sizes.
LABEL = "small"
TICK = "small"
LEGEND = "medium"
TRAINPREDICT = dict(color="#666666", fontsize="x-small")

# Labels.
TIMELABEL = "$t$"

# Number of posterior draws.
NUMDRAWS = 500

# Figure resolution.
DPI = 200
DS = 1
plt.rc("figure", dpi=DPI, figsize=(12, 5))
EXT = "pdf"

# Data and figure directories
DIRLABEL = "seird"
if not os.path.isdir(DATADIR := os.path.join("data", DIRLABEL)):
    os.mkdir(DATADIR)
if not os.path.isdir(FIGDIR := os.path.join("figures", DIRLABEL)):
    os.mkdir(FIGDIR)


# Utilities ===================================================================
def _savefig(fig: plt.Figure, filename: str):
    fig.savefig(
        (figfile := os.path.join(FIGDIR, f"{DIRLABEL}-{filename}")),
        bbox_inches="tight",
        pad_inches=0.03,
        dpi=DPI,
    )
    return print(f"Saved: {figfile}")


def _format(
    fig: plt.Figure,
    axes: np.ndarray,
    labels: list = None,
    spread: bool = False,
    t_ends: float = None,
):
    """Apply common formatting."""

    # Tick sizes
    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=TICK)
        # ax.locator_params(axis="y", nbins=2)

    # Label alignment
    for j in range(axes.shape[1]):
        fig.align_ylabels(axes[:, j])

    # Legend
    if labels is not None:
        ax = axes[-1, -1]
        lines = ax.lines[: len(labels) - 1] + [ax.lines[-1]]
        if spread:
            lines[2] = mplpatches.Patch(
                facecolor=ax.collections[0].get_facecolor(),
                alpha=0.4,
                linewidth=0,
            )

        leg = axes[-1, -1].legend(
            lines,
            labels,
            ncol=len(labels),
            loc="lower center",
            fontsize=LEGEND,
            bbox_to_anchor=(0.5, 0),
            bbox_transform=fig.transFigure,
        )
        for line in leg.get_lines():
            line.set_linewidth(2.25)
            line.set_markersize(14)
            line.set_alpha(1)

    # Training regime
    if t_ends is not None:
        for j, (t_end, axcols) in enumerate(zip(t_ends, axes.T)):
            for ax in axcols.flat:
                # ax.axvline(t_end, lw=0.5, color=TRAINPREDICT["color"])
                t0 = ax.get_xlim()[0]
                ymin, ymax = ax.get_ylim()
                ax.fill_between(
                    [t0, t_end],
                    [ymin, ymin],
                    [ymax, ymax],
                    color="gray",
                    alpha=0.1,
                    lw=0,
                )
                ax.set_ylim(ymin, ymax)

            ax = axcols[0]
            left = j == 0 and axes.shape[1] == 3
            t0 = ax.get_xlim()[0]
            ymax = ax.get_ylim()[1]
            ax.text(
                t0 + 1,
                ymin + 0.05,
                "train",
                ha="left",
                va="bottom",
                **TRAINPREDICT,
            )
            ax.text(
                t_end + 1,
                (ymin + 0.05) if left else ymax,
                "predict",
                ha="left",
                va="bottom" if left else "top",
                **TRAINPREDICT,
            )

    return


def _drawslabel(spread):
    if spread:
        return r"$95\%$ IQR of predictions"
    return "Sampled predictions"


# Main routines ===============================================================
def plot_gpfit(which="a", filename: str = "gpfit.pdf"):
    """Plot the Gaussian process fit to compressed data."""
    plot1 = step4.ODEPlotter.load(os.path.join(DATADIR, f"ex1{which}_data.h5"))
    plot2 = step4.ODEPlotter.load(os.path.join(DATADIR, f"ex2{which}_data.h5"))

    fig, all_axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(12, 7),
        sharex="col",
        sharey="row",
    )

    for plotter, axes in (
        (plot1, all_axes[:, 0]),
        (plot2, all_axes[:, 1]),
    ):
        t = plotter.training_time_domain
        end = plotter.end_train_index
        for i, ax in enumerate(axes.flat):
            # Plot noiseless data.
            plotter._plot_truth(
                ax,
                plotter.prediction_time_domain[:end],
                plotter.true_states[i, :end],
            )
            # Plot observations.
            plotter._plot_data(
                ax,
                plotter.sampling_time_domain[i],
                plotter.snapshots[i, :],
            )
            # Plot Gaussian process fit to the data.
            plotter._plot_gp(
                ax,
                t,
                plotter.gp_means[i],
                plotter.gp_stds[i],
                width=3,
            )
            ax.set_xlim(t[0], t[-1] + t[3])

        # Format axes.
        ticks = np.arange(t[0], t[-1] + 1, 30, dtype=int)
        axes[-1].set_xlabel(TIMELABEL, fontsize=LABEL)
        axes[-1].set_xticks(
            ticks,
            [f"{tt:d}" for tt in ticks],
            fontsize=TICK,
        )

    for i, ax in enumerate(all_axes[:, 0]):
        ax.set_ylabel(plot1.labels[i], fontsize="small")

    all_axes[0, 0].set_title(
        rf"${plot1.sampling_time_domain[0].size}$ observations, $10\%$ noise",
        fontsize=LABEL,
    )
    all_axes[0, 1].set_title(
        rf"${plot2.sampling_time_domain[0].size}$ observations, $5\%$ noise",
        fontsize=LABEL,
    )

    fig.subplots_adjust(wspace=0.05, hspace=0.2, bottom=0.175)
    _format(
        fig,
        all_axes,
        ["Truth", "Observations", r"Gaussian process $\mu \pm 3\sigma$"],
        spread=True,
    )

    return _savefig(fig, filename)


def plot_solution(
    prefixes: list,
    titles: list,
    filename: str = "solution.pdf",
    spread: bool = False,
):
    """Plot the model solutions."""
    plotters = [
        step4.ODEPlotter.load(os.path.join(DATADIR, f"{pfx}_data.h5"))
        for pfx in prefixes
    ]

    fig, all_axes = plt.subplots(
        nrows=5,
        ncols=len(prefixes),
        figsize=(12, 6),
        sharex="col",
        sharey="row",
    )
    fig.subplots_adjust(wspace=0.05, hspace=0.2, bottom=0.175)

    for plotter, axes in zip(plotters, all_axes.T):
        t = plotter.prediction_time_domain[::DS]
        for i, ax in enumerate(axes.flat):
            # Plot compressed noiseless data.
            plotter._plot_truth(
                ax,
                t,
                plotter.true_states[i, ::DS],
            )
            # Plot compressed observations.
            plotter._plot_data(
                ax,
                plotter.sampling_time_domain[i],
                plotter.snapshots[i, :],
            )
            # Plot ROM predictions.
            draws = [draw[i][::DS] for draw in plotter.draws[:NUMDRAWS]]
            if spread:
                plotter._plot_percentiles(ax, t, draws)
            else:
                plotter._plot_draws(ax, t, draws)
            plotter._plot_samplemean(ax, t, draws)

            ax.set_xlim(t[0], t[-1] + t[6])

        ticks = np.arange(t[0], t[-1] + 1, 60, dtype=int)
        axes[-1].set_xlabel(TIMELABEL, fontsize=LABEL)
        axes[-1].set_xticks(
            ticks,
            [f"{tt:d}" if tt else "0" for tt in ticks],
            fontsize=TICK,
        )

    for i, ax in enumerate(all_axes[:, 0]):
        ax.set_ylabel(plotters[0].labels[i], fontsize="small")

    for ax, title in zip(all_axes[0, :], titles):
        ax.set_title(title, fontsize=LABEL)

    _format(
        fig,
        all_axes,
        ["Truth", "Observations", _drawslabel(spread), "Sample mean"],
        spread=spread,
        t_ends=[plotter.training_time_domain[-1] for plotter in plotters],
    )

    return _savefig(fig, filename)


def plot_ICdiff(
    which="2",
    filename: str = "icdiff.pdf",
    spread: bool = False,
):
    """Plot the model solutions for different initial conditions."""
    plot1 = step4.ODEPlotter.load(os.path.join(DATADIR, f"ex{which}a_data.h5"))
    plot2 = step4.ODEPlotter.load(os.path.join(DATADIR, f"ex{which}b_data.h5"))

    fig, all_axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(12, 6),
        sharex="col",
        sharey="row",
    )

    for plotter, axes in (
        (plot1, all_axes[:, 0]),
        (plot2, all_axes[:, 1]),
    ):
        t = plotter.prediction_time_domain[::DS]
        for i, ax in enumerate(axes.flat):
            # Plot compressed noiseless data.
            plotter._plot_truth(
                ax,
                t,
                plotter.true_states[i, ::DS],
            )
            # Plot compressed observations.
            plotter._plot_data(
                ax,
                plotter.sampling_time_domain[i],
                plotter.snapshots[i, :],
            )
            # Plot ROM predictions.
            draws = [draw[i][::DS] for draw in plotter.draws[:NUMDRAWS]]
            if spread:
                plotter._plot_percentiles(ax, t, draws)
            else:
                plotter._plot_draws(ax, t, draws)
            plotter._plot_samplemean(ax, t, draws)
            ax.set_xlim(t[0], t[-1] + t[6])

        ticks = np.arange(t[0], t[-1] + 1, 40, dtype=int)
        axes[-1].set_xlabel(TIMELABEL, fontsize=LABEL)
        axes[-1].set_xticks(
            ticks,
            [f"{tt:d}" if tt else "0" for tt in ticks],
            fontsize=TICK,
        )

    for i, ax in enumerate(all_axes[:, 0]):
        ax.set_ylabel(plot1.labels[i], fontsize="small")

    all_axes[0, 0].set_title("True initial conditions", fontsize=LABEL)
    all_axes[0, 1].set_title("Estimated initial conditions", fontsize=LABEL)

    fig.subplots_adjust(wspace=0.05, hspace=0.2, bottom=0.175)

    _format(
        fig,
        all_axes,
        ["Truth", "Observations", _drawslabel(spread), "Sample mean"],
        spread=spread,
        t_ends=(
            plot1.training_time_domain[-1],
            plot2.training_time_domain[-1],
        ),
    )

    return _savefig(fig, filename)


def all_plots():
    plot_gpfit(filename=f"gpfit.{EXT}")
    for spread in True, False:
        end = "-spread" if spread else ""
        plot_solution(
            ["ex1c", "ex1a", "ex1d"],
            [
                r"$60$ days of data",
                r"$90$ days of data",
                r"$120$ days of data",
            ],
            filename=f"noisy-compare{end}.{EXT}",
            spread=spread,
        )
        plot_solution(
            ["ex2c", "ex2d", "ex2a"],
            [
                r"$60$ days of data",
                r"$90$ days of data",
                r"$120$ days of data",
            ],
            filename=f"sparse-compare{end}.{EXT}",
            spread=spread,
        )
        plot_ICdiff(which="2", filename=f"icdiff{end}.{EXT}", spread=spread)


def paper():
    plot_solution(
        ["ex1c", "ex1a", "ex1d"],
        [
            r"$60$ days of data",
            r"$90$ days of data",
            r"$120$ days of data",
        ],
        filename=f"noisy-compare.{EXT}",
        spread=True,
    )
    plot_solution(
        ["ex2c", "ex2d", "ex2a"],
        [
            r"$10$ observations over $60$ days",
            r"$10$ observations over $90$ days",
            r"$10$ observations over $120$ days",
        ],
        filename=f"sparse-compare.{EXT}",
        spread=True,
    )


# =============================================================================
if __name__ == "__main__":
    paper()
