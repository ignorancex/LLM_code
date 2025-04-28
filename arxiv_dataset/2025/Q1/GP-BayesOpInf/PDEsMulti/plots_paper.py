# plots_paper.py
"""Plots used in the paper for the cubic heat equation numerical example."""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.patches as mplpatches

import config_heat as heat
import step4_plot as step4


# Font sizes.
LABEL = "medium"
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
DIRLABEL = "heat3"
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
    t_end: float = None,
):
    """Apply common formatting."""

    # Tick sizes
    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=TICK)

    # Label alignment
    for j in range(axes.shape[1]):
        fig.align_ylabels(axes[:, j])

    # Legend
    if labels is not None:
        ax = axes[-1, -1]
        lines = ax.lines[: len(labels) - 1] + [ax.lines[-1]]
        if spread:
            lines[-2] = mplpatches.Patch(
                facecolor=ax.collections[0].get_facecolor(),
                alpha=0.4,
                linewidth=0,
            )
        leg = ax.legend(
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

    if t_end is not None:
        for ax in axes.flat:
            # ax.axvline(t_end, lw=.5, color=TRAINPREDICT["color"])
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

        for ax in axes[0, :]:
            t0 = ax.get_xlim()[0]
            ymax = ax.get_ylim()[1]
            ax.text(
                t0 + 0.025,
                ymax,
                "train",
                ha="left",
                va="top",
                **TRAINPREDICT,
            )
            ax.text(
                t_end + 0.025,
                ymax,
                "predict",
                ha="left",
                va="top",
                **TRAINPREDICT,
            )

    return


def _drawslabel(spread: bool):
    if spread:
        return r"$95\%$ IQR of ROM predictions"
    return "Sampled ROM predictions"


# Main routines ===============================================================
def plot_fomdata(filename: str = "fomdata.png"):
    """Plot full-order data: initial conditions, truth, and observations."""
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 5))

    XX, TT = np.meshgrid(heat.spatial_domain, heat.time_domain, indexing="ij")

    for ax, params in zip(axes.flat, heat.input_parameters):
        model = heat.FullOrderModel(params)
        Q = model.solve(heat.initial_conditions, heat.time_domain)
        ax.pcolormesh(
            XX,
            TT,
            Q,
            shading="nearest",
            cmap="magma",
            vmin=0,
            vmax=1,
        )
        ax.set_title(rf"$(a,b) = {params}$", fontsize=LABEL)

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$t$", fontsize=LABEL)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x$", fontsize=LABEL)

    fig.subplots_adjust(wspace=0.05, hspace=0.25)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=mplcolors.Normalize(), cmap="magma"),
        ax=axes,
        extend="both",
    )

    return _savefig(fig, filename)


def plot_samples(prefix: str = "ex3", filename: str = "samples.pdf"):
    splot = step4.StatePlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-full.h5")
    )
    params = heat.input_parameters

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    # Parameter space and samples.
    axes[0].plot(
        [p[0] for p in params],
        [p[1] for p in params],
        "s",
        color=step4.baseplots.COLORS["data"],
        markeredgewidth=0,
        markersize=5,
        label="Training parameter values",
    )
    axes[0].plot(
        [1.5],
        [0.5],
        "d",
        color=step4.baseplots.COLORS["GPcs"],
        label="Test parameter value",
    )
    axes[0].annotate(
        "test parameter",
        xy=(1.425, 0.425),
        xytext=(-1, 0),
        arrowprops=dict(
            arrowstyle="-",
            color=step4.baseplots.COLORS["GPcs"],
            linewidth=0.5,
        ),
        fontsize="x-small",
        color=step4.baseplots.COLORS["GPcs"],
    )

    ticks = [-2, -1, 0, 1, 2]
    axes[0].set_xlim(-2.2, 2.2)
    axes[0].set_ylim(-2.2, 2.2)
    axes[0].set_aspect("equal")
    axes[0].set_xticks(ticks)
    axes[0].set_yticks(ticks)
    axes[0].set_xlabel(r"$a$", fontsize=LABEL)
    axes[0].set_ylabel(r"$b$", fontsize=LABEL)
    axes[0].set_title("Training parameter values", fontsize=LABEL)

    # One noisy snapshot in space.
    xx = heat.spatial_domain
    t = splot.sampling_time_domain[1][10]
    truth = heat.FullOrderModel(heat.input_parameters[1]).solve(
        heat.initial_conditions, heat.time_domain
    )
    truth = truth[:, np.argmin(np.abs(heat.time_domain - t))]
    noisy = np.load(os.path.join(DATADIR, "onesnap_noisy.npy"))
    splot._plot_truth(axes[1], xx, truth, color="#a92c00")
    splot._plot_data(axes[1], xx, noisy, markersize=3)

    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xticks(ticks)
    axes[1].set_yticks(ticks)
    axes[1].set_xlabel(r"$x$", fontsize=LABEL)
    axes[1].set_ylabel(r"$q(x, t_j)$", fontsize=LABEL)
    axes[1].set_title(rf"Example snapshot, $t_j={t:.2f}$", fontsize=LABEL)

    leg = axes[1].legend(ncol=1, loc="upper left", fontsize=LEGEND)
    for line in leg.get_lines():
        line.set_linewidth(2)
        line.set_markersize(14)
        line.set_alpha(1)

    return _savefig(fig, filename)


def plot_gpfit(
    trajectories: list,
    prefix: str = "ex1a",
    filename: str = "gpfit.pdf",
):
    """Plot the Gaussian process fit to compressed data."""
    rplot = step4.ReducedPlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-reduced.h5")
    )

    fig, axes = plt.subplots(
        nrows=rplot.num_modes,
        ncols=len(trajectories),
        figsize=(12, 7.5),
        sharex=True,
        sharey="row",
    )

    t = rplot.training_time_domain
    end = rplot.end_train_index
    for col, ell in enumerate(trajectories):
        for i, ax in enumerate(axes[:, col].flat):
            # Plot compressed noiseless data.
            rplot._plot_truth(
                ax,
                rplot.prediction_time_domain[:end],
                rplot.true_states_compressed[ell][i, :end],
            )
            # Plot compressed observations.
            rplot._plot_data(
                ax,
                rplot.sampling_time_domain[ell],
                rplot.snapshots_compressed[ell][i, :],
            )
            # Plot Gaussian process fit to the data.
            rplot._plot_gp(
                ax,
                t,
                rplot.gp_means[ell][i],
                rplot.gp_stds[ell][i],
                width=3,
            )
            ax.set_xlim(t[0], t[-1] + t[2])

    for j, ax in enumerate(axes[0, :]):
        ax.set_title(
            rf"$(a, b) = {heat.input_parameters[trajectories[j]]}$",
            fontsize=LABEL,
        )
    for i, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(rf"$\hat{{q}}_{i + 1:d}(t)$", fontsize="small")

    # Format axes.
    ticks = np.arange(t[0], t[-1], 0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel(TIMELABEL, fontsize=LABEL)
        ax.set_xticks(ticks, [f"{tt:.1f}" for tt in ticks], fontsize=TICK)
    fig.subplots_adjust(wspace=0.05, hspace=0.25, bottom=0.15)

    _format(
        fig,
        axes,
        ["Truth", "Observations", r"Gaussian process $\mu \pm 3\sigma$"],
        spread=True,
    )

    return _savefig(fig, filename)


def plot_romsolution(
    trajectories: list,
    prefix: str,
    filename: str,
    spread: bool = False,
):
    """Plot the reduced-order model solution"""
    rplot = step4.ReducedPlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-reduced.h5")
    )

    fig, axes = plt.subplots(
        nrows=rplot.num_modes,
        ncols=len(trajectories),
        figsize=(12, 6),
        sharex=True,
        sharey="row",
    )
    fig.subplots_adjust(wspace=0.05, hspace=0.15, bottom=0.18)

    t = rplot.prediction_time_domain[::DS]
    for col, ell in enumerate(trajectories):
        for i, ax in enumerate(axes[:, col].flat):
            # Plot compressed noiseless data.
            rplot._plot_truth(
                ax,
                t,
                rplot.true_states_compressed[ell][i, ::DS],
            )
            # Plot compressed observations.
            rplot._plot_data(
                ax,
                rplot.sampling_time_domain[ell],
                rplot.snapshots_compressed[ell][i, :],
            )
            # Plot ROM predictions.
            draws = [
                draw[i, ::DS]
                for draw in rplot.draws_compressed[ell][:NUMDRAWS]
            ]
            rplot._plot_samplemean(ax, t, draws)
            if spread:
                rplot._plot_percentiles(ax, t, draws)
            else:
                rplot._plot_draws(ax, t, draws)

            ax.set_xlim(t[0], t[-1] + t[6])

    # Format axes.
    ticks = np.arange(t[0], t[-1], 0.6)
    for ax in axes[-1, :]:
        ax.set_xlabel(TIMELABEL, fontsize=LABEL)
        ax.set_xticks(
            ticks,
            [f"{tt:.1f}" if tt else "0" for tt in ticks],
            fontsize=TICK,
        )
    for j, ax in enumerate(axes[0, :]):
        ax.set_ylim(-7.5, 8)
        ax.set_title(
            rf"$(a,b) = {heat.input_parameters[trajectories[j]]}$",
            fontsize=LABEL,
        )
    for i, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(rf"$\hat{{q}}_{i + 1:d}(t)$", fontsize="small")

    _format(
        fig,
        axes,
        ["Truth", "Observations", _drawslabel(spread), "Sample mean"],
        spread=spread,
        t_end=rplot.training_time_domain[-1],
    )

    return _savefig(fig, filename)


def plot_fomsolution(
    trajectories: list,
    prefix: str,
    filename: str,
    spread: bool = False,
):
    """Plot full-order model solutions of each of the three variables at
    four points in space.
    """
    splot = step4.StatePlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-full.h5")
    )

    fig, axes = plt.subplots(
        nrows=splot.numspatialpoints - 2,
        ncols=len(trajectories),
        figsize=(12, 5),
        sharex=True,
        sharey="row",
    )
    fig.subplots_adjust(wspace=0.05, hspace=0.2, bottom=0.225)

    t = splot.prediction_time_domain[::DS]
    for col, ell in enumerate(trajectories):
        for ii, ax in enumerate(axes[:, col].flat):
            i = ii + 1
            # Plot compressed noiseless data.
            splot._plot_truth(
                ax,
                t,
                splot.true_states[ell][i, ::DS],
            )
            # # Plot projected noiseless data.
            # splot._plot_truth_projected(
            #     ax,
            #     t,
            #     splot.true_states_projected[ell][i, ::DS],
            # )
            # Plot compressed observations.
            splot._plot_data(
                ax,
                splot.sampling_time_domain[ell],
                splot.snapshots[ell][i, :],
            )
            # Plot ROM predictions.
            draws = [draw[i, ::DS] for draw in splot.draws[ell][:NUMDRAWS]]
            splot._plot_samplemean(ax, t, draws)
            if spread:
                splot._plot_percentiles(ax, t, draws)
            else:
                splot._plot_draws(ax, t, draws)

            ax.set_xlim(t[0], t[-1] + t[6])

    # Format axes.
    ticks = np.arange(t[0], t[-1], 0.6)
    for ax in axes[-1, :]:
        ax.set_xlabel(TIMELABEL, fontsize=LABEL)
        ax.set_xticks(
            ticks,
            [f"{tt:.1f}" if tt else "0" for tt in ticks],
            fontsize=TICK,
        )
    for j, ax in enumerate(axes[0, :]):
        ax.set_title(
            rf"$(a, b) = {heat.input_parameters[trajectories[j]]}$",
            fontsize=LABEL,
        )
        ax.set_ylim(ax.get_ylim()[0], 0.8)
    for i, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(
            f"$x = {splot.spatial_domain[i+1]:.2f}$",
            fontsize="small",
        )

    _format(
        fig,
        axes,
        ["Truth", "Observations", _drawslabel(spread), "Sample mean"],
        spread=spread,
        t_end=splot.training_time_domain[-1],
    )

    return _savefig(fig, filename)


def plot_newtrajectory(
    prefix: str,
    filename: str,
    spread: bool = False,
):
    """Plot full-order model solutions of each of the three variables at
    four points in space.
    """
    splot = step4.StatePlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-full.h5")
    )
    with h5py.File(
        os.path.join(DATADIR, f"{prefix}_newtrajectory.h5"), "r"
    ) as hf:
        truth_reduced = hf["truth_reduced"][:]
        truth_full = hf["truth_full"][:]
        draws_reduced = hf["draws_reduced"][:]
        draws_full = hf["draws_full"][:]

    fig, axes = plt.subplots(
        nrows=splot.numspatialpoints,
        ncols=2,
        figsize=(12, 6),
        sharex=True,
        sharey=False,
    )
    fig.subplots_adjust(wspace=0.25, hspace=0.25, bottom=0.18)

    # Plot reduced-order results.
    t = splot.prediction_time_domain[::DS]
    for i, ax in enumerate(axes[:, 0]):
        # Plot compressed noiseless data.
        splot._plot_truth(
            ax,
            t,
            truth_reduced[i, ::DS],
        )
        # Plot ROM predictions.
        draws = [draw[i, ::DS] for draw in draws_reduced[:NUMDRAWS]]
        splot._plot_samplemean(ax, t, draws)
        if spread:
            splot._plot_percentiles(ax, t, draws)
        else:
            splot._plot_draws(ax, t, draws)
        ax.set_ylabel(rf"$\hat{{q}}_{i+1:d}(t)$", fontsize="small")

    # Plot full-order results.
    for i, ax in enumerate(axes[:, 1]):
        index = np.argmin(
            np.abs(splot.spatial_domain[i] - heat.spatial_domain)
        )
        # Plot compressed noiseless data.
        splot._plot_truth(
            ax,
            t,
            truth_full[index, ::DS],
        )
        # Plot ROM predictions in the full space.
        draws = [draw[index, ::DS] for draw in draws_full[:NUMDRAWS]]
        if spread:
            splot._plot_percentiles(ax, t, draws)
        else:
            splot._plot_draws(ax, t, draws)
        splot._plot_samplemean(ax, t, draws)
        ax.set_ylabel(
            rf"$x = {splot.spatial_domain[i]:.2f}$",
            fontsize="small",
        )

    # Format axes.
    ticks = np.arange(t[0], t[-1], 0.6)
    for ax in axes.flat:
        ax.set_xlim(t[0], t[-1] + t[6])
    for ax in axes[-1, :]:
        ax.set_xlabel(TIMELABEL, fontsize=LABEL)
        ax.set_xticks(
            ticks,
            [f"{tt:.1f}" if tt else "0" for tt in ticks],
            fontsize=TICK,
        )
    axes[0, 0].set_title("Reduced-order predictions", fontsize=LABEL)
    axes[0, 1].set_title("Full-order predictions", fontsize=LABEL)
    fig.suptitle(rf"$(a, b) = {heat.test_parameters}$", fontsize=LABEL, y=0.95)

    _format(
        fig,
        axes,
        ["Truth", _drawslabel(spread), "Sample mean"],
        spread=spread,
    )

    return _savefig(fig, filename)


def all_plots():
    plot_fomdata("fomdata.png")
    plot_gpfit([0, 2, 4], "ex3", f"gpfit.{EXT}")
    for spread in True, False:
        end = "-spread" if spread else ""
        plot_romsolution(
            [0, 2, 4],
            "ex3",
            f"reduced{end}.{EXT}",
            spread=spread,
        )
        plot_fomsolution(
            [1, 2, 3],
            "ex3",
            f"full{end}.{EXT}",
            spread=spread,
        )
        plot_newtrajectory(
            "ex3",
            f"newtrajectory{end}.{EXT}",
            spread=spread,
        )


def paper():
    plot_samples("ex3", f"samples.{EXT}")
    plot_romsolution(
        [0, 2, 4],
        "ex3",
        f"reduced.{EXT}",
        spread=True,
    )
    plot_fomsolution(
        [1, 2, 3],
        "ex3",
        f"full.{EXT}",
        spread=True,
    )
    plot_newtrajectory(
        "ex3",
        f"newtrajectory.{EXT}",
        spread=True,
    )


# =============================================================================
if __name__ == "__main__":
    paper()
