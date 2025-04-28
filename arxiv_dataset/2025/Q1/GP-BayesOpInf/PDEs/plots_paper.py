# plots_paper.py
"""Plots used in the paper for the Euler numerical example."""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import matplotlib.patches as mplpatches

import config_euler as euler
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
DIRLABEL = "euler"
if not os.path.isdir(DATADIR := os.path.join("data", DIRLABEL)):
    os.mkdir(DATADIR)
if not os.path.isdir(FIGDIR := os.path.join("figures", DIRLABEL)):
    os.mkdir(FIGDIR)


# Utilities ===================================================================
def _savefig(fig: plt.Figure, filename: str):
    fig.savefig(
        (figfile := os.path.join(FIGDIR, f"euler-{filename}")),
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
    drawsandspread: bool = False,
):
    """Apply common formatting."""
    axes = np.atleast_2d(axes)

    # Tick sizes
    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=TICK)

    if axes.shape[0] == 3 and axes[1, 0].get_ylabel() == "Pressure":
        for ax in axes[1, :]:
            try:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            except AttributeError as ex:
                if (
                    ex.args[0]
                    != "This method only works with the ScalarFormatter"
                ):
                    raise
            ax.yaxis.get_offset_text().set_fontsize("xx-small")
            ax.locator_params(axis="y", nbins=2)

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
        if drawsandspread:
            patch = mplpatches.Patch(
                facecolor=axes[0, 1].collections[0].get_facecolor(),
                alpha=0.4,
                linewidth=0,
            )
            lines = [axes[0, 0].lines[0], axes[0, 1].lines[0], patch]

        leg = ax.legend(
            lines,
            labels,
            ncol=len(labels),
            loc="lower center",
            fontsize=14.5 if drawsandspread else LEGEND,
            bbox_to_anchor=(0.5, 0),
            bbox_transform=fig.transFigure,
        )
        for line in leg.get_lines():
            line.set_linewidth(2.25)
            line.set_markersize(14)
            line.set_alpha(1)

    # Training regime
    if t_end is not None:
        for ax in axes.flat:
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

        for ax in axes[0, :]:
            t0 = ax.get_xlim()[0]
            ymax = ax.get_ylim()[1]
            ax.text(
                t0 + 0.001,
                ymax,
                "train",
                ha="left",
                va="top",
                **TRAINPREDICT,
            )
            ax.text(
                t_end + 0.001,
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
def plot_fomdata(prefix: str = "ex2a", filename: str = "fomdata.pdf"):
    """Plot full-order data: initial conditions, truth, and observations."""
    splot = step4.StatePlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-full.h5")
    )
    model = euler.FullOrderModel()

    # Set up subplot grid.
    fig = plt.figure(constrained_layout=True, figsize=(12, 5))
    spec = fig.add_gridspec(
        nrows=3,
        ncols=2,
        hspace=0.05,
        wspace=0.025,
        width_ratios=[0.5, 1.5],
        height_ratios=[1, 1, 1],
        bottom=0.15,
    )
    ax00 = fig.add_subplot(spec[0, 0])
    ax01 = fig.add_subplot(spec[0, 1:])
    ax10 = fig.add_subplot(spec[1, 0])
    ax11 = fig.add_subplot(spec[1, 1:])
    ax20 = fig.add_subplot(spec[2, 0])
    ax21 = fig.add_subplot(spec[2, 1:])
    axes = np.array([[ax00, ax01], [ax10, ax11], [ax20, ax21]])

    # Plot initial conditions.
    v0, p0, xi0 = model.split(euler.initial_conditions)
    x = euler.spatial_domain
    x -= x[0]
    dx = x[1] - x[0]
    L = x[-1] + dx
    nodes = np.array([0, L / 3, 2 * L / 3, L])
    knots = list(euler.init_params)
    lines = dict(linewidth=1, color="#a92c00")
    dots = dict(
        linestyle="none",
        color="black",
        marker="^",
        markersize=5,
        markeredgewidth=0,
    )
    axes[0, 0].plot(x, v0, **lines)
    axes[0, 0].plot(nodes, knots[3:] + [knots[3]], **dots)
    axes[1, 0].plot(x, p0, **lines)
    # axes[1, 0].plot(nodes, [1e5] * 4, **dots)
    axes[2, 0].plot(x, xi0, **lines)
    axes[2, 0].plot(nodes, 1 / np.array(knots[:3] + [knots[0]]), **dots)

    # Format initial condition plots.
    for ax in axes[:, 0]:
        ax.set_xlim(0, L)
        ax.set_xticks(nodes, [])
    axes[0, 0].set_title("Initial conditions", fontsize=LABEL)
    axes[0, 0].set_ylabel("Velocity", fontsize=LABEL)
    axes[1, 0].set_ylabel("Pressure", fontsize=LABEL)
    axes[2, 0].set_ylabel("$1/$Density", fontsize=LABEL)
    axes[-1, 0].set_xticks(
        nodes,
        [r"$0$", r"$2/3$", r"$4/3$", r"$2$"],
        fontsize=TICK,
    )
    axes[-1, 0].set_xlabel(r"$x$", fontsize=LABEL)

    # Plot full-order data.
    end = splot.end_train_index
    t = splot.prediction_time_domain[:end]
    nlocs = splot.numspatialpoints
    colors = plt.cm.tab10(np.linspace(0, 1, nlocs + 1)[:-1])
    lines["linewidth"] = 0.75
    lines["linestyle"] = "--"
    v, p, xi = model.split(splot.true_states)
    for i in range(nlocs):
        lines["color"] = colors[i]
        axes[0, 1].plot(t, v[i, :end], **lines)
        axes[1, 1].plot(t, p[i, :end], **lines)
        axes[2, 1].plot(t, xi[i, :end], **lines)

    # Plot (sparse / noisy) full-order snapshots.
    v_data, p_data, xi_data = model.split(splot.snapshots)
    t_data = splot.sampling_time_domain
    dots["marker"] = "*"
    dots["markersize"] = 6
    for i in range(nlocs):
        dots["color"] = colors[i]
        axes[0, 1].plot(t_data, v_data[i, :], **dots)
        axes[1, 1].plot(t_data, p_data[i, :], **dots)
        axes[2, 1].plot(t_data, xi_data[i, :], **dots)

    # Colorbar.
    scale = mplcolors.Normalize(vmin=0, vmax=1)
    lscmap = mplcolors.LinearSegmentedColormap.from_list(
        name="euler",
        colors=colors,
        N=nlocs,
    )
    mappable = plt.cm.ScalarMappable(norm=scale, cmap=lscmap)
    cbar = fig.colorbar(mappable, ax=axes[:, 1:], pad=0.015)
    cbar.set_ticks(np.linspace(0, 1, 2 * nlocs + 1)[1::2])
    cbar.set_ticklabels([f"{xx:.1f}" for xx in splot.spatial_domain])
    cbar.set_label(r"Spatial coordinate", fontsize=LABEL)  # $x_i$")

    # Format full-order data plots.
    ticks = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    for ax in axes[:, 1]:
        ax.set_xlim(t[0], t[-1] + t[2])
        ax.set_xticks(ticks, [])
        ax.set_yticklabels([])
    axes[0, 1].set_title("Snapshot data", fontsize=LABEL)
    # axes[0, 1].set_ylabel(r"$v(x_i, t)$", fontsize=LABEL)
    # axes[1, 1].set_ylabel(r"$p(x_i, t)$", fontsize=LABEL)
    # axes[2, 1].set_ylabel(r"$\xi(x_i, t)$", fontsize=LABEL)
    axes[-1, 1].set_xticks(
        ticks,
        [f"{s:.2f}" for s in ticks],
        fontsize=TICK,
    )
    axes[-1, 1].set_xlabel(TIMELABEL, fontsize=LABEL)
    for i in range(axes.shape[0]):
        axes[i, 0].set_ylim(axes[i, 1].get_ylim())

    _format(fig, axes)

    return _savefig(fig, filename)


def plot_gpfit(prefix: str = "ex1a", filename: str = "gpfit.pdf"):
    """Plot the Gaussian process fit to compressed data."""
    rplot = step4.ReducedPlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-reduced.h5")
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 5),
        sharex=True,
        # sharey=True,
    )

    t = rplot.training_time_domain
    end = rplot.end_train_index
    for i, ax in enumerate(axes.flat):
        # Plot compressed noiseless data.
        rplot._plot_truth(
            ax,
            rplot.prediction_time_domain[:end],
            rplot.true_states_compressed[i, :end],
        )
        # Plot compressed observations.
        rplot._plot_data(
            ax,
            rplot.sampling_time_domain,
            rplot.snapshots_compressed[i, :],
        )
        # Plot Gaussian process fit to the data.
        rplot._plot_gp(
            ax,
            t,
            rplot.gp_means[i],
            rplot.gp_stds[i],
            width=3,
            isGP=True,
        )
        ax.set_ylabel(rf"$\hat{{q}}_{i + 1:d}(t)$", fontsize=LABEL)
        ax.set_xlim(t[0], t[-1] + t[6])

    # Format axes.
    ticks = np.arange(t[0], t[-1] + 0.005, 0.01)[::2]
    for j in range(axes.shape[1]):
        axes[-1, j].set_xlabel(TIMELABEL, fontsize=LABEL)
        axes[-1, j].set_xticks(
            ticks,
            [f"{tt:.2f}" for tt in ticks],
            fontsize=TICK,
        )
    fig.subplots_adjust(wspace=0.3, hspace=0.15, bottom=0.225)

    _format(
        fig,
        axes,
        ["Truth", "Observations", r"Gaussian process $\mu \pm 3\sigma$"],
        spread=True,
    )

    return _savefig(fig, filename)


def plot_dimension(prefix: str = "ex1r8", filename: str = "dims.pdf"):
    rplot = step4.ReducedPlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-reduced.h5")
    )
    svdvals = np.load(os.path.join(DATADIR, f"{prefix}-svdvals.npy"))

    # Set up subplot grid.
    plt.GridSpec
    fig = plt.figure(figsize=(12, 4.5))
    spec = fig.add_gridspec(
        nrows=3,
        hspace=0.15,
        height_ratios=[1, 1, 1],
        ncols=2,
        wspace=0.3,
        width_ratios=[1, 1],
        bottom=0.275,
    )
    axbig = fig.add_subplot(spec[:, 0])
    ax1 = fig.add_subplot(spec[0, 1])
    ax2 = fig.add_subplot(spec[1, 1])
    ax3 = fig.add_subplot(spec[2, 1])
    axes = np.array([[ax1], [ax2], [ax3]])

    # Plot singular value decay.
    axbig.semilogy(
        np.arange(svdvals.size) + 1,
        svdvals / svdvals[0],
        "o-",
        color="tab:blue",
        linewidth=0.5,
        markersize=5,
        markeredgewidth=0,
    )
    axbig.set_xlabel("Singular value index", fontsize=LABEL)
    axbig.set_ylabel("Normalized singular value", fontsize=LABEL)
    axbig.set_xlim(0, 20.5)
    axbig.set_ylim(5e-2, 1.2)

    # Plot Gaussian process fits.
    t = rplot.training_time_domain
    end = rplot.end_train_index
    for ii, ax in enumerate(axes[:, 0]):
        i = ii + 4
        # Plot compressed noiseless data.
        rplot._plot_truth(
            ax,
            rplot.prediction_time_domain[:end],
            rplot.true_states_compressed[i, :end],
        )
        # Plot compressed observations.
        rplot._plot_data(
            ax,
            rplot.sampling_time_domain,
            rplot.snapshots_compressed[i, :],
        )
        # Plot Gaussian process fit to the data.
        rplot._plot_gp(
            ax,
            t,
            rplot.gp_means[i],
            rplot.gp_stds[i],
            width=3,
            isGP=True,
        )
        ax.set_ylabel(rf"$\hat{{q}}_{i + 1:d}(t)$", fontsize=LABEL)
        ax.set_xlim(t[0], t[-1] + t[6])

    # Format axes.
    ticks = np.arange(t[0], t[-1] + 0.005, 0.01)[::2]
    ax3.set_xlabel(TIMELABEL, fontsize=LABEL)
    ax3.set_xticks(
        ticks,
        [f"{tt:.2f}" for tt in ticks],
        fontsize=TICK,
    )
    for ax in ax1, ax2:
        ax.set_xticks(ticks, ["" for tt in ticks])
    # fig.subplots_adjust(wspace=0.3, hspace=0.15, bottom=0.225)

    _format(
        fig,
        axes,
        ["Truth", "Observations", r"Gaussian process $\mu \pm 3\sigma$"],
        spread=True,
    )

    return _savefig(fig, filename)


def plot_derivatives(
    prefixes: list,
    nmodes: int = 4,
    noiselevels: list = None,
    nstds: int = 3,
    filename: str = "derivatives.pdf",
):
    ncol = len(prefixes)
    fig, axes = plt.subplots(
        nmodes,
        ncol,
        figsize=(6 * ncol, 5 * nmodes / 3),
        sharex=True,
        # sharey="row",
    )

    for j, prefix in enumerate(prefixes):
        rplot = step4.ReducedPlotter.load(
            os.path.join(DATADIR, f"{prefix}_data-reduced.h5")
        )

        # Load data.
        with h5py.File(
            os.path.join(DATADIR, f"{prefix}-ddtdata.h5"), "r"
        ) as hf:
            time_domain_FD = hf["time_domain_FD"][:]
            ddts_finitedifferences = hf["ddts_finitedifferences"][:]
            time_domain_GP = hf["time_domain_GP"][:]
            ddts_GPmean = hf["ddts_GPmean"][:]
            ddts_GPstd = hf["ddts_GPstd"][:]
            time_domain_truth = hf["time_domain_truth"][:]
            ddts_truth = hf["ddts_truth"][:]

        for ii, ax in enumerate(axes[:, j]):
            i = 2 * ii
            rplot._plot_truth(
                ax,
                time_domain_truth,
                ddts_truth[i],
                linewidth=1.5,
            )
            rplot._plot_gp(
                ax,
                time_domain_GP,
                ddts_GPmean[i],
                ddts_GPstd[i],
                width=nstds,
                isGP=True,
                linewidth=0.5,
            )
            rplot._plot_data(
                ax,
                time_domain_FD,
                ddts_finitedifferences[i],
                markersize=3,
                marker="p",
                zorder=0.0001,
            )
            ax.set_xlim(
                time_domain_truth[0],
                time_domain_truth[-1] + time_domain_GP[6],
            )
            if j == 0:
                ax.set_ylabel(
                    rf"$\dot{{\hat{{q}}}}_{i + 1:d}(t)$", fontsize=LABEL
                )
        if noiselevels is not None:
            axes[0, j].set_title(
                rf"${time_domain_FD.size}$ snapshots, "
                rf"${noiselevels[j]}\%$ noise"
            )
    if noiselevels is None:
        axes[0, 0].set_title("Closeup")
        axes[0, 1].set_title("Full view")

    # Format axes.
    t = time_domain_GP
    ticks = np.arange(t[0], t[-1] + 0.005, 0.01)[::2]
    for ax in axes[-1, :]:
        ax.set_xlabel(TIMELABEL, fontsize=LABEL)
        ax.set_xticks(ticks, [f"{tt:.2f}" for tt in ticks], fontsize=TICK)
    fig.subplots_adjust(wspace=0.3, hspace=0.15, bottom=0.225)

    for ax in axes[0, :1].flat:
        ax.set_ylim(-375, 375)
    for ax in axes[1, :1].flat:
        ax.set_ylim(-275, 275)
    for ax in axes[2, :1].flat:
        ax.set_ylim(-210, 210)

    _format(
        fig,
        axes,
        [
            "True derivatives",
            rf"Gaussian process $\mu \pm {nstds}\sigma$",
            "Finite difference estimates",
        ],
        spread=True,
    )

    return _savefig(fig, filename)


def plot_gpfit_and_derivatives(
    prefix: str = "ex2a",
    nmodes: int = 3,
    nstds: int = 3,
    filename: str = "gpfit-and-ddts.pdf",
):
    rplot = step4.ReducedPlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-reduced.h5")
    )

    fig, axes = plt.subplots(
        nmodes,
        2,
        figsize=(12, 5 * nmodes / 3),
        sharex=True,
    )

    # Plot Gaussian process fit to the states.
    t = rplot.training_time_domain
    end = rplot.end_train_index
    for ii, ax in enumerate(axes[:, 0]):
        i = 2 * ii
        # Plot compressed noiseless data.
        rplot._plot_truth(
            ax,
            rplot.prediction_time_domain[:end],
            rplot.true_states_compressed[i, :end],
            linewidth=1.5,
        )
        # Plot compressed observations.
        rplot._plot_data(
            ax,
            rplot.sampling_time_domain,
            rplot.snapshots_compressed[i, :],
        )
        # Plot Gaussian process fit to the data.
        rplot._plot_gp(
            ax,
            t,
            rplot.gp_means[i],
            rplot.gp_stds[i],
            width=nstds,
            isGP=True,
            linewidth=0.5,
        )
        ax.set_ylabel(rf"$\hat{{q}}_{i + 1:d}(t)$", fontsize=LABEL)
        ax.set_xlim(t[0], t[-1] + t[6])

    # Plot Gaussian process fit to the derivatives.
    with h5py.File(os.path.join(DATADIR, f"{prefix}-ddtdata.h5"), "r") as hf:
        time_domain_FD = hf["time_domain_FD"][:]
        ddts_finitedifferences = hf["ddts_finitedifferences"][:]
        time_domain_GP = hf["time_domain_GP"][:]
        ddts_GPmean = hf["ddts_GPmean"][:]
        ddts_GPstd = hf["ddts_GPstd"][:]
        time_domain_truth = hf["time_domain_truth"][:]
        ddts_truth = hf["ddts_truth"][:]

    for ii, ax in enumerate(axes[:, 1]):
        i = 2 * ii
        rplot._plot_truth(
            ax,
            time_domain_truth,
            ddts_truth[i],
            linewidth=1.5,
        )
        rplot._plot_gp(
            ax,
            time_domain_GP,
            ddts_GPmean[i],
            ddts_GPstd[i],
            width=nstds,
            isGP=True,
            linewidth=0.5,
        )
        rplot._plot_data(
            ax,
            time_domain_FD,
            ddts_finitedifferences[i],
            markersize=3,
            marker="p",
            zorder=0.0001,
        )
        ax.set_xlim(
            time_domain_truth[0],
            time_domain_truth[-1] + time_domain_GP[6],
        )
        ax.set_ylabel(rf"$\dot{{\hat{{q}}}}_{i + 1:d}(t)$", fontsize=LABEL)

    # Format axes.
    ticks = np.arange(t[0], t[-1] + 0.005, 0.01)[::2]
    for ax in axes[-1, :]:
        ax.set_xlabel(TIMELABEL, fontsize=LABEL)
        ax.set_xticks(
            ticks,
            [f"{tt:.2f}" for tt in ticks],
            fontsize=TICK,
        )
    fig.subplots_adjust(wspace=0.3, hspace=0.15, bottom=0.225)
    _format(fig, axes)
    axes[0, 0].set_title("Reduced states", fontsize=LABEL, y=0.95)
    axes[0, 1].set_title(
        "Reduced state time derivatives", fontsize=LABEL, y=0.95
    )

    # Legend
    labels = [
        "Truth",
        "Observations",
        rf"Gaussian process $\mu \pm {nstds}\sigma$",
        "Finite difference estimates",
    ]
    lines = axes[0, 0].lines[:2] + axes[0, 1].lines[-2:]
    lines[-2] = mplpatches.Patch(
        facecolor=ax.collections[0].get_facecolor(),
        alpha=0.4,
        linewidth=0,
    )
    leg = ax.legend(
        lines,
        labels,
        ncol=4,
        loc="lower center",
        fontsize=LEGEND,
        bbox_to_anchor=(0.5, 0),
        bbox_transform=fig.transFigure,
    )
    for line in leg.get_lines():
        line.set_linewidth(2.25)
        line.set_markersize(14)
        line.set_alpha(1)

    return _savefig(fig, filename)


def plot_draws_and_IQR(prefix: str, filename: str):
    """Plot reduced-order model solutions as draws and as an IQR."""
    rplot = step4.ReducedPlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-reduced.h5")
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 3.5),
        sharex=True,
        sharey=True,
    )

    t = rplot.prediction_time_domain[::DS]
    t_end = rplot.training_time_domain[-1]

    draws = [draw[0][::DS] for draw in rplot.draws_compressed[:50]]
    rplot._plot_draws(axes[0], t, draws, alpha=0.4)
    rplot._plot_percentiles(axes[1], t, draws)
    rplot._plot_samplemean(axes[1], t, draws)

    # Format axes.
    axes[0].set_ylabel(r"$\hat{q}_1(t)$", fontsize=LABEL)
    ticks = np.arange(t[0], t[-1] + 0.005, 0.03)
    for ax in axes:
        ax.set_xlim(t[0], t[-1] + t[6])
        ax.set_xlabel(TIMELABEL, fontsize=LABEL)
        ax.set_xticks(
            ticks,
            [f"{tt:.2f}" if tt else "0" for tt in ticks],
            fontsize=TICK,
        )
        ax.set_ylim(-1.4, 1.6)

    fig.subplots_adjust(wspace=0.05, hspace=0, bottom=0.3)

    _format(
        fig,
        axes,
        [
            # "Truth",
            # "Observations",
            "Sampled ROM predictions",
            "Mean of ROM predictions",
            r"$95\%$ IQR of ROM predictions",
        ],
        spread=False,
        drawsandspread=True,
        t_end=t_end,
    )

    return _savefig(fig, filename)


def plot_romsolution(prefix: str, filename: str, spread: bool = True):
    """Plot the reduced-order model solution"""
    rplot = step4.ReducedPlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-reduced.h5")
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(12, 5),
        sharex=True,
        # sharey=True,
    )

    t = rplot.prediction_time_domain[::DS]
    t_end = rplot.training_time_domain[-1]
    for i, ax in enumerate(axes.flat):
        # Plot compressed noiseless data.
        rplot._plot_truth(
            ax,
            t,
            rplot.true_states_compressed[i, ::DS],
        )
        # Plot compressed observations.
        rplot._plot_data(
            ax,
            rplot.sampling_time_domain,
            rplot.snapshots_compressed[i, :],
        )
        # Plot ROM predictions.
        draws = [draw[i][::DS] for draw in rplot.draws_compressed[:NUMDRAWS]]
        rplot._plot_samplemean(ax, t, draws)
        if spread:
            rplot._plot_percentiles(ax, t, draws)
        else:
            rplot._plot_draws(ax, t, draws)

        ax.set_ylabel(rf"$\hat{{q}}_{i + 1:d}(t)$", fontsize=LABEL)
        ax.set_xlim(t[0], t[-1] + t[6])

    # Format axes.
    ticks = np.arange(t[0], t[-1] + 0.005, 0.03)
    for ax in axes[-1, :]:
        ax.set_xlabel(TIMELABEL, fontsize=LABEL)
        ax.set_xticks(
            ticks,
            [f"{tt:.2f}" if tt else "0" for tt in ticks],
            fontsize=TICK,
        )
    axes[0, 0].set_ylim(-1.1, 1.6)
    axes[0, 1].set_ylim(-1.25, 1.7)
    axes[1, 0].set_ylim(-0.75, 0.85)
    axes[1, 1].set_ylim(-0.7, 0.6)
    axes[2, 0].set_ylim(-0.75, 0.75)
    axes[2, 1].set_ylim(-0.7, 1)

    fig.subplots_adjust(wspace=0.3, hspace=0.15, bottom=0.225)

    _format(
        fig,
        axes,
        ["Truth", "Observations", _drawslabel(spread), "Sample mean"],
        spread=spread,
        t_end=t_end,
    )

    return _savefig(fig, filename)


def plot_fomsolution(prefix: str, filename: str, spread: bool = True):
    """Plot full-order model solutions of each of the three variables at
    four points in space.
    """
    splot = step4.StatePlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-full.h5")
    )

    nlocs = splot.numspatialpoints
    fig, axes = plt.subplots(
        nrows=3,
        ncols=nlocs,
        figsize=(12, 5),
        sharex=True,
        sharey="row",
    )

    t = splot.prediction_time_domain[::DS]
    t_end = splot.training_time_domain[-1]
    variables_true = np.split(splot.true_states, 3, axis=0)
    variables_data = np.split(splot.snapshots, 3, axis=0)
    for i in range(3):
        for j in range(nlocs):
            ax = axes[i, j]

            # Plot compressed noiseless data.
            splot._plot_truth(
                ax,
                t,
                variables_true[i][j, ::DS],
            )
            # Plot compressed observations.
            splot._plot_data(
                ax,
                splot.sampling_time_domain,
                variables_data[i][j, :],
            )
            # Plot ROM predictions.
            draws = [
                np.split(draw, 3)[i][j, ::DS]
                for draw in splot.draws[:NUMDRAWS]
            ]
            splot._plot_samplemean(ax, t, draws)
            if spread:
                splot._plot_percentiles(ax, t, draws)
            else:
                splot._plot_draws(ax, t, draws)
            ax.set_xlim(t[0], t[-1] + t[6])

    # Format axes.
    axes[0, 0].set_ylabel("Velocity", fontsize=LABEL)
    axes[1, 0].set_ylabel("Pressure", fontsize=LABEL)
    axes[2, 0].set_ylabel("$1/$Density", fontsize=LABEL)
    ticks = np.arange(t[0], t[-1] + 1e-3, 0.04)
    for ax in axes[-1, :]:
        ax.set_xlabel(TIMELABEL, fontsize=LABEL)
        ax.set_xticks(
            ticks,
            [f"{tt:.2f}" if tt else "0" for tt in ticks],
            fontsize=TICK,
        )
    for i, ax in enumerate(axes[0]):
        ax.set_title(f"$x = {splot.spatial_domain[i]:.1f}$", fontsize=LABEL)

    for ax in axes[0, :]:
        ax.set_ylim(91.5, 111)
    for ax in axes[1, :]:
        ax.set_ylim(0.875e5, 1.15e5)
    # for ax in axes[2, :]:
    #     ax.set_ylim(.036, .056)

    fig.subplots_adjust(wspace=0.05, hspace=0.2, bottom=0.225)
    _format(
        fig,
        axes,
        ["Truth", "Observations", _drawslabel(spread), "Sample mean"],
        spread=spread,
        t_end=t_end,
    )

    return _savefig(fig, filename)


def plot_fomsolution_closeup(prefix: str, filename: str, spread: bool = False):
    """Plot full-order model solutions of the first two variables at one
    point in space.
    """
    splot = step4.StatePlotter.load(
        os.path.join(DATADIR, f"{prefix}_data-full.h5")
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 5),
        sharex=True,
        sharey=False,
    )
    fig.subplots_adjust(wspace=0.275, bottom=0.225, hspace=0.15)

    t = splot.prediction_time_domain[::DS]
    t_end = splot.training_time_domain[-1]
    variables_true = np.split(splot.true_states, 3, axis=0)
    variables_proj = np.split(splot.true_states_projected, 3, axis=0)
    # variables_data = np.split(splot.snapshots, 3, axis=0)

    for i, ax in enumerate(axes):
        # Plot noiseless data.
        splot._plot_truth(
            ax,
            t,
            variables_true[i][0, ::DS],
        )
        # Plot projected noiseless data.
        splot._plot_truth_projected(
            ax,
            t,
            variables_proj[i][0, ::DS],
        )
        # Plot ROM prediction sample mean.
        draws = [
            np.split(draw, 3)[i][0, ::DS] for draw in splot.draws[:NUMDRAWS]
        ]
        splot._plot_samplemean(ax, t, draws)
        ax.set_xlim(t[0], t[-1] + t[6])
        ax.locator_params(axis="y", nbins=2)

    # Format axes.
    axes[0].set_ylabel("Velocity", fontsize=LABEL)
    axes[1].set_ylabel("Pressure", fontsize=LABEL)
    ticks = np.arange(t[0], t[-1] + 1e-3, 0.04)
    axes[-1].set_xticks(
        ticks,
        [f"{tt:.2f}" if tt else "0" for tt in ticks],
        fontsize=TICK,
    )
    axes[-1].set_xlabel(TIMELABEL, fontsize=LABEL)
    axes[0].set_title(r"$x = 0$", fontsize=LABEL)

    axes[0].set_ylim(94, 106)
    axes[1].set_ylim(0.91e5, 1.12e5)
    axes[1].set_yticks(
        [1e5, 1.1e5],
        [r"$1.0$e$5$", r"$1.1$e$5$"],
        fontsize=TICK,
    )
    # fig.align_ylabels(axes)

    _format(
        fig,
        np.array([axes]).T,
        [
            "Truth",
            "Truth projected",
            "Mean of sampled ROM predictions",
        ],
        spread=False,
        t_end=t_end,
    )

    return _savefig(fig, filename)


def plot_comparison_reduced(
    which: str,
    prefixes: list,
    filename: str,
    spread: bool = True,
):
    """Plot solutions at a single point with several m'."""
    ntests = len(prefixes)
    rplots = [
        step4.ReducedPlotter.load(
            os.path.join(DATADIR, f"{prefix}_data-reduced.h5")
        )
        for prefix in prefixes
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=ntests,
        figsize=(12, 5),
        sharex=True,
        sharey="row",
    )

    t = rplots[0].prediction_time_domain[::DS]
    t_end = rplots[0].training_time_domain[-1]
    for j, rplot in enumerate(rplots):
        for i in range(3):
            ax = axes[i, j]

            # Plot compressed noiseless data.
            rplot._plot_truth(
                ax,
                t,
                rplot.true_states_compressed[i, ::DS],
            )
            # Plot compressed observations.
            rplot._plot_data(
                ax,
                rplot.sampling_time_domain,
                rplot.snapshots_compressed[i, :],
            )
            # Plot ROM predictions.
            draws = [
                draw[i][::DS] for draw in rplot.draws_compressed[:NUMDRAWS]
            ]
            rplot._plot_samplemean(ax, t, draws)
            if spread:
                rplot._plot_percentiles(ax, t, draws)
            else:
                rplot._plot_draws(ax, t, draws)
            ax.set_xlim(t[0], t[-1] + t[6])

    for i, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(rf"$\hat{{q}}_{i + 1:d}(t)$", fontsize=LABEL)
    for j, ax in enumerate(axes[0]):
        if which == "estimates":
            mprime = rplots[j].training_time_domain.size
            ax.set_title(f"$m' = {mprime}$", fontsize=LABEL)
        elif which == "sparsity":
            m = rplots[j].sampling_time_domain.size
            ax.set_title(f"$m = {m}$", fontsize=LABEL)
        elif which == "noise":
            xi = (1, 3, 5)[j]
            ax.set_title(rf"$\xi = {xi}\%$", fontsize=LABEL)
    ticks = [0, 0.05, 0.1]
    for ax in axes.flat:
        ax.set_xticks(ticks)
    for ax in axes[-1, :]:
        ax.set_xlabel(TIMELABEL, fontsize=LABEL)
        ax.set_xticks(ticks, [f"{tt:.2f}" for tt in ticks], fontsize=TICK)

    for ax in axes[0, :]:
        ax.set_ylim(-1.1, 1.9)
    for ax in axes[1, :]:
        ax.set_ylim(-1.25, 1.7)
    for ax in axes[2, :]:
        ax.set_ylim(-0.75, 0.85)

    fig.subplots_adjust(wspace=0.05, hspace=0.2, bottom=0.225)
    _format(
        fig,
        axes,
        ["Truth", "Observations", _drawslabel(spread), "Sample mean"],
        spread=spread,
        t_end=t_end,
    )

    _savefig(fig, filename)


def plot_comparison_full(
    which: str,
    prefixes: list,
    filename: str,
    locindex: int = 0,
    spread: bool = True,
):
    """Plot solutions at a single point with several m'."""
    ntests = len(prefixes)
    splots = [
        step4.StatePlotter.load(
            os.path.join(DATADIR, f"{prefix}_data-full.h5")
        )
        for prefix in prefixes
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=ntests,
        figsize=(12, 5),
        sharex=True,
        sharey="row",
    )

    t = splots[0].prediction_time_domain[::DS]
    t_end = splots[0].training_time_domain[-1]
    for j, splot in enumerate(splots):
        variables_true = [
            var[locindex, ::DS]
            for var in np.split(splot.true_states, 3, axis=0)
        ]
        variables_data = [
            var[locindex, :] for var in np.split(splot.snapshots, 3, axis=0)
        ]
        for i in range(3):
            ax = axes[i, j]

            # Plot compressed noiseless data.
            splot._plot_truth(
                ax,
                t,
                variables_true[i],
            )
            # Plot compressed observations.
            splot._plot_data(
                ax,
                splot.sampling_time_domain,
                variables_data[i],
            )
            # Plot ROM predictions.
            draws = [
                np.split(draw, 3)[i][locindex, ::DS]
                for draw in splot.draws[:NUMDRAWS]
            ]
            splot._plot_samplemean(ax, t, draws)
            if spread:
                splot._plot_percentiles(ax, t, draws)
            else:
                splot._plot_draws(ax, t, draws)
            ax.set_xlim(t[0], t[-1] + t[6])

    axes[0, 0].set_ylabel("Velocity", fontsize=LABEL)
    axes[1, 0].set_ylabel("Pressure", fontsize=LABEL)
    axes[2, 0].set_ylabel("$1/$Density", fontsize=LABEL)
    for j, ax in enumerate(axes[0]):
        if which == "estimates":
            mprime = splots[j].training_time_domain.size
            ax.set_title(f"$m' = {mprime}$", fontsize=LABEL)
        elif which == "sparsity":
            m = splots[j].sampling_time_domain.size
            ax.set_title(f"$m = {m}$", fontsize=LABEL)
        elif which == "noise":
            xi = (1, 3, 5)[j]
            ax.set_title(rf"${xi}\%$ noise", fontsize=LABEL)
    ticks = [0, 0.05, 0.1]
    for ax in axes.flat:
        ax.set_xticks(ticks)
    for ax in axes[-1, :]:
        ax.set_xlabel(TIMELABEL, fontsize=LABEL)
        ax.set_xticks(ticks, [f"{tt:.2f}" for tt in ticks], fontsize=TICK)
    for ax in axes[0, :]:
        ax.set_ylim(91.5, 111)
    for ax in axes[1, :]:
        ax.set_ylim(0.875e5, 1.15e5)
    # for ax in axes[2, :]:
    #     ax.set_ylim(.036, .056)

    fig.subplots_adjust(wspace=0.05, hspace=0.2, bottom=0.225)
    _format(
        fig,
        axes,
        ["Truth", "Observations", _drawslabel(spread), "Sample mean"],
        spread=spread,
        t_end=t_end,
    )

    _savefig(fig, filename)


def all_plots():
    plot_fomdata("ex2a", f"fomdata.{EXT}")
    plot_gpfit("ex1a", f"gpfit.{EXT}")
    plot_derivatives(["ex2a", "ex1a"], 3, [1, 3], f"derivatives.{EXT}")
    plot_dimension("ex1r8", f"dims.{EXT}")
    plot_draws_and_IQR("ex1a", f"noisy-reduced-draws.{EXT}")
    for spread in True, False:
        end = "-spread" if spread else ""
        plot_romsolution("ex1a", f"ex1a-reduced{end}.{EXT}", spread=spread)
        plot_fomsolution("ex1a", f"ex1a-full{end}.{EXT}", spread=spread)
        plot_romsolution("ex2a", f"ex2a-reduced{end}.{EXT}", spread=spread)
        plot_fomsolution("ex2a", f"ex2a-full{end}.{EXT}", spread=spread)
        plot_comparison_reduced(
            "estimates",
            ["ex1b", "ex1a", "ex1c"],
            f"ex1-mcomparison-reduced{end}.{EXT}",
            spread=spread,
        )
        plot_comparison_full(
            "estimates",
            ["ex1b", "ex1a", "ex1c"],
            f"ex1-mcomparison{end}.{EXT}",
            locindex=2,
            spread=spread,
        )
        plot_comparison_reduced(
            "noise",
            ["ex1d", "ex1a", "ex1e"],
            f"ex1-noisecomparison{end}.{EXT}",
            locindex=2,
            spread=spread,
        )
        plot_comparison_full(
            "noise",
            ["ex1d", "ex1a", "ex1e"],
            f"ex1-noisecomparison{end}.{EXT}",
            locindex=2,
            spread=spread,
        )
        plot_comparison_reduced(
            "estimates",
            ["ex2b", "ex2a", "ex2c"],
            f"ex2-mcomparison-reduced{end}.{EXT}",
            spread=spread,
        )
        plot_comparison_full(
            "estimates",
            ["ex2b", "ex2a", "ex2c"],
            f"ex2-mcomparison{end}.{EXT}",
            locindex=2,
            spread=spread,
        )
        plot_comparison_reduced(
            "sparsity",
            ["ex2d", "ex2a", "ex2e"],
            f"ex2-noisecomparison{end}.{EXT}",
            locindex=2,
            spread=spread,
        )
        plot_comparison_full(
            "sparsity",
            ["ex2d", "ex2a", "ex2e"],
            f"ex2-noisecomparison{end}.{EXT}",
            locindex=2,
            spread=spread,
        )


def paper():
    plot_fomdata("ex2a", f"sparse-fomdata.{EXT}")
    plot_dimension("ex1r8", f"noisy-dims.{EXT}")
    plot_derivatives(["ex2a", "ex1a"], 3, [1, 3], f"derivatives.{EXT}")
    plot_draws_and_IQR("ex1a", f"noisy-reduced-draws.{EXT}")
    plot_romsolution("ex1a", f"noisy-reduced.{EXT}", spread=True)
    plot_fomsolution("ex1a", f"noisy-full.{EXT}", spread=True)
    plot_fomsolution_closeup("ex1a", f"noisy-closeup.{EXT}", spread=True)
    plot_comparison_full(
        "noise",
        ["ex1d", "ex1a", "ex1e"],
        f"noisy-comparison.{EXT}",
        spread=True,
    )
    plot_comparison_reduced(
        "sparsity",
        ["ex2d", "ex2a", "ex2e"],
        f"sparse-comparison.{EXT}",
        spread=True,
    )
    plot_romsolution("ex2a", f"sparse-reduced.{EXT}", spread=True)
    plot_fomsolution("ex2a", f"sparse-full.{EXT}", spread=True)


# =============================================================================
if __name__ == "__main__":
    paper()
