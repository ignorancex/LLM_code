import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
import yaml
from matplotlib import patheffects
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(description="Plot expert bitmap")
    parser.add_argument("--id", type=str, help="Wandb ID of the run.", required=True)
    # load style-sheet
    plt.style.use("myplots.mplstyle")
    # get run config
    args = parser.parse_args()
    api = wandb.Api()
    run = api.run(f"silvretta/PaperSweeps/{args.id}")
    run_config = run.config
    run_name = run.name
    sweep_name = run.sweep.name
    data_dir = Path("../results") / sweep_name / run_name / "data"
    plot_dir = Path("../results") / sweep_name / run_name / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    args = parser.parse_args()
    plot_curves(run_config, data_dir, plot_dir)


def plot_curves(config, data_dir, plot_dir):
    NUM_CLASSES = 10
    WIDTH = config["n_exp_per_l"]
    HEIGHT = config["n_layers"]
    NUM_EXP = WIDTH * HEIGHT
    # load stuff
    preds = np.load(data_dir / "all_preds.npy")
    acts_seqs = np.load(data_dir / "acts_seqs.npy")
    firing_rates = np.load(data_dir / "experts_frs.npy")
    magic_idxs = np.load(data_dir / "magic_idxs.npy")
    bitmaps = np.load(data_dir / "bitmaps.npy")
    labels = np.load(data_dir / "labels.npy")

    all_accuracy_curves = {}
    mask = {}
    at_least = 100
    for numexp in range(NUM_EXP):
        samples_with_numexp = magic_idxs == numexp
        mask[numexp] = samples_with_numexp.sum() >= at_least
        if np.any(samples_with_numexp):
            samples = preds[samples_with_numexp][:, :numexp]
            pred_labels = np.argmax(samples, axis=-1)
            true_labels = labels[samples_with_numexp]
            is_correct_through_time = pred_labels == true_labels[:, None]
            accuracies = np.mean(is_correct_through_time, axis=0)
            all_accuracy_curves[numexp] = np.stack(
                (np.arange(1, numexp + 1), accuracies), axis=-1
            )

    # mean array
    color_coding = np.array([k for k in all_accuracy_curves])
    acc_curves = list(all_accuracy_curves.values())

    # Create 3 subplots: left (wide), center (thin), right (wide)
    fig = plt.figure(figsize=(7, 3))
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 0.5, 3], hspace=0.05, wspace=0.2)
    ax_left = fig.add_subplot(gs[0, 0])  # Line plot
    ax_center = fig.add_subplot(gs[0, 1])  # Colorbar
    ax_right = fig.add_subplot(gs[0, 2])  # Horizontal bar plot

    # Setup discrete colormap
    color_coding = np.array([k for k in all_accuracy_curves])
    n_colors = len(color_coding)
    discrete_cmap = plt.get_cmap("viridis_r", n_colors)
    boundaries = np.arange(n_colors + 1) - 0.5
    discrete_norm = BoundaryNorm(boundaries, discrete_cmap.N)

    # LEFT SUBPLOT: LineCollection
    for idx, curve in reversed(all_accuracy_curves.items()):
        if mask[idx]:
            color = discrete_cmap(idx)
            ax_left.plot(
                *curve.T,
                color=color,
                lw=1,
                markersize=5,
                marker=f"$\sf{{{idx}}}$",
                path_effects=[patheffects.withStroke(linewidth=2, foreground="w")],
            )

    ax_left.set_xlabel("Num. experts", fontsize=10)
    ax_left.set_ylabel("Accuracy")
    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_left.grid(True, alpha=0.3)
    props = dict(boxstyle="round", facecolor="white", edgecolor="grey")
    ypos = sum(ax_left.get_yticks()[1:3]) / 2  # should be around the first ytick
    xpos = ax_left.get_xticks()[-2]  # should be somewhere on the right
    ax_left.text(
        xpos,
        ypos,
        f"Cutoff: counts $\geq{{{at_least}}}$",
        ha="right",
        va="center",
        transform=ax_left.transData,
        bbox=props,
    )

    # CENTER SUBPLOT: Legend-style display
    # Remove all spines and make it look like a legend
    ax_center.spines["top"].set_visible(False)
    ax_center.spines["right"].set_visible(False)
    ax_center.spines["bottom"].set_visible(False)
    ax_center.spines["left"].set_visible(False)
    ax_center.set_xticks([])
    ax_center.set_yticks([])

    # Use the SAME coordinate system as the right plot
    # Set the same y-limits as the right plot BEFORE plotting
    ax_center.set_ylim(ax_right.get_ylim())  # This ensures same coordinate system
    ax_center.invert_yaxis()  # Match the right plot's inverted y-axis

    # Create legend-like entries using the same data coordinates as bars
    for i in range(n_colors):
        color = discrete_cmap(discrete_norm(i))
        line_x = [0.0, 0.4]
        line_y = [i, i]
        ax_center.plot(line_x, line_y, color=color, linewidth=3, solid_capstyle="round")
        # Add class label
        ax_center.text(
            0.6,
            i,
            f"{i + 1}",
            verticalalignment="center_baseline",
            fontsize=7,
            fontweight="normal",
            color="black",
        )

    # Set x limits and styling
    ax_center.text(
        0.5,
        -0.05,
        "Num. used\nat activation",
        fontsize=7,
        ha="center",
        transform=ax_center.transAxes,
    )

    # RIGHT SUBPLOT: Horizontal bar plot
    usages, counts = np.unique(magic_idxs, return_counts=True)
    colors = [discrete_cmap(discrete_norm(i)) for i in range(n_colors)]
    bars = ax_right.barh(
        range(n_colors),
        counts,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        height=0.8,
    )

    ax_right.set_xlabel("Counts")
    ax_right.set_yticks([])
    ax_right.invert_yaxis()

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax_right.text(
            width + 2.0,
            i,
            f"({val})",
            ha="left",
            va="center",
            fontsize=6,
            fontweight="bold",
        )

    # Ensure all y-axes are aligned
    ax_center.set_ylim(ax_right.get_ylim())
    plt.suptitle("Accuracy vs. number of active experts", y=0.95)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(plot_dir / "accuracy_curves.pdf")
    print(f"plots saved in  {plot_dir}")


if __name__ == "__main__":
    main()
