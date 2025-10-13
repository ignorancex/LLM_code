import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.axes_grid1 import ImageGrid
import wandb




def plot_bitmaps(config, data_dir, plot_dir):
    NUM_CLASSES = 10
    # expert_grid = [int(i) for i in config["n_exps"].split(" ")]
    WIDTH = config["n_exp_per_l"]
    HEIGHT = config["n_layers"]

    # load stuff
    magic_idxs = np.load(data_dir / "magic_idxs.npy")
    bitmaps = np.load(data_dir / "bitmaps.npy")
    labels = np.load(data_dir / "labels.npy")
    preds = np.load(data_dir / "all_preds.npy")

    # Bitmap-like plot
    NUM_CLASSES = 10
    expert_grid = [config["n_exp_per_l"]] * config["n_layers"]
    WIDTH = expert_grid[0]
    HEIGHT = len(expert_grid)

    # Automatically infer a reasonable grid shape for the subplot layout
    def compute_grid_dims(n):
        """Finds a grid (rows, cols) for n subplots as square-like as possible."""
        for i in range(int(n**0.5), 0, -1):
            if n % i == 0:
                return (i, n // i)
        return (1, n)

    grid_rows, grid_cols = compute_grid_dims(NUM_CLASSES)

    # Dynamically size the figure (tweak scale_factor as needed)
    scale_factor = 2.5
    fig_width = scale_factor * grid_cols
    fig_height = scale_factor * grid_rows +1
    fig = plt.figure(figsize=(fig_width, fig_height))

    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 5),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="4%",
        cbar_pad=0.1,
    )
    x_coords, y_coords = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    for ax, label in zip(grid, range(NUM_CLASSES)):
        class_idxs = labels == label
        act_idxs = magic_idxs[class_idxs]
        bitmap_at_act = bitmaps[class_idxs][np.arange(len(act_idxs)), act_idxs]
        avg_bitmap = np.mean(bitmap_at_act, axis=(0,))
        flat_bitmap = avg_bitmap.flatten()
        scatter = ax.scatter(
            x_flat,
            y_flat,
            s=100,  # Fixed circle size
            c=flat_bitmap,  # Color by usage
            cmap="viridis",
            alpha=1,
            edgecolors="white",
            linewidth=0.5,
            vmin=0.0,
            vmax=1.0,
        )
        for x, y, val in zip(x_flat, y_flat, flat_bitmap):
            # Choose text color based on background brightness
            normalized_val = (val - flat_bitmap.min()) / (
                flat_bitmap.max() - flat_bitmap.min()
            )
            text_color = "white" if normalized_val < 0.5 else "black"
            ax.text(
                x,
                y,
                f"{round(val, 2):0<3}"[1:].ljust(3, "0"),
                ha="center",
                va="center",
                fontsize=5,
                color=text_color,
            )
        avg_num_used = bitmap_at_act.sum((-1, -2)).mean()
        class_preds = preds[class_idxs][np.arange(len(act_idxs)), act_idxs]
        accuracy = (class_preds.argmax(-1) == label).mean()
        ax.text(
            0.5,
            1.2,
            f"Class: {label} $({avg_num_used:.1f}, {100*accuracy:2.0f}\%)$",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="center",
        )
        ax.set_xlim(-0.5, WIDTH - 0.5)
        ax.set_ylim(-0.5, HEIGHT - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_axis_off()

    ax.cax.colorbar(scatter, label="Usage fraction", aspect=20)
    plt.savefig(plot_dir / "bitmap.pdf", bbox_inches="tight")
    plt.close()


    fig = plt.figure(figsize=(fig_width, fig_height))

    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 5),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="4%",
        cbar_pad=0.1,
    )
    x_coords, y_coords = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    for ax, label in zip(grid, range(NUM_CLASSES)):
        class_idxs = labels == label
        act_idxs = magic_idxs[class_idxs]
        bitmap_at_act = bitmaps[class_idxs][np.arange(len(act_idxs)), act_idxs]
        n_act_layer = np.bitwise_or.reduce(bitmap_at_act, axis=-1).sum(0)
        avg_bitmap_layernorm = np.sum(bitmap_at_act, axis=0) / (n_act_layer[:, None] + 1)
        flat_bitmap = avg_bitmap_layernorm.flatten()
        scatter = ax.scatter(
            x_flat,
            y_flat,
            s=100,  # Fixed circle size
            c=flat_bitmap,  # Color by usage
            cmap="viridis",
            alpha=1,
            edgecolors="white",
            linewidth=0.5,
            vmin=0.0,
            vmax=1.0,
        )
        for x, y, val in zip(x_flat, y_flat, flat_bitmap):
            # Choose text color based on background brightness
            normalized_val = (val - flat_bitmap.min()) / (
                flat_bitmap.max() - flat_bitmap.min()
            )
            text_color = "white" if normalized_val < 0.5 else "black"
            ax.text(
                x,
                y,
                f"{round(val, 2):0<3}"[1:].ljust(3, "0"),
                ha="center",
                va="center",
                fontsize=5,
                color=text_color,
            )
        ax.text(
            0.5,
            1.2,
            f"Class: {label}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="center",
        )
        ax.set_xlim(-0.5, WIDTH - 0.5)
        ax.set_ylim(-0.5, HEIGHT - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_axis_off()

    ax.cax.colorbar(scatter, label="Usage fraction", aspect=20)
    plt.savefig(plot_dir / "bitmap_layernorm.pdf", bbox_inches="tight")
    plt.close()

def main():
    # load stylesheet
    plt.style.use("myplots.mplstyle")
    # load files
    parser = argparse.ArgumentParser(description="Plot expert bitmap")
    parser.add_argument(
        "--id",
        type=str,
        help="Wandb ID of the run.",
        required=True
    )
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
    plot_bitmaps(run_config, data_dir, plot_dir)


if __name__ == "__main__":
    main()
