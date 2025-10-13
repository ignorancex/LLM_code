#!/usr/bin/env python3
"""
Interactive scatter plot of M500 values with hover annotations.

Loads a NumPy `.npy` file of M500 data versus snapshot index,
plots it on a log-scaled y-axis, and displays point indices on hover.

Usage:
    python interactive_m500_plot.py --data-path path/to/m500.npy
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.widgets import Button, Slider  # imported for future use

# ---------------------------------------------------------------------------- #
# Logging
# ---------------------------------------------------------------------------- #


def setup_logger(level: int = logging.INFO) -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------- #
# Plotting
# ---------------------------------------------------------------------------- #


def load_data(data_path: Path) -> np.ndarray:
    """Load M500 data from a .npy file."""
    return np.load(data_path)


def create_scatter(ax: plt.Axes, data: np.ndarray) -> PathCollection:
    """Create a scatter plot on given axes with log-scaled y-axis."""
    x = np.arange(data.size)
    scatter = ax.scatter(x, data)
    ax.set_yscale("log")
    ax.set_xlabel("Snapshot Index")
    ax.set_ylabel("M500")
    ax.set_title("M500 vs Snapshot Index")
    return scatter


def init_annotation(ax: plt.Axes) -> plt.Annotation:
    """Initialize an invisible annotation box for hover info."""
    annot = ax.annotate(
        text="",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)
    return annot


def update_annotation(
    annot: plt.Annotation, ind: dict, scatter: PathCollection
) -> None:
    """Update position and text of annotation based on hover index."""
    # Get index of first hovered point
    idx = ind["ind"][0]
    x, y = scatter.get_offsets()[idx]
    annot.xy = (x, y)
    text = f"Index: {idx}\nValue: {y:.3E}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.8)


def connect_hover(
    fig: plt.Figure, ax: plt.Axes, scatter: PathCollection, annot: plt.Annotation
) -> None:
    """Connect hover event to show annotations."""

    def on_hover(event):
        if event.inaxes != ax:
            return
        contains, ind = scatter.contains(event)
        if contains:
            update_annotation(annot, ind, scatter)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #


def main(data_path: Path, verbose: bool) -> None:
    """Load data, set up plot, and start interactive session."""
    setup_logger(logging.DEBUG if verbose else logging.INFO)
    logging.info("Loading data from %s", data_path)

    data = load_data(data_path)

    fig, ax = plt.subplots()
    scatter = create_scatter(ax, data)
    annot = init_annotation(ax)
    connect_hover(fig, ax, scatter, annot)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive M500 scatter plot with hover annotations."
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=Path,
        default=Path(__file__).parent / "analysis" / "m500.npy",
        help="Path to the M500 .npy file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug-level logging"
    )
    args = parser.parse_args()
    main(data_path=args.data_path, verbose=args.verbose)
