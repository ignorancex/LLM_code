# utils.py
"""Utilities for logging, timing, loading, and saving."""
import os
import platform
import subprocess
import matplotlib.pyplot as plt

import opinf

import config


# Logging / timing tools ======================================================
def _array2string(arr):
    if arr.ndim > 1:
        return (
            "[" + "\n ".join([_array2string(subarr) for subarr in arr]) + "]"
        )

    return "[ " + ", ".join([f"{x:.4e}" for x in arr]) + " ]"


def summarize_experiment(
    training_span: tuple[float, float],
    num_samples: int,
    noiselevel: float,
    num_regression_points: int,
    numPODmodes: int,
    gp_regularizer: float = None,
    ndraws: int = None,
):
    """Summarize the experimental setup."""
    num_trajectories = len(config.input_parameters)
    low, high = training_span
    predict = config.time_domain
    low2, high2 = predict[0], predict[-1]

    report = [
        "\nEXPERIMENTAL SCENARIO",
        f"Data:"
        f"\n\t* {num_trajectories:d} trajectories"
        f"\n\t* {num_samples:d} randomly sampled snapshots in each trajectory"
        f"\n\t* {noiselevel:.2%} noise"
        f"\n\t* training   {low:.2f} ≤ t < {high:.2f}"
        f"\n\t* prediction {low2:.2f} ≤ t ≤ {high2:.2f}",
        f"Dimension: retaining {numPODmodes} POD modes",
        f"Training: using {num_regression_points:d} regression points",
    ]
    if gp_regularizer is not None:
        report.append(f"GP regularization: eta = {gp_regularizer:.2e}")
    if ndraws is not None:
        report.append(f"Posterior: {ndraws} draws")
    report = "\n".join(report)

    with open(os.path.join(config.figures_path(), "report.txt"), "w") as out:
        out.write(report)
    return print("\n" + report + "\n")


# Figure saving ===============================================================
def _open_file(file_path: str):
    if os.path.isfile(file_path):
        if platform.system() == "Darwin":  # MacOS
            subprocess.call(("open", file_path))
        elif platform.system() == "Windows":  # Windows
            os.startfile(file_path)
        else:  # Linux
            subprocess.call(("xdg-open", file_path))


def save_figure(figname: str, andopen: bool = False, fig=None):
    """Save the current matplotlib figure to the figures folder."""
    if fig is None:
        fig = plt.gcf()
    save_path = os.path.join(config.figures_path(), figname)

    with opinf.utils.TimedBlock(f"Saving {save_path}"):
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.001, dpi=250)
        plt.close(fig)

    if andopen:
        _open_file(save_path)
