# utils.py
"""Utilities for timing code, logging actions, and exporting figures."""
import os
import platform
import subprocess
import matplotlib.pyplot as plt

import opinf

import config


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
    gp_regularizer: float = None,
    opinf_regularizer: float = None,
    ndraws: int = None,
):
    """Summarize the experimental setup."""
    report = [
        "EXPERIMENTAL SCENARIO",
        f"Data: {num_samples:d} uniformly sampled snapshots "
        f"over {training_span[0]:.2f} ≤ t < {training_span[1]:.2f} "
        f"with {noiselevel:.2%} noise",
        f"Training: using {num_regression_points:d} regression points",
    ]
    if gp_regularizer is not None:
        opreg = "lambda TBD via optimization"
        if opinf_regularizer is not None:
            opreg = f"lambda={opinf_regularizer:.2e}"
        report.append(f"Regularization: eta = {gp_regularizer:.2e}, {opreg}")
    if ndraws is not None:
        report.append(f"Posterior: {ndraws} draws")
    report = "\n".join(report)

    with open(os.path.join(config.figures_path(), "report.txt"), "w") as out:
        out.write(report)
    return print("\n" + report)


def summarize_posterior(parameters, bayesian_model) -> None:
    """Summarize the experimental results."""
    report = "\n".join(
        [
            "\nPOSTERIOR DISTRIBUTION",
            f"True parameters:\t{_array2string(parameters)}",
            f"Posterior mean:\t\t{_array2string(bayesian_model.mean)}",
            f"Posterior covariance:\n{_array2string(bayesian_model.cov)}",
        ]
    )

    with open(os.path.join(config.figures_path(), "report.txt"), "w") as out:
        out.write(report)
    print("\n" + report)


# Figure saving ===============================================================
def _open_file(file_path):
    if os.path.isfile(file_path):
        if platform.system() == "Darwin":  # MacOS
            subprocess.call(("open", file_path))
        elif platform.system() == "Windows":  # Windows
            os.startfile(file_path)
        else:  # Linux
            subprocess.call(("xdg-open", file_path))


def save_figure(figname, andopen=False, fig=None):
    """Save the current matplotlib figure to the figures folder."""
    if fig is None:
        fig = plt.gcf()
    save_path = os.path.join(config.figures_path(), figname)

    with opinf.utils.TimedBlock(f"Saving {save_path}"):
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.001, dpi=250)
        plt.close(fig)

    if andopen:
        _open_file(save_path)
