#!/usr/bin/env python3
"""
plot_runtime.py

Compute and plot the evolution of the scale-factor (a) for gas in a cosmological
simulation based on wall-clock timings and timesteps data, with optional extrapolation.
"""
import logging
import argparse
from pathlib import Path
import datetime
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Initialize module-level logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SimulationRun:
    """
    Metadata for a simulation run.

    Attributes:
        name: Identifier for the run (e.g., parent directory name).
        timesteps_file: Path to the file containing timestep data.
    """

    name: str
    timesteps_file: Path


def find_latest_timesteps_file(directory: Path) -> Path:
    """
    Find the most recently modified file in `directory` matching 'timesteps*'.

    Returns:
        Path object of the latest file.

    Raises:
        FileNotFoundError: if no matching file is found.
    """
    files = list(directory.glob("timesteps*"))
    if not files:
        raise FileNotFoundError(f"No 'timesteps*' files in {directory}")
    latest = max(files, key=lambda p: p.stat().st_mtime)
    logger.debug(f"Latest timesteps file: {latest}")
    return latest


def count_lines(file_path: Path) -> int:
    """Return the number of lines in the given file."""
    with file_path.open("r") as f:
        for i, _ in enumerate(f, 1):
            pass
    return i


def moving_average(data: np.ndarray, window: int = 256) -> np.ndarray:
    """
    Compute a simple moving average over the last axis of `data`.

    Parameters:
        data: 1D array of values.
        window: Number of points to average over.
    """
    if window < 1:
        raise ValueError("Window size must be at least 1")
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def load_timestep_data(file_path: Path):
    """
    Load timestep log data from a text file.

    Assumes columns: step, ?, a, z, ..., wallclock_ms, ...
    Returns:
        steps: step indices
        scale_factors: array of 'a' values
        wallclock_hours: cumulative wall-clock time in hours
        time_bins: moving-average of CPU time bins
    """
    raw = np.genfromtxt(str(file_path), invalid_raise=False)
    steps = raw[:, 0].astype(int)
    scale_factors = raw[:, 2]
    wallclock_ms = raw[:, 12]
    wallclock_hours = np.cumsum(wallclock_ms) / 1000.0 / 3600.0
    cpu_bins = raw[:, 5]
    time_bins = moving_average(cpu_bins)
    return steps, scale_factors, wallclock_hours, time_bins


def plot_scale_factor_vs_time(
    run: SimulationRun,
    wallclock: np.ndarray,
    a: np.ndarray,
    extrapolate: bool,
    interp_range: int,
    x_limit: float,
    z_targets: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Plot scale-factor a vs. cumulative wall-clock time, with optional linear
    extrapolation to future a values.
    """
    fig, ax = plt.subplots()
    ax.plot(wallclock, a, lw=0.7, label="Simulation data")

    # Highlight end point
    ax.plot(wallclock[-1], a[-1], "o", ms=4)
    ax.axvline(wallclock[-1], ls=":", lw=0.8)
    ax.axhline(a[-1], ls=":", lw=0.8)

    if extrapolate and len(wallclock) > interp_range:
        dx = wallclock[-1] - wallclock[-interp_range]
        dy = a[-1] - a[-interp_range]
        slope = dy / dx
        intercept = a[-1] - slope * wallclock[-1]
        x_extrap = np.linspace(wallclock[-1], x_limit, 100)
        y_extrap = slope * x_extrap + intercept
        ax.plot(x_extrap, y_extrap, "--", lw=0.5, label="Extrapolation")

        for z_target in z_targets:
            a_target = 1.0 / (1 + z_target)
            if a_target <= a[-1]:
                continue
            wc_target = (a_target - intercept) / slope
            cpu_cost = wc_target * count_lines(run.timesteps_file)
            finish = datetime.datetime.now() + datetime.timedelta(
                hours=wc_target - wallclock[-1]
            )
            logger.info(
                f"Extrapolated to z={z_target:.2f}: {wc_target:.0f}h wall-clock, "
                f"{cpu_cost:.0f} CPU-hours, finish ~{finish:%Y-%m-%d %H:%M}"
            )

    ax.set_xlabel("Wall-clock time [hr]")
    ax.set_ylabel("Scale-factor a")
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()

    outfile = output_dir / f"runtime_evolution_{run.name.lower()}.png"
    fig.savefig(outfile, dpi=200)
    logger.info(f"Saved plot: {outfile}")


def plot_steps_vs_scale_factor(
    run: SimulationRun, steps: np.ndarray, a: np.ndarray, output_dir: Path
) -> None:
    """Plot number of steps vs. scale-factor a."""
    fig, ax = plt.subplots()
    ax.plot(steps, a)
    ax.set_xlabel("Step")
    ax.set_ylabel("Scale-factor a")
    fig.tight_layout()
    outfile = output_dir / f"steps_a_{run.name.lower()}.png"
    fig.savefig(outfile, dpi=200)
    logger.info(f"Saved plot: {outfile}")


def plot_time_bins_vs_scale_factor(
    run: SimulationRun, a: np.ndarray, time_bins: np.ndarray, output_dir: Path
) -> None:
    """Plot moving-average CPU time bin vs. scale-factor (excluding initial warm-up)."""
    fig, ax = plt.subplots()
    ax.plot(a[len(a) - len(time_bins) :], time_bins)
    ax.set_xlabel("Scale-factor a")
    ax.set_ylabel("Time-bin [CPU units]")
    fig.tight_layout()
    outfile = output_dir / f"timebins_a_{run.name.lower()}.png"
    fig.savefig(outfile, dpi=200)
    logger.info(f"Saved plot: {outfile}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot simulation runtime vs. scale-factor."
    )
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=Path(".."),
        help="Directory containing timesteps files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."), help="Directory to save plots."
    )
    parser.add_argument(
        "--no-extrapolate",
        action="store_false",
        dest="extrapolate",
        help="Disable extrapolation of future scale-factors.",
    )
    parser.add_argument(
        "--interp-range",
        type=int,
        default=1000,
        help="Number of points for linear extrapolation fit.",
    )
    parser.add_argument(
        "--x-limit",
        type=float,
        default=None,
        help="Max wall-clock time on x-axis (hr). Defaults to latest time.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        latest = find_latest_timesteps_file(args.snapshots_dir)
    except FileNotFoundError as e:
        logger.error(e)
        return

    run = SimulationRun(name=args.snapshots_dir.resolve().name, timesteps_file=latest)

    steps, a, wallclock, time_bins = load_timestep_data(run.timesteps_file)

    # Determine x-axis limit
    xlim = args.x_limit or wallclock[-1]

    # Define redshift targets for extrapolation
    z_targets = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0]
    )

    plot_scale_factor_vs_time(
        run,
        wallclock,
        a,
        args.extrapolate,
        args.interp_range,
        xlim,
        z_targets,
        args.output_dir,
    )

    plot_steps_vs_scale_factor(run, steps, a, args.output_dir)

    plot_time_bins_vs_scale_factor(run, a, time_bins, args.output_dir)


if __name__ == "__main__":
    main()
