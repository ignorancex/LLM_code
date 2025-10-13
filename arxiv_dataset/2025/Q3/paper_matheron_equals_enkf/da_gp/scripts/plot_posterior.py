#!/usr/bin/env python3
# Copyright (c) 2025 Commonwealth Scientific and Industrial Research Organisation (CSIRO)
#
# All rights reserved.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate posterior comparison plots with truth, observations, and samples."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import unified styling
from da_gp.figstyle import setup_figure_style
from da_gp.logging_setup import get_logger, setup_logging
from da_gp.src.gp_common import Problem, generate_experiment_data

logger = get_logger(__name__)


def run_backend(
    backend: str,
    problem: Problem,
    truth: np.ndarray,
    mask: np.ndarray,
    obs: np.ndarray,
    n_draws: int = 200,
):
    """Run a specific backend with given data."""
    if backend == "sklearn":
        from da_gp.src.gp_sklearn import run

        return run(problem, truth=truth, mask=mask, obs=obs, n_ens=n_draws)
    if backend == "dapper_enkf":
        from da_gp.src.gp_dapper import run_enkf

        return run_enkf(
            problem, n_ens=n_draws, truth=truth, mask=mask, obs=obs, seed=42
        )
    if backend == "dapper_letkf":
        from da_gp.src.gp_dapper import run_letkf

        return run_letkf(
            problem, n_ens=n_draws, truth=truth, mask=mask, obs=obs, seed=42
        )
    raise ValueError(f"Unknown backend: {backend}")


def main():
    """Generate posterior comparison plot."""
    parser = argparse.ArgumentParser(
        description="Generate posterior plots with dual-backend comparison"
    )

    # Logging arguments
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="WARNING",
        help="Set the logging level (default: WARNING)",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Use JSON formatting for logs",
    )

    parser.add_argument(
        "--backends",
        nargs="+",
        default=["sklearn", "dapper_enkf", "dapper_letkf"],
        choices=["sklearn", "dapper_enkf", "dapper_letkf"],
        help="Backends to compare",
    )
    parser.add_argument(
        "--n_obs", type=int, default=50, help="Number of observations (default: 50)"
    )
    parser.add_argument(
        "--n_draws",
        type=int,
        default=50,
        help="Number of posterior draws per backend (default: 50)",
    )
    parser.add_argument(
        "--out", default="figures/posterior_samples.pdf", help="Output figure path"
    )
    parser.add_argument(
        "--show_truth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the ground truth as a thick dotted line",
    )
    parser.add_argument(
        "--colorblind-friendly",
        action="store_true",
        help="Use color-blind friendly palette",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=2000,
        help="Grid size (state dimension, default: 2000)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, json=args.log_json)

    # Create the single, master RNG instance for this run
    rng = np.random.default_rng(42)

    # --- FIX 1: Use an accessible color palette ---
    # Use 'viridis', a perceptually uniform and color-blind-friendly palette
    color_map = plt.colormaps["viridis"]
    colors = {
        backend: color_map(i / (len(args.backends)))
        for i, backend in enumerate(args.backends)
    }
    # Add black for specific elements if needed
    colors["truth"] = "black"
    colors["obs"] = "black"

    # --- FIX 3: Calculate dynamic alpha for posterior samples ---
    # Alpha scales down as the number of draws increases to prevent solid bands
    dynamic_alpha = np.clip(1.0 / float(args.n_draws) ** 0.75, 0.01, 1.0)

    # Ensure output directory exists
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Generate consistent data for all backends using Problem-based approach
    logger.info(f"Generating data with {args.n_obs} observations...")
    problem = Problem(
        grid_size=args.grid_size, n_obs=args.n_obs, noise_std=0.1, rng=rng
    )
    truth, mask, obs = generate_experiment_data(problem)

    # Set up plotting
    # Setup unified styling
    backend_styles = setup_figure_style(colorblind_friendly=args.colorblind_friendly)
    plt.figure()
    x = np.arange(problem.grid_size)  # Create local grid coordinates

    # Plot for each backend
    for backend in args.backends:
        try:
            logger.info(f"Running {backend} backend...")
            result = run_backend(backend, problem, truth, mask, obs, args.n_draws)
            samples = result["posterior_samples"]
            color = colors.get(backend, "gray")

            # --- FIX 2: Set zorder for posterior samples ---
            # Lower zorder means they are drawn first (in the background).
            plt.plot(x, samples.T, lw=0.5, color=color, alpha=dynamic_alpha, zorder=10)

            # Add a single representative line for the legend
            plt.plot([], [], lw=2, color=color, label=f"{backend.upper()}")

            logger.info(f"  {backend}: RMSE = {result.get('rmse', 0.0):.6f}")

        except Exception as e:
            logger.error(f"Error with {backend}: {e}")
            continue

    # Plot truth (optional)
    if args.show_truth:
        # --- FIX 2: Set zorder for truth line ---
        # Draw it behind observations but on top of the posterior cloud.
        plt.plot(
            x,
            truth,
            color=colors["truth"],
            lw=1.5,
            ls="--",
            alpha=0.8,
            label="True Field",
            zorder=20,
        )

    # Plot observations
    plt.scatter(
        x[mask],
        obs,
        marker="x",
        color=colors["obs"],
        s=40,
        linewidths=1.5,
        alpha=1.0,
        label="Observations",
        # --- FIX 2: Set high zorder to ensure observations are on top ---
        zorder=30,
    )

    # Formatting
    plt.xlabel("State Dimension (Grid Index)")
    plt.ylabel("Value")
    plt.title(
        f"Posterior Comparison (Observations = {args.n_obs}, Draws = {args.n_draws})"
    )
    leg = plt.legend(loc="upper right", frameon=True, framealpha=0.95)
    # Ensure legend is painted above all plot elements
    leg.set_zorder(1000)
    # Improve readability of legend
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")

    plt.grid(True, which="both", ls="--", alpha=0.2)
    plt.tight_layout()

    # Save figure
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    logger.info(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
