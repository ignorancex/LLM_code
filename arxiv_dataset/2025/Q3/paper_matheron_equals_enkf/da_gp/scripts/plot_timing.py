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

"""Generate dual-curve log-log timing plots from benchmark data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Import unified figure styling
from da_gp.figstyle import setup_figure_style
from da_gp.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def _choose_sweep(df: pd.DataFrame, sweep: str):
    """Choose sweep parameters and filter data."""
    if sweep == "obs":
        return "n_obs", "observations (m)", "n_obs", "", df
    if sweep == "dim":
        return "grid_size", "state dimension (d)", "grid_size", "", df
    if sweep == "auto":
        # Auto-detect based on data variation
        n_obs_unique = df["n_obs"].nunique()
        grid_size_unique = df["grid_size"].nunique()

        if n_obs_unique > 1 and grid_size_unique == 1:
            return "n_obs", "observations (m)", "n_obs", "", df
        if grid_size_unique > 1 and n_obs_unique == 1:
            return "grid_size", "state dimension (d)", "grid_size", "", df
        # Fallback - use n_obs if both vary or neither vary
        logger.warning(
            f"Ambiguous sweep type (n_obs unique: {n_obs_unique}, grid_size unique: {grid_size_unique}), defaulting to n_obs"
        )
        return "n_obs", "observations (m)", "n_obs", "", df
    raise ValueError(
        f"Invalid sweep mode: {sweep}. Must be 'obs', 'dim', or 'auto'"
    )


def _draw_axis(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    backend_styles: dict,
    sort_col: str,
    show_legend: bool,
):
    """Draw plotting lines for a single axis."""
    for backend, style in backend_styles.items():
        backend_data = df[df["backend"] == backend]
        if len(backend_data) > 0:
            # Hardening: Skip backends with insufficient data points
            if backend_data.shape[0] < 2:
                logger.warning(
                    f"Skipping {backend} - insufficient data points ({backend_data.shape[0]} < 2)"
                )
                continue

            # Sort by sweep variable for clean line plots
            backend_data = backend_data.sort_values(sort_col)

            ax.loglog(
                backend_data[x_col],
                backend_data[y_col],
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linestyle=style["linestyle"],
                alpha=0.8,
                markersize=6,
            )

    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")


def plot_timing(
    df: pd.DataFrame,
    *,
    output_path: str = "figures/timing_comparison.pdf",
    sweep: str = "auto",
    fixed_n_obs: int = None,
    fixed_grid: int = None,
    colorblind_friendly: bool = False,
    show_legend: bool = True,
):
    """Create unified dual-curve log-log plots showing fit and predict times."""

    # Setup unified styling
    backend_styles = setup_figure_style(colorblind_friendly=colorblind_friendly)

    # Choose sweep parameters
    x_col, x_label, sort_col, _, filtered_df = _choose_sweep(df, sweep)

    # Apply fixed value filters
    if sweep == "dim" or (sweep == "auto" and x_col == "grid_size"):
        # For dimension sweeps, filter by fixed n_obs
        if fixed_n_obs is None:
            # Find the most common n_obs value
            n_obs_counts = filtered_df["n_obs"].value_counts()
            if len(n_obs_counts) == 0:
                logger.error("No data available for dimension scaling plot")
                return None
            fixed_n_obs = n_obs_counts.index[0]

        filtered_df = filtered_df[filtered_df["n_obs"] == fixed_n_obs]
        fixed_label_suffix = f" ($m={fixed_n_obs}$)"

        if len(filtered_df) == 0:
            logger.error(f"No data found for fixed n_obs={fixed_n_obs}")
            return None

    elif sweep == "obs" or (sweep == "auto" and x_col == "n_obs"):
        # For observation sweeps, filter by fixed grid_size if provided
        if fixed_grid is not None:
            filtered_df = filtered_df[filtered_df["grid_size"] == fixed_grid]
            fixed_label_suffix = f" ($d={fixed_grid}$)"

            if len(filtered_df) == 0:
                logger.error(f"No data found for fixed grid_size={fixed_grid}")
                return None
        else:
            fixed_label_suffix = ""
    else:
        fixed_label_suffix = ""

    # Create figure with two subplots side by side
    fig, (ax_fit, ax_pred) = plt.subplots(1, 2)

    # Plot fit times (left subplot)
    ax_fit.set_title(f"Training Time{fixed_label_suffix}")
    ax_fit.set_xlabel(x_label)
    ax_fit.set_ylabel("wall-clock time [s]")

    _draw_axis(
        ax_fit, filtered_df, x_col, "fit_time", backend_styles, sort_col, show_legend
    )

    # Plot predict times (right subplot)
    ax_pred.set_title(f"Prediction Time{fixed_label_suffix}")
    ax_pred.set_xlabel(x_label)
    ax_pred.set_ylabel("wall-clock time [s]")

    _draw_axis(
        ax_pred,
        filtered_df,
        x_col,
        "predict_time",
        backend_styles,
        sort_col,
        show_legend,
    )

    # Adjust layout and save
    plt.tight_layout()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    logger.info(f"Timing plot saved to: {output_path}")

    return fig


def plot_timing_curves(
    df: pd.DataFrame,
    output_path: str = "figures/timing_comparison.pdf",
    colorblind_friendly: bool = False,
    show_legend: bool = True,
):
    """Create dual-curve log-log plots showing fit and predict times.

    DEPRECATED: Use plot_timing() instead. This function is maintained for backwards compatibility.
    """
    return plot_timing(
        df,
        output_path=output_path,
        sweep="auto",
        colorblind_friendly=colorblind_friendly,
        show_legend=show_legend,
    )


def plot_dimension_scaling(
    df: pd.DataFrame,
    output_path: str = "figures/dimension_scaling.pdf",
    fixed_n_obs: int = None,
    colorblind_friendly: bool = False,
    show_legend: bool = True,
):
    """Create plots showing scaling with state dimension.

    DEPRECATED: Use plot_timing() instead. This function is maintained for backwards compatibility.
    """
    return plot_timing(
        df,
        output_path=output_path,
        sweep="dim",
        fixed_n_obs=fixed_n_obs,
        colorblind_friendly=colorblind_friendly,
        show_legend=show_legend,
    )


def main():
    """Main plotting script."""
    parser = argparse.ArgumentParser(description="Generate timing comparison plots")

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

    parser.add_argument("csv_file", help="CSV file containing benchmark timing data")
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Output directory for plots (default: figures)",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--fixed-n-obs",
        type=int,
        help="Fixed n_obs value for dimension scaling plot (default: auto-detect)",
    )
    parser.add_argument(
        "--fixed-grid",
        type=int,
        help="Fixed grid_size value for observation scaling plot (default: auto-detect)",
    )
    parser.add_argument(
        "--colorblind-friendly",
        action="store_true",
        help="Use color-blind friendly palette",
    )
    parser.add_argument(
        "--no-legend", action="store_true", help="Hide legends on plots"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, json=args.log_json)

    # Load data
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        logger.error(f"Error loading data from {args.csv_file}: {e}")
        return 1

    # Sanity guard: Fail fast if wrong CSV is used
    n_obs_unique = df["n_obs"].nunique()
    grid_size_unique = df["grid_size"].nunique()

    if n_obs_unique <= 1 and grid_size_unique <= 1:
        raise ValueError(
            f"Dataset has insufficient variation for timing plots: "
            f"n_obs has {n_obs_unique} unique values, "
            f"grid_size has {grid_size_unique} unique values. "
            f"At least one dimension must have multiple values for meaningful timing analysis."
        )

    logger.info(f"Loaded {len(df)} timing records from {args.csv_file}")
    logger.info(f"Backends: {sorted(df['backend'].unique())}")
    logger.info(f"N_obs range: {df['n_obs'].min()}-{df['n_obs'].max()}")
    logger.info(f"Grid size range: {df['grid_size'].min()}-{df['grid_size'].max()}")
    logger.info("")

    # Generate plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    show_legend = not args.no_legend

    # Only generate plots that make sense for the data
    if n_obs_unique > 1:
        # Observation scaling plot (fit + predict vs n_obs)
        obs_plot_path = output_dir / "timing_vs_observations.pdf"
        plot_timing(
            df,
            output_path=str(obs_plot_path),
            sweep="obs",
            fixed_grid=args.fixed_grid,
            colorblind_friendly=args.colorblind_friendly,
            show_legend=show_legend,
        )
        logger.info(f"Generated observation scaling plot: {obs_plot_path}")
    else:
        logger.info(
            f"Skipping observation scaling plot - only {n_obs_unique} unique n_obs value(s)"
        )

    if grid_size_unique > 1:
        # Dimension scaling plot (fit + predict vs grid_size)
        dim_plot_path = output_dir / "timing_vs_dimensions.pdf"
        plot_timing(
            df,
            output_path=str(dim_plot_path),
            sweep="dim",
            fixed_n_obs=args.fixed_n_obs,
            colorblind_friendly=args.colorblind_friendly,
            show_legend=show_legend,
        )
        logger.info(f"Generated dimension scaling plot: {dim_plot_path}")
    else:
        logger.info(
            f"Skipping dimension scaling plot - only {grid_size_unique} unique grid_size value(s)"
        )

    if args.show:
        plt.show()
    else:
        plt.close("all")

    logger.info("Plotting complete!")
    return 0


if __name__ == "__main__":
    exit(main())
