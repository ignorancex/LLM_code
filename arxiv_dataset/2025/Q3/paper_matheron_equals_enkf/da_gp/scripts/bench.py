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

"""Benchmark script for DA vs GP performance comparison with internal timing."""

import argparse
import os
import platform
from pathlib import Path
from statistics import median
from typing import NamedTuple

import numpy as np
import pandas as pd

from da_gp.logging_setup import get_logger, setup_logging
from da_gp.src.gp_common import Problem, generate_experiment_data

logger = get_logger(__name__)

DEFAULT_BACKENDS = ["sklearn", "dapper_enkf", "dapper_letkf"]


class Timing(NamedTuple):
    """Timing results for a single experiment."""

    fit_time: float
    predict_time: float
    total_time: float
    rmse: float


def get_backend_runner(backend: str):
    """Get the appropriate backend runner function."""
    if backend == "sklearn":
        from da_gp.src.gp_sklearn import run

        return run
    if backend == "dapper_enkf":
        from da_gp.src.gp_dapper import run_enkf

        return run_enkf
    if backend == "dapper_letkf":
        from da_gp.src.gp_dapper import run_letkf

        return run_letkf
    raise ValueError(f"Unknown backend: {backend}")


def run_experiment_once(
    backend: str, problem: Problem, shared_data: dict[tuple, tuple] = None
) -> Timing:
    """Run single experiment and return timing results.

    Args:
        backend: Backend name ('sklearn', 'dapper_enkf', 'dapper_letkf')
        problem: Problem specification (grid_size, n_obs, noise_std, rng)
        shared_data: Optional pre-generated data to reuse

    Returns:
        Timing results as named tuple
    """
    runner = get_backend_runner(backend)

    # Use shared data if available for this problem configuration
    truth, mask, obs = None, None, None
    key = (problem.n_obs, problem.grid_size)
    if shared_data and key in shared_data:
        truth, mask, obs = shared_data[key]

    # Run experiment (backend will generate data if not provided)
    if backend.startswith("dapper_"):
        result = runner(problem, truth=truth, mask=mask, obs=obs, seed=42)
    else:
        result = runner(problem, truth=truth, mask=mask, obs=obs)

    return Timing(
        fit_time=result["fit_time"],
        predict_time=result["predict_time"],
        total_time=result["total_time"],
        rmse=result["rmse"],
    )


def run_with_repeats(
    backend: str,
    problem: Problem,
    shared_data: dict[tuple, tuple] = None,
    n_repeats: int = 5,
) -> Timing:
    """Run experiment multiple times and return median timings.

    Args:
        backend: Backend name
        problem: Problem specification
        shared_data: Optional pre-generated data to reuse
        n_repeats: Number of repeat runs for statistical robustness

    Returns:
        Median timing results as named tuple
    """
    results = []

    # Warm-up run (discarded) - fail fast on critical errors
    try:
        run_experiment_once(backend, problem, shared_data)
    except Exception as e:
        logger.warning(f"Warm-up failed for {backend}: {e}")
        # Propagate errors during warm-up to help CI detect issues
        raise

    # Collect timing results from multiple runs
    for i in range(n_repeats):
        try:
            # Create fresh problem instance for each repeat to avoid mutation
            fresh_problem = Problem(
                grid_size=problem.grid_size,
                n_obs=problem.n_obs,
                noise_std=problem.noise_std,
                rng=np.random.default_rng(42 + i),  # Different seed per repeat
            )
            timing = run_experiment_once(backend, fresh_problem, shared_data)
            results.append(timing)
        except Exception as e:
            logger.error(f"Experiment {i + 1}/{n_repeats} failed for {backend}: {e}")
            raise

    # Compute medians
    return Timing(
        fit_time=median([r.fit_time for r in results]),
        predict_time=median([r.predict_time for r in results]),
        total_time=median([r.total_time for r in results]),
        rmse=median([r.rmse for r in results]),
    )


def generate_shared_data(n_obs_list: list, grid_size_list: list) -> dict[tuple, tuple]:
    """Generate shared synthetic datasets using functional approach.

    Only generates shared data when all experiments use the same grid size
    to avoid mask index mismatches.

    Args:
        n_obs_list: List of observation counts to generate data for
        grid_size_list: List of grid sizes for experiments

    Returns:
        Dictionary mapping (n_obs, grid_size) -> (truth, mask, obs)
    """
    # Only generate shared data if all experiments use the same grid size
    if len(set(grid_size_list)) > 1:
        logger.info(
            "Multiple grid sizes detected - backends will generate their own data"
        )
        return {}  # Empty dict means backends generate their own data

    # Single grid size - we can safely share data
    shared_data = {}
    grid_size = grid_size_list[0]  # All same size

    for n_obs in n_obs_list:
        # Create problem for this configuration and generate data
        problem = Problem(
            grid_size=grid_size,
            n_obs=n_obs,
            noise_std=0.1,
            rng=np.random.default_rng(42),  # Fixed seed for reproducibility
        )
        truth, mask, obs = generate_experiment_data(problem)
        shared_data[(n_obs, grid_size)] = (truth, mask, obs)

    return shared_data


def log_system_info():
    """Log system information for reproducibility."""
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")

    # Try to get BLAS info
    try:
        import numpy as np

        logger.info(f"NumPy version: {np.__version__}")
        config = np.show_config(mode="dicts")
        if "blas_info" in config:
            logger.info(f"BLAS: {config['blas_info'].get('name', 'unknown')}")
    except Exception:
        pass

    logger.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'unset')}")


def main():
    """Main benchmark runner with internal timing."""
    parser = argparse.ArgumentParser(
        description="Benchmark DA vs GP performance with internal timing"
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

    # Observation sweep parameters
    parser.add_argument(
        "--n_obs_grid", type=int, nargs="+", help="Grid of observation counts to test"
    )
    parser.add_argument(
        "--n_obs_fixed",
        type=int,
        default=2000,
        help="Fixed observation count when sweeping dimensions (default: 2000)",
    )

    # Dimension sweep parameters
    parser.add_argument(
        "--dim_grid",
        type=int,
        nargs="+",
        help="Grid of state dimensions (grid sizes) to test",
    )
    parser.add_argument(
        "--grid_size_fixed",
        type=int,
        default=2000,
        help="Fixed grid size when sweeping observations (default: 2000)",
    )

    parser.add_argument(
        "--csv",
        default="data/bench_timing.csv",
        help="Output CSV file (default: data/bench_timing.csv)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        choices=["sklearn", "dapper_enkf", "dapper_letkf"],
        help="Backends to test (default: all)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timing repeats per configuration (default: 5)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, json=args.log_json)

    # Validate arguments - need either obs_grid or dim_grid
    if not args.n_obs_grid and not args.dim_grid:
        parser.error("Must specify either --n_obs_grid or --dim_grid (or both)")

    # Ensure output directory exists
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BENCHMARK WITH INTERNAL TIMING")
    logger.info("=" * 60)
    log_system_info()

    logger.info(f"Backends: {args.backends}")
    logger.info(f"Timing repeats: {args.repeats}")
    if args.n_obs_grid:
        logger.info(
            f"Observation counts: {args.n_obs_grid} (grid_size={args.grid_size_fixed})"
        )
    if args.dim_grid:
        logger.info(f"Grid sizes: {args.dim_grid} (n_obs={args.n_obs_fixed})")
    logger.info(f"Output: {args.csv}")
    logger.info("")

    # Build complete parameter lists - ensure separate sweeps
    if args.n_obs_grid and not args.dim_grid:
        # Pure observation sweep
        n_obs_list = args.n_obs_grid
        grid_size_list = [args.grid_size_fixed]
    elif args.dim_grid and not args.n_obs_grid:
        # Pure dimension sweep
        n_obs_list = [args.n_obs_fixed]
        grid_size_list = args.dim_grid
    elif args.n_obs_grid and args.dim_grid:
        # Both sweeps specified - validate early to avoid single-point collapse
        if len(args.n_obs_grid) == 1 and len(args.dim_grid) == 1:
            parser.error(
                f"When both --n_obs_grid and --dim_grid are specified, at least one must have multiple values. Got n_obs_grid={args.n_obs_grid}, dim_grid={args.dim_grid}"
            )

        # Both sweeps specified - this creates a cross-product
        n_obs_list = args.n_obs_grid + [args.n_obs_fixed]
        grid_size_list = [args.grid_size_fixed] + args.dim_grid
    else:
        # Should not reach here due to earlier validation
        raise ValueError("Must specify either --n_obs_grid or --dim_grid")

    # Remove duplicates and sort
    n_obs_list = sorted(set(n_obs_list))
    grid_size_list = sorted(set(grid_size_list))

    # Final validation: ensure we have multiple data points in at least one dimension
    if len(n_obs_list) <= 1 and len(grid_size_list) <= 1:
        raise ValueError(
            f"Sweep collapsed to single point: n_obs={n_obs_list}, grid_size={grid_size_list}. Need multiple values in at least one dimension."
        )

    logger.info("Generating shared synthetic datasets...")
    shared_data = generate_shared_data(n_obs_list, grid_size_list)
    logger.info(f"Generated {len(shared_data)} datasets")
    logger.info("")

    # Collect results
    results = []
    total_runs = len(args.backends) * len(n_obs_list) * len(grid_size_list)
    current_run = 0

    for backend in args.backends:
        for grid_size in grid_size_list:
            for n_obs in n_obs_list:
                current_run += 1
                logger.info(
                    f"[{current_run}/{total_runs}] {backend} (n_obs={n_obs}, grid_size={grid_size})"
                )

                # Create problem instance for this configuration
                problem = Problem(
                    grid_size=grid_size,
                    n_obs=n_obs,
                    noise_std=0.1,
                    rng=np.random.default_rng(42),  # Fixed seed for reproducibility
                )

                timing_result = run_with_repeats(
                    backend, problem, shared_data, args.repeats
                )

                results.append(
                    {
                        "backend": backend,
                        "n_obs": n_obs,
                        "grid_size": grid_size,
                        "fit_time": timing_result.fit_time,
                        "predict_time": timing_result.predict_time,
                        "total_time": timing_result.total_time,
                        "rmse": timing_result.rmse,
                    }
                )

                logger.info(
                    f"  fit: {timing_result.fit_time:.3f}s, predict: {timing_result.predict_time:.3f}s, total: {timing_result.total_time:.3f}s"
                )
                logger.info(f"  RMSE: {timing_result.rmse:.4f}")
                logger.info("")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.csv, index=False)

    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {args.csv}")
    logger.info(f"Total experiments: {len(results)}")
    logger.info("")

    # Log summary statistics
    logger.info("SUMMARY:")
    for backend in args.backends:
        backend_results = df[df["backend"] == backend]
        if len(backend_results) > 0:
            logger.info(f"{backend}:")
            logger.info(f"  Mean fit time: {backend_results['fit_time'].mean():.3f}s")
            logger.info(
                f"  Mean predict time: {backend_results['predict_time'].mean():.3f}s"
            )
            logger.info(f"  Mean RMSE: {backend_results['rmse'].mean():.4f}")
            logger.info("")


if __name__ == "__main__":
    main()
