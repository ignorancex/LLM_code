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

"""Command-line interface for da-gp benchmark."""

import argparse
import sys
import time
from typing import Any

import numpy as np

from da_gp.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

try:
    from mpi4py import MPI

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def _load_backend(
    backend: str, n_obs: int, grid_size: int = 2000, n_ens: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load backend and return posterior statistics for plotting.

    Returns:
        mean: Posterior mean
        lower: Lower confidence bound (mean - 2*std)
        upper: Upper confidence bound (mean + 2*std)
        x: Grid coordinates
    """
    from .gp_common import Problem

    problem = Problem(
        grid_size=grid_size, n_obs=n_obs, noise_std=0.1, rng=np.random.default_rng(42)
    )

    if backend == "sklearn":
        from . import gp_sklearn as module

        result = module.run(problem, n_ens=n_ens)
        mean = result["posterior_mean"]
        std = result.get("posterior_std", np.ones_like(mean) * 0.1)

    elif backend == "dapper_enkf":
        from . import gp_dapper as module

        result = module.run_enkf(problem, n_ens=n_ens)
        mean = result["posterior_mean"]
        ensemble = result.get("posterior_ensemble", np.zeros((n_ens, len(mean))))
        std = (
            np.std(ensemble, axis=0) if ensemble.size > 0 else np.ones_like(mean) * 0.1
        )

    elif backend == "dapper_letkf":
        from . import gp_dapper as module

        result = module.run_letkf(problem, n_ens=n_ens)
        mean = result["posterior_mean"]
        ensemble = result.get("posterior_ensemble", np.zeros((n_ens, len(mean))))
        std = (
            np.std(ensemble, axis=0) if ensemble.size > 0 else np.ones_like(mean) * 0.1
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    x = np.arange(len(mean))  # Recreate grid coordinates on the fly
    lower = mean - 2 * std
    upper = mean + 2 * std

    return mean, lower, upper, x


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data Assimilation vs Gaussian Process benchmark"
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
        "--backend",
        choices=["sklearn", "dapper_enkf", "dapper_letkf"],
        required=True,
        help="Backend to use for the experiment",
    )
    parser.add_argument(
        "--n_obs",
        type=int,
        default=5_000,
        help="Number of observations (default: 5000)",
    )
    parser.add_argument(
        "--n_ens", type=int, default=40, help="Number of ensemble members (default: 40)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument(
        "--grid_size",
        type=int,
        default=None,
        help="State dimension d (overrides gp_common.GRID_SIZE)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, json=args.log_json)

    # Set default grid size if not provided
    grid_size = args.grid_size if args.grid_size is not None else 2000

    # Check MPI rank for multi-process backends
    rank = 0
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.rank

    # Import and run backend using Problem-based API
    start_time = time.perf_counter()

    try:
        from .gp_common import Problem

        problem = Problem(
            grid_size=grid_size,
            n_obs=args.n_obs,
            noise_std=0.1,
            rng=np.random.default_rng(42),
        )

        if args.backend == "sklearn":
            from . import gp_sklearn as backend

            result = backend.run(problem, n_ens=args.n_ens)
        elif args.backend == "dapper_enkf":
            from . import gp_dapper as backend

            result = backend.run_enkf(problem, n_ens=args.n_ens)
        elif args.backend == "dapper_letkf":
            from . import gp_dapper as backend

            result = backend.run_letkf(problem, n_ens=args.n_ens)
        else:
            raise ValueError(f"Unknown backend: {args.backend}")

    except ImportError as e:
        if rank == 0:
            logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        if rank == 0:
            logger.error(f"Error running {args.backend}: {e}")
        sys.exit(1)

    elapsed_time = time.perf_counter() - start_time

    # Print results (only from rank 0 for MPI jobs)
    if rank == 0:
        print_results(args, result, elapsed_time)


def print_results(
    args: argparse.Namespace, result: dict[str, Any], elapsed_time: float
) -> None:
    """Print benchmark results."""
    grid_size = args.grid_size if args.grid_size is not None else 2000

    print(f"Backend: {args.backend}")
    print(f"Grid size: {grid_size:,}")
    print(f"Observations: {args.n_obs:,}")
    print(f"Ensemble size: {args.n_ens}")
    print(f"Total elapsed time: {elapsed_time:.3f}s")

    # Print internal timings if available
    if "fit_time" in result and "predict_time" in result:
        print(f"  Fit time: {result['fit_time']:.3f}s")
        print(f"  Predict time: {result['predict_time']:.3f}s")
        print(
            f"  Internal total: {result.get('total_time', result['fit_time'] + result['predict_time']):.3f}s"
        )

    if "rmse" in result:
        print(f"RMSE: {result['rmse']:.6f}")

    if args.verbose:
        print("\nDetailed results:")
        for key, value in result.items():
            if key not in ["posterior_mean", "posterior_ensemble", "posterior_std"]:
                print(f"  {key}: {value}")

    # Enhanced CSV output for benchmarking - now includes separate fit/predict times
    fit_time = result.get("fit_time", elapsed_time)
    predict_time = result.get("predict_time", 0.0)
    rmse = result.get("rmse", 0.0)
    print(
        f"\nCSV: {args.backend},{args.n_obs},{grid_size},{fit_time:.6f},{predict_time:.6f},{elapsed_time:.6f},{rmse:.6f}"
    )


if __name__ == "__main__":
    main()
