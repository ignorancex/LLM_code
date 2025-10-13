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

"""Test Problem-based functional approach and benchmark pipeline."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from da_gp.src.gp_common import Problem


@pytest.mark.parametrize("grid_size", [500, 1000, 4000])
def test_benchmark_with_different_grid_sizes(grid_size):
    """Test that benchmark pipeline works with different grid sizes using functional approach."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / f"test_grid_{grid_size}.csv"

        # Run benchmark with specific grid size
        cmd = [
            sys.executable,
            "-m",
            "da_gp.scripts.bench",
            "--n_obs_grid",
            "50",
            "100",
            "--grid_size_fixed",
            str(grid_size),
            "--backends",
            "sklearn",
            "--csv",
            str(csv_path),
            "--repeats",
            "1",  # Minimal for testing
        ]

        # This should not raise any exceptions
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        if result.returncode != 0:
            pytest.fail(
                f"Benchmark failed for grid_size={grid_size}:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Verify CSV was created and has data
        assert csv_path.exists(), f"CSV file not created for grid_size={grid_size}"

        import pandas as pd

        df = pd.read_csv(csv_path)
        assert len(df) > 0, f"CSV is empty for grid_size={grid_size}"
        assert (df["grid_size"] == grid_size).all(), "Grid size mismatch in CSV"


def test_dimension_scaling_benchmark():
    """Test dimension scaling benchmark to ensure no broadcast errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_dim_scaling.csv"

        # Run dimension scaling benchmark
        cmd = [
            sys.executable,
            "-m",
            "da_gp.scripts.bench",
            "--dim_grid",
            "250",
            "500",
            "1000",
            "--n_obs_fixed",
            "100",
            "--backends",
            "sklearn",
            "--csv",
            str(csv_path),
            "--repeats",
            "1",
        ]

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        if result.returncode != 0:
            pytest.fail(
                f"Dimension scaling benchmark failed:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Verify results
        assert csv_path.exists()
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Should have 3 different grid sizes
        unique_sizes = df["grid_size"].unique()
        assert len(unique_sizes) == 3
        assert set(unique_sizes) == {250, 500, 1000}


def test_problem_consistency():
    """Test that Problem instances ensure consistency and prevent mismatch errors."""
    from da_gp.src.gp_common import generate_experiment_data
    from da_gp.src.gp_sklearn import run

    # Create problem and generate data
    problem1 = Problem(
        grid_size=1000, n_obs=50, noise_std=0.1, rng=np.random.default_rng(42)
    )
    truth, mask, obs = generate_experiment_data(problem1)

    # With functional approach, using the same problem should always work
    result = run(problem1, truth=truth, mask=mask, obs=obs)
    assert result["posterior_mean"].shape == (1000,)
    assert result["posterior_mean"].shape == truth.shape

    # Different problem with different grid size should also work
    problem2 = Problem(
        grid_size=500, n_obs=50, noise_std=0.1, rng=np.random.default_rng(42)
    )
    result2 = run(problem2)  # Generate its own data
    assert result2["posterior_mean"].shape == (500,)


def test_functional_determinism():
    """Test that functional approach gives deterministic results."""
    from da_gp.src.gp_sklearn import run

    # Same problem, same seed should give identical results
    problem = Problem(
        grid_size=500, n_obs=50, noise_std=0.1, rng=np.random.default_rng(42)
    )
    result1 = run(problem)

    # Create identical problem
    problem2 = Problem(
        grid_size=500, n_obs=50, noise_std=0.1, rng=np.random.default_rng(42)
    )
    result2 = run(problem2)

    # Results should be identical (deterministic)
    assert result1["rmse"] == result2["rmse"]
    np.testing.assert_array_equal(result1["posterior_mean"], result2["posterior_mean"])


@pytest.mark.parametrize("backend", ["sklearn"])  # Add dapper backends when available
@pytest.mark.parametrize("grid_size", [500, 1000, 2000])
def test_single_backend_different_sizes(backend, grid_size):
    """Test individual backend with different grid sizes using functional approach."""
    if backend == "sklearn":
        from da_gp.src.gp_sklearn import run

    # Create problem for this grid size
    problem = Problem(
        grid_size=grid_size, n_obs=50, noise_std=0.1, rng=np.random.default_rng(42)
    )

    # Run experiment
    result = run(problem)

    # Should work correctly with proper shapes
    assert result["posterior_mean"].shape == (grid_size,)
    assert "rmse" in result
    assert result["rmse"] > 0  # Should be reasonable RMSE value


def test_problem_validation():
    """Test that Problem validates inputs properly."""
    # Valid problem should work
    problem = Problem(grid_size=1000, n_obs=50, noise_std=0.1)
    assert problem.grid_size == 1000
    assert problem.n_obs == 50

    # Invalid inputs should raise errors
    with pytest.raises(ValueError, match="grid_size must be positive"):
        Problem(grid_size=0, n_obs=50)

    with pytest.raises(ValueError, match="n_obs must be positive"):
        Problem(grid_size=1000, n_obs=0)

    with pytest.raises(ValueError, match="noise_std must be non-negative"):
        Problem(grid_size=1000, n_obs=50, noise_std=-1.0)
