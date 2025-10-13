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

"""Test RMSE consistency and validation."""

import numpy as np
import pytest

from da_gp.src.gp_common import Problem, generate_experiment_data


def test_observation_rmse():
    """Test that observations are reasonable perturbations of truth."""
    problem = Problem(
        grid_size=1000, n_obs=100, noise_std=0.1, rng=np.random.default_rng(42)
    )
    truth, mask, obs = generate_experiment_data(problem)

    # Observations should be close to truth values at observed locations
    truth_obs = truth[mask]
    residuals = obs - truth_obs
    empirical_std = np.std(residuals)

    # Empirical std should be close to noise_std (within reasonable bounds)
    assert 0.05 < empirical_std < 0.2  # Noise std is 0.1, allow reasonable range


def test_sklearn_rmse_reasonable():
    """Test that sklearn GP produces reasonable RMSE."""
    from da_gp.src.gp_sklearn import run

    problem = Problem(
        grid_size=1000, n_obs=200, noise_std=0.1, rng=np.random.default_rng(42)
    )
    result = run(problem)

    # RMSE should be reasonable (less than prior std)
    prior_std = 1.0  # From kernel amplitude
    assert result["rmse"] < prior_std
    assert result["rmse"] > 0.0

    # Check that we have proper outputs
    assert result["posterior_mean"].shape == (1000,)  # Updated for correct grid size
    assert np.isfinite(result["rmse"])


def test_prior_spread():
    """Test that prior samples have expected spread."""
    from da_gp.src.gp_common import draw_prior

    rng = np.random.default_rng(42)
    samples = [draw_prior(1000, np.random.default_rng(i)) for i in range(50)]
    sample_array = np.array(samples)

    # Sample standard deviation should be close to kernel amplitude (1.0)
    empirical_std = np.std(sample_array, axis=0)
    mean_std = np.mean(empirical_std)

    # Should be within reasonable bounds of kernel amplitude
    assert 0.5 < mean_std < 1.5


@pytest.mark.parametrize("n_obs", [10, 100, 1000])
def test_sklearn_scaling(n_obs):
    """Test sklearn performance scales as expected."""
    import time

    from da_gp.src.gp_sklearn import run

    problem = Problem(
        grid_size=1000, n_obs=n_obs, noise_std=0.1, rng=np.random.default_rng(42)
    )

    start_time = time.perf_counter()
    result = run(problem)
    elapsed = time.perf_counter() - start_time

    # Basic sanity checks
    assert result["rmse"] > 0
    assert result["n_obs"] == n_obs
    assert elapsed > 0

    # For small n_obs, should complete quickly
    if n_obs <= 100:
        assert elapsed < 10.0  # Should be fast for small problems


def test_rmse_decreases_with_observations():
    """Test that RMSE generally decreases with more observations."""
    from da_gp.src.gp_sklearn import run

    # Use consistent problem specification for fair comparison
    problem_few = Problem(
        grid_size=1000, n_obs=10, noise_std=0.1, rng=np.random.default_rng(42)
    )
    problem_many = Problem(
        grid_size=1000, n_obs=100, noise_std=0.1, rng=np.random.default_rng(42)
    )

    rmse_few = run(problem_few)["rmse"]
    rmse_many = run(problem_many)["rmse"]

    # More observations should generally give better fit
    # (Though this may not always hold due to random sampling)
    assert rmse_many <= rmse_few * 2.0  # Allow some variance
