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

"""Regression tests for backend consistency."""

import numpy as np
import pytest


def test_backends_match():
    """Test that different backends produce consistent results."""
    from da_gp.src.gp_common import Problem, generate_experiment_data
    from da_gp.src.gp_dapper import run_enkf
    from da_gp.src.gp_sklearn import run as run_sklearn

    # Test with moderate number of observations
    problem = Problem(
        grid_size=1000, n_obs=300, noise_std=0.1, rng=np.random.default_rng(42)
    )
    truth, mask, obs = generate_experiment_data(problem)

    try:
        result_skl = run_sklearn(problem, truth=truth, mask=mask, obs=obs)
        result_dap = run_enkf(problem, truth=truth, mask=mask, obs=obs)

        # RMSE between means should be reasonable (backends use different algorithms)
        rmse = np.sqrt(
            np.mean((result_skl["posterior_mean"] - result_dap["posterior_mean"]) ** 2)
        )
        assert rmse < 3.0, (
            f"Backend RMSE too large: {rmse:.6f}"
        )  # Relaxed tolerance for different methods

        # Both should produce valid posterior means
        assert np.all(np.isfinite(result_skl["posterior_mean"])), (
            "sklearn backend produced invalid results"
        )
        assert np.all(np.isfinite(result_dap["posterior_mean"])), (
            "dapper backend produced invalid results"
        )

    except ImportError:
        pytest.skip("DAPPER not available for comparison")


def test_backend_output_format():
    """Test that backends return required format."""
    from da_gp.src.gp_common import Problem
    from da_gp.src.gp_sklearn import run as run_sklearn

    problem = Problem(
        grid_size=1000, n_obs=100, noise_std=0.1, rng=np.random.default_rng(42)
    )
    result = run_sklearn(problem)

    # Check required keys are present
    required_keys = ["posterior_mean", "posterior_samples", "obs", "mask"]
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"

    # Check shapes
    assert result["posterior_mean"].shape == (1000,), "Wrong posterior_mean shape"
    assert result["posterior_samples"].shape[1] == 1000, "Wrong posterior_samples shape"
    assert len(result["obs"]) == 100, "Wrong obs length"
    assert len(result["mask"]) == 100, "Wrong mask length"


def test_truth_and_mask_consistency():
    """Test that truth/mask generation is consistent with same problem specifications."""
    from da_gp.src.gp_common import Problem, generate_experiment_data

    # Test with consistent problem specifications
    problem1 = Problem(
        grid_size=1000, n_obs=100, noise_std=0.1, rng=np.random.default_rng(42)
    )
    truth1, mask1, obs1 = generate_experiment_data(problem1)

    problem2 = Problem(
        grid_size=1000, n_obs=100, noise_std=0.1, rng=np.random.default_rng(42)
    )
    truth2, mask2, obs2 = generate_experiment_data(problem2)

    assert np.allclose(truth1, truth2), "Truth generation not deterministic"
    assert np.array_equal(mask1, mask2), "Mask generation not deterministic"
    assert np.allclose(obs1, obs2), "Observation generation not deterministic"


def test_observations_generation():
    """Test observation generation with different noise realizations."""
    from da_gp.src.gp_common import Problem, generate_experiment_data, make_observations

    # Generate consistent truth and mask
    problem = Problem(
        grid_size=1000, n_obs=50, noise_std=0.1, rng=np.random.default_rng(42)
    )
    truth, mask, _ = generate_experiment_data(problem)

    # Generate observations with different noise
    problem1 = Problem(
        grid_size=1000, n_obs=50, noise_std=0.1, rng=np.random.default_rng(1)
    )
    problem2 = Problem(
        grid_size=1000, n_obs=50, noise_std=0.1, rng=np.random.default_rng(2)
    )

    obs1 = make_observations(truth, mask, problem1)
    obs2 = make_observations(truth, mask, problem2)

    # Different noise realizations should be different
    assert not np.allclose(obs1, obs2), "Observations should have different noise"

    # But should be close to truth values
    truth_obs = truth[mask]
    assert np.std(obs1 - truth_obs) < 0.2, "Observation noise too large"
    assert np.std(obs2 - truth_obs) < 0.2, "Observation noise too large"
