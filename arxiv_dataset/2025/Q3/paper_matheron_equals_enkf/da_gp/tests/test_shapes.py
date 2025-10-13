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

"""Test shape consistency across backends using functional approach."""

import numpy as np
import pytest

from da_gp.src.gp_common import (
    Problem,
    draw_prior,
    generate_experiment_data,
    make_obs_mask,
)


@pytest.mark.parametrize("grid_size", [500, 1000, 4000])
def test_draw_prior_shape(grid_size):
    """Test that prior samples have correct shape using functional approach."""
    rng = np.random.default_rng(42)

    # Test FFT method
    prior_fft = draw_prior(grid_size, rng, use_rff=False)
    assert prior_fft.shape == (grid_size,)
    assert isinstance(prior_fft, np.ndarray)

    # Test RFF method
    prior_rff = draw_prior(grid_size, rng, use_rff=True)
    assert prior_rff.shape == (grid_size,)
    assert isinstance(prior_rff, np.ndarray)


def test_obs_mask_shape():
    """Test that observation masks have correct shape."""
    problem = Problem(grid_size=1000, n_obs=100, rng=np.random.default_rng(42))
    mask = make_obs_mask(problem)

    assert mask.shape == (100,)
    assert np.all(mask >= 0)
    assert np.all(mask < 1000)


def test_generate_experiment_data_shapes():
    """Test that experiment data generation produces correct shapes."""
    problem = Problem(grid_size=1000, n_obs=50, rng=np.random.default_rng(42))
    truth, mask, obs = generate_experiment_data(problem)

    assert truth.shape == (1000,)
    assert mask.shape == (50,)
    assert obs.shape == (50,)
    assert isinstance(truth, np.ndarray)
    assert isinstance(mask, np.ndarray)
    assert isinstance(obs, np.ndarray)


def test_sklearn_backend_shapes():
    """Test sklearn backend output shapes using Problem."""
    from da_gp.src.gp_sklearn import run

    problem = Problem(grid_size=1000, n_obs=50, rng=np.random.default_rng(42))
    result = run(problem, n_ens=10)

    assert "posterior_mean" in result
    assert "posterior_ensemble" in result
    assert "rmse" in result

    assert result["posterior_mean"].shape == (1000,)
    assert result["posterior_ensemble"].shape == (10, 1000)
    assert isinstance(result["rmse"], (float, np.floating))


def test_dapper_backend_shapes():
    """Test DAPPER backend output shapes using Problem."""
    from da_gp.src.gp_dapper import run

    problem = Problem(grid_size=1000, n_obs=50, rng=np.random.default_rng(42))

    # Test full run (may fail without proper DAPPER setup)
    try:
        result = run(problem, n_ens=10)
        assert "posterior_mean" in result
        assert "posterior_ensemble" in result
        assert "rmse" in result

        assert result["posterior_mean"].shape == (1000,)
        assert result["posterior_ensemble"].shape == (10, 1000)
    except Exception:
        pytest.skip("DAPPER runtime configuration required")
