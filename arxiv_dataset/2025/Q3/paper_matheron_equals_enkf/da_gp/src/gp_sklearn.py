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

"""Scikit-learn baseline for Gaussian Process regression."""

import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from .gp_common import Problem, generate_experiment_data, make_grid, make_kernel


def run(
    problem: Problem,
    *,
    truth: np.ndarray = None,
    mask: np.ndarray = None,
    obs: np.ndarray = None,
    **kwargs,
) -> dict:
    """Run scikit-learn Gaussian Process regression.

    Args:
        problem: Problem specification containing grid_size, n_obs, noise_std, rng
        truth: Optional pre-generated truth (if None, generates from problem)
        mask: Optional pre-generated observation mask (if None, generates from problem)
        obs: Optional pre-generated observations (if None, generates from problem)
        **kwargs: Additional configuration (e.g. n_ens for sample count)

    Returns:
        Dictionary with posterior statistics and timing information
    """
    # Use provided data or generate from problem specification
    if truth is None or mask is None or obs is None:
        truth, mask, obs = generate_experiment_data(problem)

    # Create spatial grid and kernel
    X_grid = make_grid(problem.grid_size)
    kernel = make_kernel()

    # Create GP with fixed kernel (no optimization)
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, alpha=0.01)

    # Fit GP to observations
    start_time = time.perf_counter()
    X_obs = X_grid[mask]
    gp.fit(X_obs, obs)
    fit_time = time.perf_counter() - start_time

    # Predict at all grid points: time mean-only for fair comparison with DA
    start_time = time.perf_counter()
    posterior_mean = gp.predict(X_grid, return_std=False)
    predict_time = time.perf_counter() - start_time

    # Get std separately (untimed) for sampling
    posterior_mean_full, posterior_std = gp.predict(X_grid, return_std=True)

    # Sample from posterior (approximate) using problem's RNG
    n_samples = kwargs.get("n_ens", 40)
    posterior_samples = problem.rng.standard_normal(
        size=(n_samples, len(posterior_mean))
    )
    posterior_samples = posterior_mean + posterior_samples * posterior_std

    # Compute RMSE
    rmse = np.sqrt(np.mean((posterior_mean - truth) ** 2))

    return {
        "posterior_mean": posterior_mean,
        "posterior_ensemble": posterior_samples,
        "posterior_samples": posterior_samples,  # Required format
        "posterior_std": posterior_std,
        "obs": obs,  # Required format
        "mask": mask,  # Required format
        "rmse": rmse,
        "fit_time": fit_time,
        "predict_time": predict_time,
        "total_time": fit_time + predict_time,
        "n_obs": problem.n_obs,
        "log_marginal_likelihood": gp.log_marginal_likelihood(),
    }
