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

"""Common utilities for GP and DA experiments."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.gaussian_process.kernels import RBF

# Set up logger for this module
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Problem:
    """Immutable problem specification for GP/DA experiments.

    This replaces global state variables and ensures all functions receive
    explicit configuration rather than relying on hidden module-level state.
    """

    grid_size: int
    n_obs: int
    noise_std: float = 0.1
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    def __post_init__(self):
        """Validate problem parameters."""
        if self.grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {self.grid_size}")
        if self.n_obs <= 0:
            raise ValueError(f"n_obs must be positive, got {self.n_obs}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")


# Constants
RFF_DIM = 1024  # number of random features (tunable)


# Pure helper functions to replace global state
def make_grid(grid_size: int) -> np.ndarray:
    """Create spatial grid for given size."""
    return np.arange(grid_size).reshape(-1, 1)


def make_kernel() -> Any:
    """Create RBF kernel with fixed hyperparameters."""
    return 1.0 * RBF(length_scale=30.0)


def make_rff_params(
    grid_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Create Random Fourier Feature parameters for given grid size."""
    kernel = make_kernel()
    # Extract length scale from the RBF kernel
    length_scale = kernel.k2.length_scale if hasattr(kernel, "k2") else 10.0

    W = rng.normal(0, 1 / length_scale, size=(RFF_DIM, 1))
    b = rng.uniform(0, 2 * np.pi, size=RFF_DIM)
    return W, b


def draw_prior_fft(grid_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw an exact sample from the GP prior using FFT-based circulant embedding.
    Pure functional version that takes explicit grid_size parameter.

    Args:
        grid_size: Size of the spatial grid
        rng: Random number generator

    Returns:
        Sample from the GP prior of shape (grid_size,)
    """
    X_grid = make_grid(grid_size)
    kernel = make_kernel()

    # 1. Compute the first row of the covariance matrix
    cov_row = kernel(X_grid[:1], X_grid).flatten()

    # 2. Embed into circulant row for periodic convolution
    N_circ = 2 * grid_size
    circ_row = np.zeros(N_circ)
    circ_row[:grid_size] = cov_row
    circ_row[N_circ - grid_size + 1 :] = cov_row[1:][::-1]

    # 3. Eigenvalues are FFT of first row
    lambda_ = np.fft.fft(circ_row).real
    lambda_ = np.maximum(lambda_, 0)  # Ensure non-negativity

    # 4. Generate complex Gaussian white noise in Fourier domain
    noise_freq = rng.normal(size=N_circ) + 1j * rng.normal(size=N_circ)
    noise_freq[0] = rng.normal()  # DC component is real
    if N_circ % 2 == 0:
        noise_freq[N_circ // 2] = rng.normal()  # Nyquist component is real
    else:
        noise_freq[(N_circ + 1) // 2 :] = np.conj(
            noise_freq[1 : (N_circ + 1) // 2][::-1]
        )

    # 5. Color the noise and transform back
    sample_freq = noise_freq * np.sqrt(lambda_)
    sample = np.fft.ifft(sample_freq).real * np.sqrt(N_circ)

    # 6. Truncate to original grid size
    return sample[:grid_size]


def draw_prior_rff(grid_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw a sample from the GP prior using Random Fourier Features.
    Pure functional version that takes explicit grid_size parameter.

    Args:
        grid_size: Size of the spatial grid
        rng: Random number generator

    Returns:
        Sample from the GP prior of shape (grid_size,)
    """
    X_grid = make_grid(grid_size)
    W, b = make_rff_params(grid_size, rng)

    # RFF mapping function
    def phi(x: np.ndarray) -> np.ndarray:
        return np.sqrt(2 / RFF_DIM) * np.cos(x @ W.T + b)

    z = rng.standard_normal(RFF_DIM)  # a_j coefficients
    return phi(X_grid) @ z  # O(d m) matrix-vector product


def draw_prior(
    grid_size: int, rng: np.random.Generator, use_rff: bool = False
) -> np.ndarray:
    """
    Draw a sample from the GP prior using FFT (default) or RFF.

    Args:
        grid_size: Size of the spatial grid
        rng: Random number generator
        use_rff: If True, use Random Fourier Features. If False, use FFT method.

    Returns:
        Sample from the GP prior of shape (grid_size,)
    """
    if use_rff:
        return draw_prior_rff(grid_size, rng)
    return draw_prior_fft(grid_size, rng)


# Functional data generation helpers
def make_obs_mask(problem: Problem) -> np.ndarray:
    """Create observation mask by randomly selecting grid points."""
    return problem.rng.choice(problem.grid_size, size=problem.n_obs, replace=True)


def generate_truth(problem: Problem) -> np.ndarray:
    """Generate synthetic truth for validation."""
    return draw_prior(problem.grid_size, problem.rng)


def make_observations(
    truth: np.ndarray, mask: np.ndarray, problem: Problem
) -> np.ndarray:
    """Generate noisy observations from truth at masked locations."""
    obs = truth[mask]
    noise = problem.rng.normal(0, problem.noise_std, size=len(obs))
    return obs + noise


def generate_experiment_data(
    problem: Problem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate complete experiment dataset (truth, mask, observations).

    Args:
        problem: Problem specification

    Returns:
        truth, mask, observations
    """
    truth = generate_truth(problem)
    mask = make_obs_mask(problem)
    obs = make_observations(truth, mask, problem)
    return truth, mask, obs


def make_dataset(
    grid_size: int = 2000, n_obs: int = 100, noise_std: float = 0.1, seed: int = None
) -> tuple[Problem, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience factory for creating a complete dataset.

    This is a shorthand for creating a Problem and generating experimental data,
    useful for notebooks, interactive sessions, and simple test cases.

    Args:
        grid_size: State dimension (default: 2000)
        n_obs: Number of observations (default: 100)
        noise_std: Observation noise standard deviation (default: 0.1)
        seed: Random seed for reproducibility (default: None)

    Returns:
        problem, truth, mask, observations

    Example:
        >>> problem, truth, mask, obs = make_dataset(grid_size=1000, n_obs=50, seed=42)
        >>> result = run_sklearn(problem, truth=truth, mask=mask, obs=obs)
    """
    rng = np.random.default_rng(seed)
    problem = Problem(grid_size=grid_size, n_obs=n_obs, noise_std=noise_std, rng=rng)
    truth, mask, obs = generate_experiment_data(problem)
    return problem, truth, mask, obs
