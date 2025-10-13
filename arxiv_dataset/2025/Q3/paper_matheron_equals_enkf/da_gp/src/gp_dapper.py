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

# FILE: da-gp/src/gp_dapper.py

import time

import numpy as np
from dapper import da_methods, mods
from dapper.tools.randvars import RV
from dapper.tools.seeding import set_seed

from .gp_common import Problem, draw_prior, generate_experiment_data


def _run(
    problem: Problem,
    *,
    n_ens: int = 40,
    truth: np.ndarray = None,
    mask: np.ndarray = None,
    obs: np.ndarray = None,
    method: str = "EnKF",
    seed: int = None,
) -> dict:
    """Internal, generalized DAPPER runner.

    Args:
        problem: Problem specification containing grid_size, n_obs, noise_std, rng
        n_ens: Number of ensemble members
        truth: Optional pre-generated truth (if None, generates from problem)
        mask: Optional pre-generated observation mask (if None, generates from problem)
        obs: Optional pre-generated observations (if None, generates from problem)
        method: DA method ('EnKF' or 'LETKF')
        seed: Optional seed for DAPPER's internal seeding

    Returns:
        Dictionary with posterior statistics and timing information
    """
    # Set DAPPER's internal seed if provided
    if seed is not None:
        set_seed(seed)

    # Use provided data or generate from problem specification
    if truth is None or mask is None or obs is None:
        truth, mask, obs = generate_experiment_data(problem)

    # Prior: Zero-mean ensemble with kernel-derived covariance
    def init_ensemble():
        return np.stack(
            [draw_prior(problem.grid_size, problem.rng) for _ in range(n_ens)]
        )

    initial_ensemble = init_ensemble()
    X0 = RV(M=problem.grid_size, func=lambda N: initial_ensemble)

    # HMM components: Use the minimal working timeline
    Dyn = {"M": problem.grid_size, "model": mods.Id_op(), "noise": 0}
    Obs = mods.partial_Id_Obs(problem.grid_size, mask)
    Obs["noise"] = 0.01  # Variance

    # Add localizer for LETKF if needed
    if method == "LETKF":
        from dapper.tools.localization import nd_Id_localization

        Obs["localizer"] = nd_Id_localization((problem.grid_size,), obs_inds=mask)

    tseq = mods.Chronology(dt=1, dko=1, T=1)  # Working chronology
    HMM = mods.HiddenMarkovModel(Dyn, Obs, tseq=tseq, X0=X0)

    # --- Select DA method based on argument ---
    if method == "EnKF":
        da_method = da_methods.EnKF("Serial", N=n_ens, infl=1.0, rot=False)
    elif method == "LETKF":
        da_method = da_methods.LETKF(N=n_ens, infl=1.05, loc_rad=50)
    else:
        raise ValueError(f"Unknown DAPPER method: {method}")

    # Data shapes matching the timeline
    xx = np.vstack([truth, truth])
    yy = obs.reshape(1, -1)

    # Time the assimilation (fit) step
    t0 = time.perf_counter()
    da_method.assimilate(HMM, xx, yy)
    fit_time = time.perf_counter() - t0

    # Time the prediction/extraction (predict) step: read out the mean only
    t0 = time.perf_counter()
    posterior_mean = da_method.stats.mu.a[0]
    predict_time = time.perf_counter() - t0

    # Build posterior ensemble OUTSIDE the predict timer (needed for downstream plotting),
    # and do it in O(N d) without forming a dense d x d covariance.
    ens = getattr(da_method, "E", None)
    if ens is not None:
        posterior_ensemble = ens
    else:
        sigma = float(da_method.stats.spread.a[0].mean())
        posterior_ensemble = posterior_mean + sigma * problem.rng.standard_normal(
            size=(n_ens, problem.grid_size)
        )
    truth_at_analysis_time = xx[HMM.tseq.kko[0]]

    return {
        "posterior_mean": posterior_mean,
        "posterior_ensemble": posterior_ensemble,
        "posterior_samples": posterior_ensemble,
        "obs": obs,
        "mask": mask,
        "rmse": np.sqrt(np.mean((posterior_mean - truth_at_analysis_time) ** 2)),
        "fit_time": fit_time,
        "predict_time": predict_time,
        "total_time": fit_time + predict_time,
        "n_ens": n_ens,
        "n_obs": problem.n_obs,
    }


def run_enkf(problem: Problem, **kwargs):
    """Public-facing runner for standard EnKF."""
    return _run(problem, method="EnKF", **kwargs)


def run_letkf(problem: Problem, **kwargs):
    """Public-facing runner for LETKF."""
    return _run(problem, method="LETKF", **kwargs)


def run(problem: Problem, **kwargs):
    """Main entry point: defaults to EnKF."""
    return run_enkf(problem, **kwargs)
