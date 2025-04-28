# step2_fitgps.py
"""Fit Gaussian processes (GPs) to snapshot training data."""

__all__ = [
    "fit_single_gaussian_process",
    "fit_gaussian_processes",
]

import numpy as np
from typing import Iterable

import opinf

import config
import gpkernels


def fit_single_gaussian_process(
    stateindex: int,
    time_domain_training: np.ndarray,
    time_domain_sampled: np.ndarray,
    state_variable_sampled: np.ndarray,
    gp_regularizer: float = 1e-8,
) -> gpkernels.GP_RBFW:
    """Fit a single Gaussian process (GP) to snapshot data for one variable.

    Parameters
    ----------
    stateindex : int
        Index of the state variable.
    time_domain_training : (m',) ndarray
        Time domain at which to estimate states and time derivatives
        for the parameter estimation.
    time_domain_sampled : (m,) ndarray
        Time domain corresponding to the available training snapshots.
    state_variable_sampled : (m,) ndarray
        Observations of a single state variable over the time domain
        ``time_domain_sampled``.
    gp_regularizer : float >= 0
        Regularization hyperparameter for the GP inference in inverting for
        the least-squares weight matrix.

    Returns
    -------
    gpkernels.GP_RBFW
        One-dimensional Gaussian process with parameters fit to training data.
    """
    with opinf.utils.TimedBlock(
        f"\nfitting GP model for state '{config.DIMFMT(stateindex)}'\n"
    ):
        gp = gpkernels.GP_RBFW(
            config.CONSTANT_VALUE_BOUNDS,
            config.LENGTH_SCALE_BOUNDS,
            config.NOISE_LEVEL_BOUNDS,
            config.N_RESTARTS_OPTIMIZER,
        )
        gp.fit(time_domain_sampled, state_variable_sampled)
        print(gp)

    with opinf.utils.TimedBlock("computing weight matrix", timelimit=600):
        gp.compute_lstsq_matrices(time_domain_training, eta=gp_regularizer)

    return gp


def fit_gaussian_processes(
    time_domain_training: np.ndarray,
    time_domains_sampled: list[np.ndarray],
    snapshots_sampled: np.ndarray,
    gp_regularizer: float = 1e-8,
) -> Iterable[gpkernels.GP_RBFW]:
    """Fit Gaussian Process (GP) regression models to the snapshot data,
    one state variable at a time.

    Parameters
    ----------
    time_domain_training : (m',) ndarray
        Time domain at which to estimate states and time derivatives
        for the parameter estimation.
    time_domains_sampled : list of num_variables (m,) ndarrays
        Time domains corresponding to the available training snapshots,
        one for each variable.
    snapshots_sampled : (num_variables, m) ndarray
        Observed training snapshots.
    """
    return [
        fit_single_gaussian_process(
            stateindex=stateindex,
            time_domain_training=time_domain_training,
            time_domain_sampled=time_domains_sampled[stateindex],
            state_variable_sampled=snapshots_sampled[stateindex],
            gp_regularizer=gp_regularizer,
        )
        for stateindex in range(config.NUMVARS)
    ]
