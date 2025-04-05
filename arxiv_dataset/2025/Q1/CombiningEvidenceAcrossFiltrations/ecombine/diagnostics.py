"""
diagnostic functions for e-processes (validity & growth)
"""


from typing import Callable, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import trange


from ecombine.calibrators import adjuster


def compute_e_combined(
        data_generator: Callable[[int], np.ndarray],       # accepts T
        eprocess_fn0: Callable[[np.ndarray], np.ndarray],  # accepts x
        eprocess_fn1: Callable[[np.ndarray], np.ndarray],  # accepts x
        lift: Tuple[bool] = (False, True),
        adjuster_kwargs: Dict = None,  # dict(use_maximum=True, kappa=None)
        avg_weight0: float = 0.5,
        n_repeats: int = 100,
        T: int = 10000,
) -> pd.DataFrame:
    """Compute two e-processes across repeated data samples
    and combine them by (possibly) e-lifting and averaging.

    Results are stored in a "tall" data frame.
    """
    if not adjuster_kwargs:
        adjuster_kwargs = {}

    all_x, all_e0, all_e1, all_ec = [], [], [], []
    for _ in trange(n_repeats, desc="compute_e_combined repeated trials"):
        x = data_generator(T)
        e0 = eprocess_fn0(x)
        e1 = eprocess_fn1(x)

        ec0 = adjuster(e0, **adjuster_kwargs) if lift[0] else e0
        ec1 = adjuster(e1, **adjuster_kwargs) if lift[1] else e1
        ec = avg_weight0 * ec0 + (1 - avg_weight0) * ec1

        all_x.append(x)
        all_e0.append(e0)
        all_e1.append(e1)
        all_ec.append(ec)

    all_times = np.tile(np.arange(T), n_repeats)  # [0, 1, 2, ..., T, ..., 0, 1, 2, ..., T]
    all_ids = np.repeat(np.arange(n_repeats), T)  # [0, 0, 0, 1, 1, 1, ..., n, n, n]

    df = pd.DataFrame({
        "Time": all_times,
        "id": all_ids,
        "x": np.concatenate(all_x),
        "e0": np.concatenate(all_e0),
        "e1": np.concatenate(all_e1),
        "ec": np.concatenate(all_ec),
    }).astype({"Time": int, "id": int})
    return df


def compute_e_and_stopping(
        data_generator: Callable[[int], np.ndarray],      # accepts T
        eprocess_fn: Callable[[np.ndarray], np.ndarray],  # accepts x
        stopping_fn: Callable[[np.ndarray], int],         # accepts x
        n_repeats: int = 100,
        T: int = 10000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute e-processes across repeated data samples and apply the stopping function for each e-process.

    Results are stored in a "tall" data frame.
    Additionally returns the list of stopping times and the stopped e-values.
    """

    # for mp
    # def _single_run():
    #     x = data_generator(T)
    #     e = eprocess_fn(x)
    #     tau = stopping_fn(x)
    #     stopped = np.concatenate([np.zeros(tau), np.ones(T - tau)]).astype(bool)
    #     return x, e, stopped

    all_x, all_e, all_stopped = [], [], []  # list of lists
    all_tau, all_e_stopped = [], []         # list of numbers
    for _ in trange(n_repeats, desc="compute_e_and_stopping repeated trials"):
        x = data_generator(T)
        e = eprocess_fn(x)
        tau = stopping_fn(x)            # int
        # stopped is a binary vector indicating whether the process is stopped
        stopped = np.concatenate([np.zeros(tau), np.ones(T - tau)])
        # (it is zero at tau for plotting purposes)
        if tau < T:
            stopped[tau] = 0
        e_stopped = e[min(tau, T - 1)]    # float

        all_x.append(x)
        all_e.append(e)
        all_stopped.append(stopped)
        all_tau.append(tau)
        all_e_stopped.append(e_stopped)

    # all_x, all_e, all_stopped = [np.concatenate(all_res) for all_res in zip(*all_results)]
    all_times = np.tile(np.arange(T), n_repeats)    # [0, 1, 2, ..., T, ..., 0, 1, 2, ..., T]
    all_ids = np.repeat(np.arange(n_repeats), T)    # [0, 0, 0, 1, 1, 1, ..., n, n, n]

    full_df = pd.DataFrame({
        "Time": all_times,
        "id": all_ids,
        "x": np.concatenate(all_x),
        "e": np.concatenate(all_e),
        "stopped": np.concatenate(all_stopped),
    }).astype({"Time": int, "id": int, "stopped": bool})
    estop_df = pd.DataFrame({
        "id": np.arange(n_repeats),
        "tau": all_tau,
        "e_stopped": all_e_stopped,
    }).astype({"id": int, "tau": int})
    return full_df, estop_df
