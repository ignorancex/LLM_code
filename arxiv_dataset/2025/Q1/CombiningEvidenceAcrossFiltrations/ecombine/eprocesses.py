"""
e-processes for testing exchangeability
"""


from typing import Tuple
import numpy as np
import pandas as pd
from scipy.special import loggamma

from ecombine.calibrators import adjuster


MAX_LOG_E = np.log(10) * 200


def eprocess_exch_universal(x: np.ndarray):
    """Compute the universal inference e-process for testing exchangeability,
    as defined in Ramdas et al. (IJAR'22) at https://arxiv.org/pdf/2102.00630.pdf

    This e-process allows optional stopping in the data filtration, and
    it is also powerful against first-order Markov alternatives.
    It only accepts binary sequences as inputs.

    Complexity: O(T)
    """
    T = len(x)
    assert T > 1, "requires at least 2 data points"
    assert np.logical_or(x == 0, x == 1).all(), "requires a binary sequence"
    x = np.array(x, dtype=int)

    denom_const = np.log(2) + 4 * loggamma(0.5)

    def _compute_loge(n, n_total):
        """Compute the e-process in logarithm, given all cumulative counts."""
        if n_total[0] < 1 or n_total[1] < 1:
            return 0
        res = (
            loggamma(n[0, 0] + 0.5) + loggamma(n[0, 1] + 0.5) + loggamma(n[1, 0] + 0.5) + loggamma(n[1, 1] + 0.5)
        ) - (
            denom_const + loggamma(n[0, 0] + n[1, 0] + 1) + loggamma(n[0, 1] + n[1, 1] + 1)
        )
        res -= (n_total[1] * np.log(n_total[1] / t) + n_total[0] * np.log(n_total[0] / t))
        return res

    # n[k, j] = #(j -> k)
    n = np.zeros((2, 2), dtype=int)
    n_total = np.zeros(2, dtype=int)
    log_e = np.zeros(T)

    # tally counts & compute e
    n_total[x[0]] += 1
    for t, (x_t, x_tm1) in enumerate(zip(x[1:], x[:-1]), 1):
        n[x_t, x_tm1] += 1
        n_total[x_t] += 1
        log_e[t] = _compute_loge(n, n_total)

    return np.exp(np.minimum(log_e, MAX_LOG_E))


def precompute_conformal_pvalues(
        nc_scores: np.ndarray,
        randomize: bool = True,
        rng: np.random.Generator = None,
):
    """Pre-compute the conformal p-values given nonconformity scores."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    if not randomize:
        raise NotImplementedError("non-randomized conformal p-values are not supported yet")

    T = len(nc_scores)
    n = np.zeros(2, dtype=int)
    u = rng.uniform(size=T)
    p = np.zeros(T)
    for t in range(T):
        x_t = nc_scores[t]
        n[x_t] += 1
        p[t] = ((1 - x_t) * n[1] + u[t] * n[x_t]) / (t + 1)
    return p


def eprocess_exch_conformal(
        x: np.ndarray,
        method: str = "jumper",
        jump: float = 0.01,
        jumper_weights: Tuple[float] = (1/3, 1/3, 1/3),
        rng: np.random.Generator = None,
):
    """Compute the e-process for testing exchangeability based on
    conformal p-values, specifically the Simple Jumper algorithm by Vovk (StatSci'21).

    The nonconformity measure is the identity, following Vovk et al. (COPA'22).

    The default betting strategy is the (simple) `jumper`, whose
    weights are the initial capital of the three "jumpers" (1/3 each by default).
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    T = len(x)

    # conformal p-values with the identity nonconformity measure
    p = precompute_conformal_pvalues(x, rng=rng)

    method = method.lower()
    # simple jumper: three processes
    if method == "jumper":
        assert 0 <= jump <= 1, f"jumping parameter must be within [0, 1], got {jump}"
        assert (np.array(jumper_weights) >= 0).all() and np.allclose(sum(jumper_weights), 1), (
            "jumper weights must be nonnegative and sum to one"
        )
        e = np.zeros(T)
        e_sub = {eps: np.zeros(T) for eps in [-1, 0, 1]}
        e[0] = 1
        for eps, w in zip(e_sub, jumper_weights):
            e_sub[eps][0] = w
        for t in range(1, T):
            for eps in e_sub:
                # jump
                e_sub[eps][t] = (1 - jump) * e_sub[eps][t - 1] + (jump / 3) * e[t - 1]
                # bet
                e_sub[eps][t] *= (1 + eps * (p[t] - 0.5))
            e[t] = e_sub[-1][t] + e_sub[0][t] + e_sub[1][t]
        return np.minimum(e, np.exp(MAX_LOG_E))
    else:
        raise ValueError(f"invalid conformal e-process method {method}"
                         "(valid options are: jumper)")


def eprocess_exch_pairwise():
    raise NotImplementedError


def combined_exch_eprocesses(
        data,
        methods=None,    # 'UI', 'Conformal'
        methods_kwargs=None, 
        needs_adj=None,  # [False, True]
        adjuster_kwargs=None,
        data_col="y",
        rng=None,
        return_df=True,
):
    """Compute the combined e-processes for exchangeability, given data and methods.
    
    The default option computes the universal inference (UI) and conformal e-processes.

    Return a combined data frame with the e-processes and their average.
    """
    if methods is None:
        methods = ["UI", "Conformal"]
    if methods_kwargs is None:
        methods_kwargs = {}
    if needs_adj is None:
        needs_adj = [False, True]
    
    x_df = data[data_col].dropna().astype(int)
    x_np = x_df.values
    eprocesses = {}
    for method, method_kw, need_adj, adj_kw in zip(methods, methods_kwargs, needs_adj, adjuster_kwargs):
        method_kw = dict() if method_kw is None else method_kw
        adj_kw = dict() if adj_kw is None else adj_kw
        if method == "UI":
            e_fn = eprocess_exch_universal
        elif method == "Conformal":
            e_fn = eprocess_exch_conformal
        else:
            raise ValueError(f"Unknown method: {method}")
        try:
            e = e_fn(x_np, rng=rng, **method_kw)
        except:
            e = e_fn(x_np, **method_kw)
        if need_adj:
            e = adjuster(e, **adj_kw)
        eprocesses[method + (" [e-lifted]" if need_adj else "")] = e

    eprocesses["Combined"] = np.mean(list(eprocesses.values()), axis=0)
    if return_df:
        eprocesses = pd.DataFrame(eprocesses)

    return eprocesses, x_df
