# File: markovianess/ci/conditional_independence_test.py
"""
conditional_independence_test.py

Provides:
1. A helper function `get_markov_violation_score()` that computes a lag-dependent
   Markov violation metric from PCMCI p-values and partial correlations.

2. A class `ConditionalIndependenceTest` that wraps Tigramite's PCMCI procedure
   using partial correlation (ParCorr) to test conditional independence on time-series
   data.

Requires:
- tigramite
- numpy
"""

import numpy as np
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


def get_markov_violation_score(p_matrix, val_matrix, alpha_level=0.05):
    """
    Computes a dimension-aware Markov violation score, with lag-dependent weighting
    for deeper lags. Specifically:

      Score =  [ Sum_{k=2..tau_max}  (k-1) * |val[i,j,k]| * -ln(p[i,j,k])  (for significant links) ]
               ------------------------------------------------------------------------------
                                  (N^2) * ( Sum_{k=2..tau_max}  (k-1) )

    where:
      - p_matrix[i, j, k] is the p-value of j->i at lag k in PCMCI (k > 0 means t-k => t).
      - val_matrix[i, j, k] is the partial correlation for that link.
      - N is the number of variables.
      - A link is considered "significant" if p_matrix[i, j, k] <= alpha_level.
      - We only sum over lags >= 2.

    If tau_max <= 1, returns 0.0 (no "beyond first-order" lags).

    Parameters
    ----------
    p_matrix : np.ndarray
        Shape (N, N, tau_max+1) p-values from PCMCI results.
    val_matrix : np.ndarray
        Shape (N, N, tau_max+1) partial correlation values.
    alpha_level : float, optional
        Significance threshold for p-values (default=0.05).

    Returns
    -------
    float
        Markov violation score (>= 0.0).
    """
    N, _, lag_count = p_matrix.shape
    tau_max = lag_count - 1
    if tau_max <= 1:
        return 0.0

    # Avoid log(0)
    clip = 1e-15
    p_clipped = np.clip(p_matrix, clip, 1.0)

    # Sum of weights for k = 2..tau_max
    weight_sum = sum((k - 1) for k in range(2, lag_count))

    raw_sum = 0.0
    for i in range(N):
        for j in range(N):
            for k in range(2, lag_count):
                if p_clipped[i, j, k] <= alpha_level:
                    w_k = (k - 1)  # Weight for lag k
                    raw_sum += w_k * abs(val_matrix[i, j, k]) * -np.log(p_clipped[i, j, k])

    denom = (N ** 2) * weight_sum
    if denom == 0.0:
        return 0.0
    return raw_sum / denom


class ConditionalIndependenceTest:
    """
    A thin wrapper around PCMCI (from Tigramite), using ParCorr as the conditional
    independence test. Provides a single method `run_pcmci()` to execute the pipeline.
    """

    def __init__(self):
        """
        Constructor: sets up a placeholder to store PCMCI results if desired.
        """
        self.results = None

    def run_pcmci(
        self,
        observations: np.ndarray,
        tau_max: int = 5,
        alpha_level: float = 0.05,
        var_names=None
    ):
        """
        Conducts conditional independence testing via PCMCI on the given time-series data.

        Steps:
          1) Wraps `observations` in a Tigramite DataFrame.
          2) Runs PCMCI with ParCorr up to `tau_max`.
          3) Thresholds edges at `alpha_level` => get_graph_from_pmatrix(...).
          4) Extracts parents with `return_parents_dict(..., include_lagzero_parents=False)`.
          5) Filters to only links that have |lag| >= 2, returning a list of
             (child_var, parent_var, lag, partial_corr).

        Returns
        -------
        dict containing:
          - 'p_matrix' : shape [num_vars, num_vars, tau_max+1]
          - 'val_matrix' : same shape as p_matrix
          - 'lag2_or_more_links' : list of (child_idx, parent_idx, lag, partial_corr)
        """
        if not isinstance(observations, np.ndarray):
            raise TypeError("Observations must be a NumPy array.")
        if observations.ndim != 2:
            raise ValueError("Observations array should be 2D: (time, variables).")

        T, d = observations.shape
        if var_names is None:
            var_names = [f"Var{i}" for i in range(d)]

        # 1) Wrap in DataFrame
        dataframe = pp.DataFrame(data=observations, var_names=var_names)

        # 2) Run PCMCI
        pcmci_obj = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
        results = pcmci_obj.run_pcmci(tau_max=tau_max, pc_alpha=None)
        p_matrix = results["p_matrix"]
        val_matrix = results["val_matrix"]
        self.results = results  # Optionally store for future reference

        # 3) Build graph from p_matrix at alpha_level
        graph = pcmci_obj.get_graph_from_pmatrix(
            p_matrix=p_matrix,
            alpha_level=alpha_level,
            tau_min=0,
            tau_max=tau_max
        )

        # 4) Extract parents, ignoring contemporaneous links
        parents_dict = pcmci_obj.return_parents_dict(
            graph=graph,
            val_matrix=val_matrix,
            include_lagzero_parents=False
        )

        # 5) Filter only |lag| >= 2
        lag2_or_more_links = []
        for child_var, parent_list in parents_dict.items():
            for (parent_var, lag) in parent_list:
                lag_index = -lag  # negative lag => index
                partial_corr = val_matrix[child_var, parent_var, lag_index]
                if abs(lag) >= 2:
                    lag2_or_more_links.append(
                        (child_var, parent_var, lag, partial_corr)
                    )

        return {
            'p_matrix': p_matrix,
            'val_matrix': val_matrix,
            'lag2_or_more_links': lag2_or_more_links,
        }