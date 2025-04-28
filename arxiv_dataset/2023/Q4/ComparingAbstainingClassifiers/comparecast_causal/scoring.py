"""
Score functions for abstaining classifiers
"""

from typing import Union, Dict
import numpy as np
from numpy.typing import ArrayLike

import comparecast as cc
from comparecast.scoring import ScoringRule, ZeroOneScore


def compute_scores(
        predictions: np.ndarray,
        abstentions: np.ndarray,
        labels: np.ndarray,
        scoring_rule: Union[str, ScoringRule] = ZeroOneScore(),
        compute_se: bool = False,
) -> Dict:
    """Evaluate an abstaining classifier's selective score, coverage, and oracle counterfactual score."""
    non_absts = np.logical_not(abstentions)
    score_fn = cc.get_scoring_rule(scoring_rule)

    # The counterfactual score (requires oracle access)
    oracle_cf_score = score_fn(predictions, labels)

    # Baseline 1. selective score + coverage
    selective_score = score_fn(predictions[non_absts], labels[non_absts])
    coverage = non_absts

    # Baseline 2. Condessa et al.'s classification quality score
    # (also requires oracle access)
    if isinstance(score_fn, ZeroOneScore):
        # accurately classified & non-rejected
        a_n = np.sum(predictions[non_absts] == labels[non_absts])
        # misclassified & rejected
        m_r = np.sum(predictions[abstentions] != labels[abstentions])
        condessa_score = (a_n + m_r) / len(labels)
    else:
        condessa_score = None

    if compute_se:
        return {
            "oracle_cf_score": (oracle_cf_score.mean(), oracle_cf_score.std() / np.sqrt(len(oracle_cf_score))),
            "selective_score": (selective_score.mean(), selective_score.std() / np.sqrt(len(selective_score))),
            "coverage": (coverage.mean(), coverage.std() / np.sqrt(len(coverage))),
            "condessa_score": (condessa_score, None),
        }
    else:
        return {
            "oracle_cf_score": oracle_cf_score.mean(),
            "selective_score": selective_score.mean(),
            "condessa_score": condessa_score,
            "coverage": coverage.mean(),
        }


def compute_chow_score(
        predictions: ArrayLike,
        abstentions: ArrayLike,
        labels: ArrayLike,
        scoring_rule: Union[str, ScoringRule] = ZeroOneScore(),
        gamma: float = 0.1,
        reduction: str = "mean",
        **kwargs
) -> float:
    """A generalized Chow's score for abstaining classifiers.

        chow_score((p, a), y) = scoring_rule(p[~a], y[~a]) + gamma * #(~a).
    """
    non_absts = np.logical_not(abstentions)
    selective_score = scoring_rule(predictions[non_absts], labels[non_absts])
    coverage = non_absts.astype(float)
    score = selective_score + gamma * coverage

    reduction = reduction.lower()
    if reduction == "mean":
        return score.mean()
    elif reduction == "sum":
        return score.sum()
    elif reduction == "none":
        return score
    else:
        raise ValueError(f"unrecognized reduction method: {reduction}")

