"""
Helpers for experiments
"""

import numpy as np

from comparecast_causal.scoring import compute_scores
from comparecast_causal.drconfint import DREvalAbst
from comparecast_causal.learners import get_learner


def run_experiment(X, y, predictions, abstentions, predictions_b, abstentions_b,
                   estimator, learner, mixed_estimation, mixed_coef,
                   alpha, scoring_rule, clip_pi, rng, verbose=False):
    """Compute CI and calculate diagnostics."""

    # compute scores
    scores_a = compute_scores(predictions, abstentions, y, scoring_rule)
    scores_b = compute_scores(predictions_b, abstentions_b, y, scoring_rule)

    if verbose:
        print("{:g}% CI for CF score difference [estimator: {:s}, learner: {:s}]".format(
            100 * (1 - alpha), estimator, str(learner),
        ))

    # preprocess predictions if binary
    if len(predictions.shape) == 1:
        predictions = np.array([1 - predictions, predictions]).T
    if len(predictions_b.shape) == 1:
        predictions_b = np.array([1 - predictions_b, predictions_b]).T

    # compute CI
    drci_obj = DREvalAbst(
        inputs=X,
        labels=y,
        predictions=predictions,
        abstentions=abstentions,
        predictions_b=predictions_b,
        abstentions_b=abstentions_b,
        scoring_rule=scoring_rule,
        pi_fn=get_learner("c", **learner) if isinstance(learner, dict) else get_learner("c", learner),
        mu0_fn=get_learner("r", **learner) if isinstance(learner, dict) else get_learner("r", learner),
        clip_pi=clip_pi,
        estimator=estimator,
        mixed_estimation=mixed_estimation,
        mixed_coef=mixed_coef,  # find optimal one
        cross_fit=True,
        alpha=alpha,
        rng=rng,
        verbose=verbose,
    )
    ci = drci_obj.compute_ci()
    oracle_cf_diff = scores_a["oracle_cf_score"] - scores_b["oracle_cf_score"]
    if verbose:
        print("Target CF score difference: {:.5f}".format(oracle_cf_diff))
        print("Cross-fit estimate: {:.5f}".format(drci_obj.estimate.mean()))
        print("Asymptotic CI: ({:.5f}, {:.5f})".format(ci[0], ci[1]))
        print("CI width: {:.5f}".format(ci[1] - ci[0]))
        print("Contains oracle CF score:", ci[0] <= oracle_cf_diff <= ci[1])
        print("Rejection (for H0: diff=0):", (ci[0] > 0) or (ci[1] < 0))
        print("-" * 40)

    # returns a summary dict
    summary = dict(
        alpha=alpha,
        scoring_rule=scoring_rule,
        estimator=estimator,
        learner=learner,
        oracle_cf_diff=oracle_cf_diff,
        estimate=(ci[0] + ci[1]) / 2,
        ci=ci,
        width=ci[1] - ci[0],
        miscovered=(oracle_cf_diff < ci[0]) or (ci[1] < oracle_cf_diff),
        rejection=(ci[0] > 0) or (ci[1] < 0),
    )
    return drci_obj, summary
