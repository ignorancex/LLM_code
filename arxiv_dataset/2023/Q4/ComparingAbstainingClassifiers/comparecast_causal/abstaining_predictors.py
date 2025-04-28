"""
Abstaining predictors based on existing models
"""

import numpy as np
import pandas as pd


def compute_confidence(pred_probs: np.ndarray, metric: str = "sr"):
    """Compute a confidence metric of probabilistic predictions.

    Options are: softmax response (sr), gini impurity (gini), and base-2 entropy (entropy).
    """
    if len(pred_probs.shape) == 1:
        pred_probs = pred_probs[np.newaxis, :]

    metric = metric.lower()
    if metric == "sr":
        return np.max(pred_probs, axis=-1)
    elif metric == "impurity":
        return np.sum(pred_probs ** 2, axis=-1)
    elif metric == "entropy":
        return 1 + (pred_probs * np.log2(np.maximum(1e-8, pred_probs))).sum(axis=-1)
    else:
        raise ValueError(f"confidence metric must be either 'sr', 'impurity', or 'entropy'")


def predict_or_abstain(
        clf_or_preds,
        X: np.ndarray = None,
        confidence_metric: str = "sr",
        threshold=0.7,
        stochastic=True,
        eps=0.001,  # minimum/maximum confidence
        rng=np.random.default_rng(),
):
    """Predict or abstain based on a scikit-learn predictor or a set of base predictions and
    a confidence metric (deterministically or stochastically)."""
    if isinstance(clf_or_preds, np.ndarray):
        predictions = clf_or_preds
    else:
        clf = clf_or_preds
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)
        predictions = clf.predict_proba(X_np)
    pred_labels = np.argmax(predictions, axis=1)
    confidence = np.clip(compute_confidence(predictions, metric=confidence_metric), eps, 1 - eps)

    if stochastic:
        rng = rng if rng is not None else np.random.default_rng()
        abstentions = rng.binomial(1, 1 - confidence).astype(bool)
    else:
        abstentions = confidence < threshold
    return predictions, abstentions, confidence, pred_labels



