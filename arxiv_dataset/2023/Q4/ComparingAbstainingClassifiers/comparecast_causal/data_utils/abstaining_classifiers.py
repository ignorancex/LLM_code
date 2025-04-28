"""
Simulated setup for abstaining classifiers
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid


def _process_return_data(X, y, predictions, abstentions, predictions_b=None, abstentions_b=None,
                         return_np_arrays=False, for_plotting=False):
    """A helper for processing return formats for generated data."""

    is_comparison = predictions_b is not None and abstentions_b is not None

    if return_np_arrays:
        if is_comparison:
            return X, y, predictions, abstentions, predictions_b, abstentions_b
        else:
            return X, y, predictions, abstentions

    if for_plotting:
        data = pd.DataFrame({
            r"$X_0$": X[:, 0],
            r"$X_1$": X[:, 1],
            r"$Y$": y,
            "Classifier": "A",
            r"$P$": predictions,
            r"$\hat{Y}$": (predictions >= 0.5).astype(int),
            "Abstain": np.where(abstentions, "Yes", "No"),
        })
        data["Correct"] = np.where(data[r"$Y$"] == data[r"$\hat{Y}$"], "Yes", "No")
        if is_comparison:
            data_b = pd.DataFrame({
                r"$X_0$": X[:, 0],
                r"$X_1$": X[:, 1],
                r"$Y$": y,
                "Classifier": "B",
                r"$P$": predictions_b,
                r"$\hat{Y}$": (predictions_b >= 0.5).astype(int),
                "Abstain": np.where(abstentions_b, "Yes", "No"),
            })
            data_b["Correct"] = np.where(data_b[r"$Y$"] == data_b[r"$\hat{Y}$"], "Yes", "No")
            data = pd.concat([data, data_b], ignore_index=True)
    else:
        data = pd.DataFrame({
            "X0": X[:, 0],
            "X1": X[:, 1],
            "Y": y,
            "Classifier": "A",
            "prediction": predictions,
            "Yhat": (predictions > 0.5).astype(int),
            "abstention": abstentions,
        })
        data["correct"] = data["Y"] == data["Yhat"]
        if is_comparison:
            data["Classifier"] = "A"  # name the previous one as "A"
            data_b = pd.DataFrame({
                "X0": X[:, 0],
                "X1": X[:, 1],
                "Y": y,
                "Classifier": "B",
                "prediction": predictions_b,
                "Yhat": (predictions_b >= 0.5).astype(int),
                "abstention": abstentions_b,
            })
            data_b["correct"] = data_b["Y"] == data_b["Yhat"]
            data = pd.concat([data, data_b], ignore_index=True)
    return data


def _make_clf_binary_mar_linear(X, threshold, epsilon, mu, rng):
    """Make a linear classifier for the binary MAR setup."""
    n = len(X)
    predictions = sigmoid(X[:, 0] + X[:, 1] - 1 - mu)
    # MAR; propensity to abstain is higher near the border
    propensity = np.abs(X[:, 0] + X[:, 1] - 1 - mu)
    propensity = np.where(propensity < threshold, 1 - epsilon, epsilon)
    abstentions = rng.binomial(1, propensity, size=n).astype(bool)
    return predictions, abstentions


def _make_clf_binary_mar_curved(X, threshold, epsilon, mu, rng):
    """Make a curved (biased) classifier for the binary MAR setup."""
    n = len(X)
    predictions = np.clip(0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2) + 0.1 + 0.5 * mu, 0, 1)
    # MAR; propensity to abstain is higher near the border
    propensity = np.abs(X[:, 0] ** 2 + X[:, 1] ** 2 - 0.8 - mu)
    propensity = np.where(propensity < threshold, 1 - epsilon, epsilon)
    abstentions = rng.binomial(1, propensity, size=n).astype(bool)
    return predictions, abstentions


def generate_binary_mar(
        n: int = 500,
        epsilon: float = 0.2,
        threshold: float = 0.2,
        noise_level: float = 0.15,
        is_comparison: bool = False,
        threshold_b_mult: float = 0.8,
        is_power_analysis: bool = False,
        power_mu: float = 0.1,
        return_np_arrays: bool = False,
        for_plotting: bool = False,
        rng: np.random.Generator = np.random.default_rng(),
) -> Union[pd.DataFrame, Tuple]:
    """Generate a synthetic abstaining binary classifier on a simulated 2D dataset.

    Abstentions are "missing at random": they are a function of the inputs.

    True decision boundary is `x0 + x1 = 1`.

    Args:
        n: data size (evaluation set)
        epsilon: positivity level (default: 0.1)
        threshold: a threshold between 0 and 1 for determining
            how far away from the decision boundary to start abstaining more often
        noise_level: label noise on the true model
        is_comparison: whether to return a pair of classifiers for comparison (the other is the optimal classifier)
        threshold_b_mult: width multiplier for the border around B's boundary
        return_np_arrays: return a tuple of `(X, y, predictions, abstentions)` without creating a pandas dataframe
        for_plotting: if True, set column names to mathjax names and print names (for plots)
        rng: np.random.Generator instance
    """

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    p = 2
    X = rng.uniform(size=(n, p))
    # y = (X[:, 0] + X[:, 1] + rng.normal(scale=noise_level, size=n) >= 1).astype(int)
    y_gt = (X[:, 0] + X[:, 1] >= 1).astype(int)
    y = np.where(rng.binomial(1, noise_level, size=n), 1 - y_gt, y_gt)  # label noise

    # power analysis with linear (shifting boundary)
    if is_power_analysis:
        predictions, abstentions = _make_clf_binary_mar_linear(X, threshold, epsilon, 0, rng)
        predictions_b, abstentions_b = _make_clf_binary_mar_linear(X, threshold, epsilon, power_mu, rng)

    # compare linear vs. curved
    elif is_comparison:
        predictions, abstentions = _make_clf_binary_mar_linear(X, threshold, epsilon, 0, rng)
        predictions_b, abstentions_b = _make_clf_binary_mar_curved(X, threshold_b_mult * threshold, epsilon, 0, rng)

    # evaluate curved model
    else:
        predictions, abstentions = _make_clf_binary_mar_curved(X, threshold, epsilon, 0, rng)
        predictions_b, abstentions_b = None, None

    return _process_return_data(X, y, predictions, abstentions, predictions_b, abstentions_b,
                                return_np_arrays=return_np_arrays, for_plotting=for_plotting)


def generate_differently_imbalanced_data(
        n_train: int = 15000,
        n_test: int = 15000,
        n_classes: int = 3,
        n_dim: int = 2,
        base_noise: float = 1.0,
        noise_scales: Tuple[float] = (1.0, 1.0, 1.0),
        subsplits: Tuple[Tuple[str, int]] = (("subsplit_A", 2), ("subsplit_B", 1)),
        drop_minor_label: float = 0.99,
        rng: np.random.Generator = np.random.default_rng(),
):
    """Generate different versions of training data with different imbalances."""
    means = np.array([
        [0, -1],
        [-1, 1],
        [1, 1],
    ])
    covs = np.array([
        base_noise * noise_scale * np.eye(n_dim)
        for noise_scale in noise_scales
    ])  # square covariances for now

    dfs = []
    ns_per_class = {
        "train": n_train // n_classes,
        "test": n_test // n_classes,
    }
    for split, n_per_class in ns_per_class.items():
        X = np.vstack([
            rng.multivariate_normal(means[k], covs[k], size=n_per_class)
            for k in range(n_classes)
        ])
        y = np.vstack([
            k * np.ones(n_per_class, dtype=int)[:, np.newaxis]
            for k in range(n_classes)
        ])  # [0, ..., 0, 1, ..., 1, 2, ..., 2]
        df = pd.DataFrame(
            X,
            columns=[f"x{j}" for j in range(1, n_dim + 1)],
        )
        df["y"] = y
        df["split"] = split

        for subsplit, minor_label in subsplits:
            drop_indices = (y.flatten() == minor_label)
            if drop_indices.any():
                undrop = rng.choice(np.where(drop_indices)[0],
                                    int((1 - drop_minor_label) * n_per_class),
                                    replace=False)
                drop_indices[undrop] = False

            # e.g., 99% of minor labels
            df[subsplit] = np.logical_not(drop_indices)

        dfs.append(df)

    # merge into one data frame
    df = pd.concat(dfs)
    return df, means, covs
