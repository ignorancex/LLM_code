from math import ceil

import jax
import jax.numpy as jnp
import polars as pl
from jax import Array
from jax.typing import ArrayLike

from madnet.estimators import BaseEstimator, Predictions


def cross_fit_predict(
    estimator: BaseEstimator,
    a: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    folds: ArrayLike | None = None,
    n_folds: int | None = None,
    *,
    key: Array,
    **fit_kwargs,
) -> tuple[Predictions, pl.DataFrame]:
    """Cross fitted predictions.

    When folds and n_folds are both None, then fits without cross fitting.
    Otherwise cross fit according to folds or generate folds if one is not provided.
    """
    a = jnp.asarray(a)
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    if folds is None:
        if isinstance(n_folds, int):
            folds = get_folds(key, n=y.shape[0], n_folds=n_folds)
        else:
            fitted_estimator, logs = estimator.fit(a, x, y, **fit_kwargs, key=key)
            return fitted_estimator.predict(a, x, y), logs
    else:
        folds = jnp.asarray(folds)

    predictions = list()
    logs = pl.DataFrame()
    for fold in jnp.unique(folds):
        key, subkey = jax.random.split(key)
        in_fold = folds == fold
        a_train, a_test = a[~in_fold], a[in_fold]
        x_train, x_test = x[~in_fold], x[in_fold]
        y_train, y_test = y[~in_fold], y[in_fold]
        fitted_estimator, fold_logs = estimator.fit(
            a_train, x_train, y_train, **fit_kwargs, key=subkey
        )
        logs = pl.concat([logs, fold_logs.with_columns(fold=fold)])
        predictions.append(fitted_estimator.predict(a_test, x_test, y_test))

    predictions = jax.tree.map(lambda *xs: jnp.hstack(xs), *predictions)
    return predictions, logs


def get_folds(key: Array, n: int, n_folds: int) -> Array:
    """Generate random folds for cross fitting.

    Args:
        key: The PRNG key for generating random numbers.
        n: The total number of samples.
        n_folds: The number of folds to generate.

    Returns:
        Array: An array of shape (n,) containing the fold indices for each sample.
    """
    return jax.random.permutation(
        key, jnp.tile(jnp.arange(n_folds), ceil(n / n_folds))[:n]
    )
