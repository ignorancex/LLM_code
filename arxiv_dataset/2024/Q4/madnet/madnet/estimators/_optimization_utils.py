from collections.abc import Callable
from contextlib import suppress
from functools import partial
from typing import Any, NamedTuple

import equinox as eqx
import jax
import optax
import polars as pl
from chex import ArrayTree
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.typing import ArrayLike
from tqdm import tqdm

from madnet.logging import get_logger

from ._loss_utils import ObjectiveOutput

logger = get_logger(__name__)

ObjectiveFn = Callable[[ArrayTree, Any, Array, Array, Array], ObjectiveOutput]


class Log(NamedTuple):
    epoch: int
    learner: str
    objective: float
    mse: float = jnp.nan
    ps_loss: float = jnp.nan
    tmle_loss: float = jnp.nan
    rr_loss: float = jnp.nan
    srr_param: float = jnp.nan
    mse: float = jnp.nan
    constraint_violation: float = jnp.nan
    lagrange_multiplier: float = jnp.nan
    mse_y: float = jnp.nan
    constraint_violation_y: float = jnp.nan
    lagrange_multiplier_y: float = jnp.nan
    mse_g: float = jnp.nan
    constraint_violation_g: float = jnp.nan
    lagrange_multiplier_g: float = jnp.nan
    estimator: list[str] = []
    estimate: list[float] = []
    standard_error: list[float] = []
    rr_pred: float = jnp.nan
    outcome_ipw: float = jnp.nan
    gamma_ipw: float = jnp.nan


class Fitter(NamedTuple):
    """Fitter with optax and early stopping utils.

    Early stopping adapted from `flax.training.early_stopping`.
    """

    optimizer: str = "adam"
    learning_rate: float = 0.0001
    min_delta: float = 0
    patience: int = 2
    weight_decay: float = 0.001
    best_metric: float = jnp.inf
    patience_count: int = 0
    should_stop: bool = False
    has_improved: bool = False

    def partition(self, model) -> tuple:
        params, static = eqx.partition(model, eqx.is_array)
        return params, static

    @staticmethod
    def combine(params, static):
        return eqx.combine(params, static)

    @property
    def solver(self) -> optax.GradientTransformation:
        if self.optimizer == "adamw":
            return getattr(optax, self.optimizer)(self.learning_rate, self.weight_decay)
        else:
            return getattr(optax, self.optimizer)(self.learning_rate)

    def reset(self) -> "Fitter":
        return self._replace(
            best_metric=jnp.inf, patience_count=0, should_stop=False, has_improved=False
        )

    def update(self, objective: Array, aux: dict) -> "Fitter":
        """Update the state based on objective."""
        del aux
        metric = objective.item()
        if jnp.isinf(self.best_metric) or self.best_metric - metric > self.min_delta:
            return self._replace(
                best_metric=metric, patience_count=0, has_improved=True
            )
        else:
            should_stop = self.patience_count >= self.patience or self.should_stop
            return self._replace(
                patience_count=self.patience_count + 1,
                should_stop=should_stop,
                has_improved=False,
            )


def train_validation_split(
    *arrays, validation_size: float = 0.2, key: Array | None = None
) -> tuple:
    """Splits the input arrays into random train and test subsets.

    Similar in spirit to `sklearn.model_selection.train_test_split`, but with some
    slight differences (see example below).

    Example
    -------

    ```python
    X = jnp.arange(50.0).reshape(5, 10)
    y = jnp.arange(5.0)
    (X_train, y_train), (X_val, y_val) = train_validation_split(X, y, validation_size=0.2)
    ```

    """
    folds = 1 / validation_size
    maybe_shuffle_fn = partial(jr.permutation, key) if key is not None else lambda x: x

    def f(x: Array) -> tuple[Array, Array]:
        splits = jnp.array_split(maybe_shuffle_fn(x), folds)
        return jnp.concat(splits[:-1]), splits[-1]

    train, test = jax.tree.map(lambda *xs: tuple(xs), *jax.tree.map(f, arrays))
    return train, test


def dataloader(
    a: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    batch_size: int | None = None,
    *,
    key: Array,
):
    """Dataloader returning a generator yielding batches of the form $W=(A, X, Y)$.

    If `batch_size` is not specified, the generator yields the entire dataset
    (randomly shuffled according to `key`). The same is true for the case when
    `batch_size` happens to be larger than the size of the dataset.
    """
    n = jnp.asarray(x).shape[0]
    num_batches = jnp.ceil(n / batch_size) if batch_size is not None else 1
    minibatches = jax.tree.map(
        lambda ary: jnp.array_split(jr.permutation(key, ary), num_batches), (a, x, y)
    )
    yield from zip(*minibatches)


def build_step_fn(
    static, objective_fn: ObjectiveFn, solver: optax.GradientTransformation
) -> Callable:
    obj_and_grad_fn = jax.value_and_grad(objective_fn, has_aux=True)

    def step(params, opt_state, batch):
        (obj, _), grads = obj_and_grad_fn(params, static, *batch)
        updates, opt_state = solver.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, obj

    return step


def estimates2dict(model, data) -> dict:
    out = {"estimator": [], "estimate": [], "standard_error": []}
    with suppress(NotImplementedError):
        estimates = model.estimate(*data)
        for k, v in estimates.items():
            out["estimator"].append(k)
            out["estimate"].append(v.estimate)
            out["standard_error"].append(v.standard_error)
    return out


def generic_fit(
    model,
    train_data: tuple[ArrayLike, ArrayLike, ArrayLike],
    validation_data: tuple[ArrayLike, ArrayLike, ArrayLike] | None,
    objective_fn: ObjectiveFn,
    fitter: Fitter,
    num_epochs,
    batch_size: int | None,
    verbose: bool,
    description: str,
    logging_epochs: int | float,
    validation_size: float,
    *,
    key: Array,
):
    """Fit the model to the data."""
    split_key, *load_keys = jr.split(key, num_epochs + 1)

    if validation_data is not None:
        _train_data = train_data
        _validation_data = validation_data
    elif validation_size == 0:
        _train_data = train_data
        _validation_data = train_data
    else:
        _train_data, _validation_data = train_validation_split(
            *train_data, validation_size=validation_size, key=split_key
        )

    solver = fitter.solver
    params, static = fitter.partition(model)
    opt_state = solver.init(params)

    step_fn = build_step_fn(static, objective_fn, solver)

    jitted_step_fn = jax.jit(step_fn)
    jitted_objective_fn = jax.jit(objective_fn, static_argnames=["static"])

    logs = []

    # Main training loop
    pbar = tqdm(range(1, num_epochs + 1), disable=not verbose, leave=False)
    for epoch in pbar:
        for batch in dataloader(*_train_data, batch_size, key=load_keys[epoch - 1]):
            params, opt_state, _ = jitted_step_fn(params, opt_state, batch)

        objective, aux = jitted_objective_fn(params, static, *_validation_data)
        pbar.set_description(f"{description}: {objective.item():.3f}")

        # Maybe log estimates
        log_this_epoch = epoch % logging_epochs == 0
        if log_this_epoch:
            model = fitter.combine(params, static)
            estimates_dict = estimates2dict(model, _train_data)
        else:
            estimates_dict = {}

        logs.append(Log(epoch, description, objective, **aux, **estimates_dict))

        # Check for early stopping
        fitter = fitter.update(objective, aux)
        if fitter.should_stop:
            if verbose:
                logger.info(f"Met early stopping criteria, breaking at epoch {epoch}")
            break

    fitted_model = fitter.combine(params, static)
    logs_df = pl.DataFrame(logs).explode(["estimator", "estimate", "standard_error"])
    return fitted_model, logs_df
