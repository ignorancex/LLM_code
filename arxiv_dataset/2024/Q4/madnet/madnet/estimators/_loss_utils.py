from typing import Any

from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

Aux = dict[str, Any]
ObjectiveOutput = tuple[Array, Aux]


def mean_squared_error(y: Array, pred: Array) -> Array:
    return jnp.mean(jnp.square(y - pred))


def binary_log_loss(a: Array, pred: Array) -> Array:
    """Log loss for binary classification (aka binary cross entropy)."""
    loss = jnp.where(a, jnp.log(pred), jnp.log(1 - pred))
    return -jnp.mean(loss)


def binary_log_loss_from_rr(rr: Array) -> Array:
    r"""Log loss for binary classification from Riesz representer.

    Uses a neat trick:

    $$-a\log(p)-(1-a)\log(1-p)=\log(|\alpha|)$$

    where $\alpha=\frac{a}{p}-\frac{1-a}{1-p}$.
    """
    return jnp.mean(jnp.log(jnp.abs(rr)))


def auto_debiased_loss(rr: Array, rr_moment: Array) -> Array:
    return jnp.mean(jnp.square(rr) - 2 * rr_moment)


def constrained_lagrangian(
    target: Array,
    pred: Array,
    moment: Array,
    multiplier: Array,
    damping: float = 0.1,
    inequality: bool = True,
    scale: ArrayLike = 1.0,
) -> ObjectiveOutput:
    """Calculates the constrained Lagrangian loss."""
    mse = mean_squared_error(target, pred) / jnp.square(scale)
    violation = jnp.mean(moment) / jnp.square(scale)
    maybe_constrain_fn = jnp.abs if inequality else lambda x: x
    infeasibility = maybe_constrain_fn(multiplier * violation)
    lagrangian = mse + infeasibility + damping * jnp.square(violation)
    aux = {
        "mse": mse,
        "constraint_violation": violation,
        "lagrange_multiplier": multiplier,
    }
    return lagrangian, aux
