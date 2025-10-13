from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import jax
from chex import ArrayTree
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

Func = Callable[[ArrayLike, ArrayLike], Array | ArrayTree]


class BaseEstimand(ABC):
    """Base class for estimands.

    Both `moment` and `value_and_moment` should support nested Python containers
    (i.e. pytrees) as outputs. This is useful when `func` is the forward pass of
    a model that returns a tuple of predictions.
    """

    @staticmethod
    @abstractmethod
    def moment(func: Func, a: ArrayLike, x: ArrayLike):
        """Linear moment functional.

        Args:
            a: Treatment.
            x: Covariates.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def value_and_moment(func: Func, a: ArrayLike, x: ArrayLike) -> tuple:
        """More efficient to evaluate together for certain estimands.

        Args:
            a: Treatment.
            x: Covariates.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def treatment_moment(self) -> Array:
        """The expected value of the moment when `func(a, x) = a`.

        This value is used downstream in the estimand-informed estimators.
        """
        raise NotImplementedError


class AverageTreatmentEffect(BaseEstimand):
    """Average Treatment Effect (ATE)."""

    @staticmethod
    def moment(func: Func, a: ArrayLike, x: ArrayLike):
        f0 = func(0, x)
        f1 = func(1, x)
        return jax.tree.map(jnp.subtract, f1, f0)

    @staticmethod
    def value_and_moment(func: Func, a: ArrayLike, x: ArrayLike) -> tuple:
        """Only perform the forward pass once for each binary treatment."""
        f0 = func(0, x)
        f1 = func(1, x)
        value = jax.tree.map(partial(jnp.where, a), f1, f0)
        moment = jax.tree.map(jnp.subtract, f1, f0)
        return value, moment

    @property
    def treatment_moment(self) -> Array:
        return jnp.array(1.0)


class AverageDerivativeEffect(BaseEstimand):
    """Average Derivative Effect (ADE).

    By default, we evaluate the Jacobian using reverse-mode auto-differentiation.
    Note that `jax.jacrev` supports multi-output arrays and pytrees of arrays out
    of the box.
    """

    @staticmethod
    def moment(func: Func, a: ArrayLike, x: ArrayLike):
        a = jnp.asarray(a).astype(float)
        return jax.jacrev(func)(a, x)

    @staticmethod
    def value_and_moment(func: Func, a: ArrayLike, x: ArrayLike) -> tuple:
        """`jax.jacrev` doesn't seem to have a `jax.value_and_grad` equivalent."""
        value = func(a, x)
        a = jnp.asarray(a).astype(float)
        moment = jax.jacrev(func)(a, x)
        return value, moment

    @property
    def treatment_moment(self) -> Array:
        return jnp.array(1.0)
