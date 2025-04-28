from abc import abstractmethod

import equinox as eqx
import jax
import polars as pl
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

from madnet.estimands import BaseEstimand

from ._estimation_utils import Estimates, Predictions
from ._optimization_utils import Fitter, generic_fit


class BaseEstimator(eqx.Module):
    """Base class for all estimators.

    Descendents must implement `forward` and `predict` methods.
    """

    estimand: BaseEstimand

    @abstractmethod
    def forward(self, a: ArrayLike, x: ArrayLike):
        """Defines the forward pass.

        Args:
            a: Treatment.
            x: Covariates.
        """
        raise NotImplementedError("No regression implemented for base learner.")

    def moment(self, a: ArrayLike, x: ArrayLike):
        r"""The moment functional of the forward pass.

        Args:
            a: Treatment.
            x: Covariates.
        """
        return self.estimand.moment(self.forward, a, x)

    def forward_and_moment(self, a: ArrayLike, x: ArrayLike):
        r"""Evaluating the forward pass and moment together can be more efficient.

        Args:
            a: Treatment.
            x: Covariates.
        """
        return self.estimand.value_and_moment(self.forward, a, x)

    def fit(
        self,
        a: ArrayLike,
        x: ArrayLike,
        y: ArrayLike,
        validation_data: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
        *,
        num_epochs: int = 100,
        batch_size: int | None = 64,
        verbose: bool = True,
        logging_epochs: int | float = float("inf"),
        description: str = "",
        optimizer: str = "adam",
        learning_rate: float = 0.0001,
        min_delta: float = 1e-04,
        patience: int = 2,
        validation_size: float = 0.2,
        weight_decay: float = 0.001,
        key: Array,
    ) -> tuple["BaseEstimator", pl.DataFrame]:
        """Fit the model to the data.

        If `validation_data` is present, early stopping and estimate logging will be
        performed on the validation set rather than the training set.

        Args:
            a: Treatment.
            x: Covariates.
            y: Outcome.
            validation_data: An optional tuple containing validation data in the form
            (a, x, y).
            num_epochs: The maximum number of training epochs.
            batch_size: The batch size for training, or None for full batch training.
            verbose: Whether to display progress information during training.
            logging_epochs: The number of epochs between logging estimates. Defaults to
            no logging.
            description: A description for the progress bar.
            optimizer : The name of the optax optimizer.
            learning_rate : The learning rate for the optimizer.
            min_delta: Minimum delta between loss updates to be considered an improvement.
            patience: Number of steps of no improvement before early stopping.

        Returns:
            fitted_model: The fitted model.
            logs: A polars DataFrame containing the training or validation logs.
        """
        fitter_args = (optimizer, learning_rate, min_delta, patience, weight_decay)
        if objective_fn := getattr(self, "_loss", None):
            fitter = Fitter(*fitter_args)
        else:
            raise NotImplementedError

        description = self.__class__.__name__ + description
        return generic_fit(
            model=self,
            train_data=(a, x, y),
            validation_data=validation_data,
            objective_fn=objective_fn,
            fitter=fitter,
            num_epochs=num_epochs,
            batch_size=batch_size,
            verbose=verbose,
            description=description,
            logging_epochs=logging_epochs,
            validation_size=validation_size,
            key=key,
        )

    @abstractmethod
    def predict(self, a: ArrayLike, x: ArrayLike, y: ArrayLike) -> Predictions:
        raise NotImplementedError("No predict implemented for base learner.")

    @eqx.filter_jit
    def estimate(self, a: ArrayLike, x: ArrayLike, y: ArrayLike) -> Estimates:
        predictions = self.predict(a, x, y)
        return predictions.estimates

    @property
    def num_params(self) -> int:
        """Returns the number of parameters in the model."""
        params, _ = eqx.partition(self, eqx.is_array)
        return sum(jax.tree.map(jnp.size, jax.tree.leaves(params)))
