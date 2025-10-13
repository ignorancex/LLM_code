from typing import NamedTuple

import equinox as eqx
import jax
import optax
import polars as pl
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

from madnet.architectures import MultiHeadMLP
from madnet.estimands import BaseEstimand

from ._base import BaseEstimator
from ._estimation_utils import MADPredictions, Predictions
from ._loss_utils import ObjectiveOutput, mean_squared_error
from ._optimization_utils import Fitter, generic_fit


class MADFitter(NamedTuple):
    """Similar to Fitter but adapted for separate psi learning rate."""

    optimizer: str = "adam"
    learning_rate: float = 0.0001
    min_delta: float = 0
    patience: int = 2
    weight_decay: float = 0.001
    psi_optimizer: str = "adam"
    psi_learning_rate: float = 0.01
    best_metric: float = jnp.inf
    patience_count: int = 0
    should_stop: bool = False
    has_improved: bool = False

    def partition(self, model) -> tuple:
        nn_params, static = eqx.partition(model, eqx.is_array)
        params = (nn_params, nn_params.psi)
        return params, static

    @staticmethod
    def combine(params, static):
        nn_params, psi = params
        model = eqx.combine(nn_params, static)
        return eqx.tree_at(lambda t: t.psi, model, psi)

    @property
    def solver(self) -> optax.GradientTransformation:
        if self.optimizer == "adamw":
            nn_solver = getattr(optax, self.optimizer)(
                self.learning_rate, self.weight_decay
            )
        else:
            nn_solver = getattr(optax, self.optimizer)(self.learning_rate)
        psi_solver = getattr(optax, self.psi_optimizer)(self.psi_learning_rate)
        return optax.multi_transform(
            {"nn": nn_solver, "psi": psi_solver}, param_labels=("nn", "psi")
        )

    def reset(self):
        return Fitter.reset(self)

    def update(self, objective: Array, aux: dict):
        return Fitter.update(self, objective, aux)


class MADNet(BaseEstimator):
    """Moment-constrained Automatic Debiasing Network."""

    mlp: MultiHeadMLP
    psi: Array
    lm_y: float
    lm_g: float
    binary: bool

    def __init__(
        self,
        estimand: BaseEstimand,
        covariate_dim: int,
        treatment_dim: int = 1,
        shared_depth: int = 3,
        nonshared_depth: int = 2,
        width_size: int = 200,
        lm_y: float = 0.0,
        lm_g: float = 5.0,
        binary: bool = True,
        *,
        key: Array,
    ):
        """Initialise MADNet.

        Args:
            estimand: Object from the `estimands` module.
            covariate_dim: Covariate dimension.
            width_size: Neurons per layer. Defaults to 200.
            key: A `jax.random.key`. Keyword Only.
        """
        super().__init__(estimand=estimand)
        if binary:
            self.mlp = MultiHeadMLP(
                in_size=covariate_dim + treatment_dim,
                width_size=width_size,
                shared_depth=shared_depth,
                nonshared_depths=(nonshared_depth, nonshared_depth, 0),
                activation=jax.nn.elu,
                key=key,
            )
        else:
            self.mlp = MultiHeadMLP(
                in_size=covariate_dim + treatment_dim,
                width_size=width_size,
                shared_depth=shared_depth,
                nonshared_depths=(nonshared_depth, 0),
                activation=jax.nn.elu,
                key=key,
            )
        self.psi = jnp.zeros(())
        self.lm_y = lm_y
        self.lm_g = lm_g
        self.binary = binary

    def forward(self, a: ArrayLike, x: ArrayLike) -> tuple[Array, Array]:
        if self.binary:
            y0, y1, gamma_pred = self.mlp(jnp.hstack((a, x)))
            outcome_pred = a * y1 + (1 - a) * y0
        else:
            outcome_pred, gamma_pred = self.mlp(jnp.hstack((a, x)))
        return outcome_pred, gamma_pred

    def predict(self, a: ArrayLike, x: ArrayLike, y: ArrayLike) -> Predictions:
        (y_perp, g_perp), (y_moment, g_moment) = jax.vmap(self.forward_and_moment)(a, x)
        return MADPredictions(
            y_outcome=jnp.asarray(y),
            y_perp=y_perp,
            y_perp_moment=y_moment,
            gamma_outcome=jnp.asarray(a),
            gamma_perp=g_perp,
            gamma_perp_moment=g_moment,
            gamma_moment=self.estimand.treatment_moment * jnp.ones_like(a),
            psi=self.psi,
        )

    @staticmethod
    def _loss(params, static, a: Array, x: Array, y: Array) -> ObjectiveOutput:
        nn_params, psi = params
        model: MADNet = eqx.combine(nn_params, static)
        (y_perp, g_perp), (y_moment, g_moment) = jax.vmap(model.forward_and_moment)(
            a, x
        )
        violation_y = jnp.mean(y_moment)
        violation_g = jnp.mean(g_moment)
        y_pred = y_perp + psi * a
        mse_y = mean_squared_error(y, y_pred)
        mse_g = mean_squared_error(a, g_perp)
        loss = (
            mse_y
            + mse_g
            + jnp.abs(model.lm_y * violation_y)
            + jnp.abs(model.lm_g * violation_g)
        )
        gamma_res = a - g_perp
        correction = 1 + violation_g / (1 - violation_g)
        rr_norm_sq = jnp.mean(jnp.square(gamma_res))
        rr_pred = gamma_res / rr_norm_sq / correction
        aux = {
            "mse": loss,
            "mse_y": mse_y,
            "mse_g": mse_g,
            "constraint_violation_y": violation_y,
            "constraint_violation_g": violation_g,
            "rr_pred": jnp.mean(rr_pred),
            "outcome_ipw": jnp.mean(y * rr_pred),
            "gamma_ipw": jnp.mean(a * rr_pred),
        }
        return loss, aux

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
        fitter = MADFitter(
            optimizer,
            learning_rate,
            min_delta,
            patience,
            weight_decay,
            psi_optimizer="adam",
            psi_learning_rate=0.9,
        )

        description = self.__class__.__name__ + description
        return generic_fit(
            model=self,
            train_data=(a, x, y),
            validation_data=validation_data,
            objective_fn=self._loss,
            fitter=fitter,
            num_epochs=num_epochs,
            batch_size=batch_size,
            verbose=verbose,
            description=description,
            logging_epochs=logging_epochs,
            validation_size=validation_size,
            key=key,
        )
