import equinox as eqx
import jax
from jax import Array, vmap
from jax import numpy as jnp
from jax.typing import ArrayLike

from madnet.architectures import MultiHeadMLP
from madnet.estimands import BaseEstimand

from ._base import BaseEstimator
from ._estimation_utils import Predictions, StandardTheory, StandardTheoryTMLE
from ._loss_utils import ObjectiveOutput, auto_debiased_loss, mean_squared_error


class RieszNet(BaseEstimator):
    """RieszNet.

    RieszNet is an automatic debiasing procedure based on automatically learning the
    Riesz representation of the linear functional.

    Original PyTorch implementation:
    [RieszLearning](https://github.com/victor5as/RieszLearning).

    References
    ----------
    Chernozhukov, V., Newey, W., Quintas-Martinez, V. M., & Syrgkanis, V. (2022).
    RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets
    and Random Forests.
    """

    # Hyperparameters
    rr_weight: float
    target_reg: float
    binary: bool

    # Parameters
    mlp: MultiHeadMLP
    srr_param: Array  # Scharfstein-Rotnitzky-Robins correction

    def __init__(
        self,
        estimand: BaseEstimand,
        covariate_dim: int,
        treatment_dim: int = 1,
        width_size: int = 200,
        shared_depth: int = 3,
        nonshared_depth: int = 2,
        rr_weight: float = 0.1,
        target_reg: float = 1.0,
        binary: bool = True,
        *,
        key: Array,
    ):
        """Initialise RieszNet multilayer perceptron layers.

        Args:
            estimand: Object from the `estimands` module.
            covariate_dim: Covariate dimension.
            width_size: Neurons per layer. Defaults to 200.
            rr_weight: Positive hyperparameter weighting the Riesz representer loss
            component. Defaults to 1.0.
            target_reg: Hyperparameter in $[0,1]$ weighting the ratio between targeted
            regularization and mean squared error. Defaults to 0.1.
            key: A `jax.random.key`. Keyword Only.
        """
        super().__init__(estimand=estimand)
        self.rr_weight = rr_weight
        self.target_reg = target_reg
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
        self.srr_param = jnp.zeros(())
        self.binary = binary

    def forward(self, a: ArrayLike, x: ArrayLike) -> tuple[Array, Array, Array]:
        r"""Three-headed RieszNet architecture.

        See Figure 1 in Chernozhukov et al. (2022). There the Riesz representer
        is denoted $\alpha$.
        """
        if self.binary:
            y0, y1, rr_pred = self.mlp(jnp.hstack((a, x)))
            y_pred = a * y1 + (1 - a) * y0
        else:
            y_pred, rr_pred = self.mlp(jnp.hstack((a, x)))
        y_tmle = y_pred + self.srr_param * rr_pred
        return y_pred, rr_pred, y_tmle

    def predict(self, a: ArrayLike, x: ArrayLike, y: ArrayLike) -> Predictions:
        pred, moment = vmap(self.forward_and_moment)(a, x)
        y_pred, rr_pred, y_tmle_pred = pred
        y_moment, _, y_tmle_moment = moment
        y = jnp.asarray(y)
        return StandardTheoryTMLE(
            standard=StandardTheory(y, y_pred, y_moment, rr_pred),
            tmle=StandardTheory(y, y_tmle_pred, y_tmle_moment, rr_pred),
        )

    @staticmethod
    def _loss(params, static, a: Array, x: Array, y: Array) -> ObjectiveOutput:
        model: RieszNet = eqx.combine(params, static)
        pred, moment = jax.vmap(model.forward_and_moment)(a, x)
        y_pred, rr_pred, y_tmle = pred
        _, rr_moment, _ = moment
        mse = mean_squared_error(y, y_pred)
        rr_loss = auto_debiased_loss(rr_pred, rr_moment)
        tmle_loss = mean_squared_error(y, y_tmle)
        loss = mse + model.rr_weight * rr_loss + model.target_reg * tmle_loss
        aux = {
            "mse": mse,
            "rr_loss": rr_loss,
            "tmle_loss": tmle_loss,
            "srr_param": model.srr_param,
            "rr_pred": jnp.mean(rr_pred),
            "outcome_ipw": jnp.mean(y * rr_pred),
            "gamma_ipw": jnp.mean(a * rr_pred),
        }
        return loss, aux
