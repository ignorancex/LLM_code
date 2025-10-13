import equinox as eqx
import jax
from jax import Array
from jax import numpy as jnp
from jax.typing import ArrayLike

from madnet.architectures import MultiHeadMLP
from madnet.estimands import AverageTreatmentEffect

from ._base import BaseEstimator
from ._estimation_utils import Predictions, StandardTheory, StandardTheoryTMLE
from ._loss_utils import ObjectiveOutput, binary_log_loss_from_rr, mean_squared_error


class DragonNet(BaseEstimator):
    r"""DragonNet.

    Introduced in Shi et al. (2019), DragonNet is a three-headed architecture predicting
    propensity score and conditional outcome.

    Original TensorFlow implementation:
    [dragonnet](https://github.com/claudiashi57/dragonnet).

    DragonNet is only valid for binary treatments $A\in\{0,1\}$.

    References
    ----------
    Shi, C., Blei, D., & Veitch, V. (2019). Adapting Neural Networks for the Estimation
    of Treatment Effects. Advances in Neural Information Processing Systems, 32.
    """

    # Hyperparameters
    ps_weight: float
    target_reg: float

    # Parameters
    mlp: MultiHeadMLP
    srr_param: Array  # Scharfstein-Rotnitzky-Robins correction

    def __init__(
        self,
        covariate_dim: int,
        width_size: int = 200,
        shared_depth: int = 3,
        nonshared_depth: int = 2,
        ps_weight: float = 1.0,
        target_reg: float = 1.0,
        *,
        key: Array,
    ):
        """Initialise DragonNet multilayer perceptron layers.

        Args:
            covariate_dim: Covariate dimension.
            width_size: Neurons per layer. Defaults to 200.
            ps_weight: Positive hyperparameter weighting the propensity score loss
            component. Defaults to 1.0.
            target_reg: Hyperparameter in $[0,1]$ weighting the ratio between targeted
            regularization and mean squared error. Defaults to 0.1.
            key: A `jax.random.key`. Keyword Only.
        """
        ate = AverageTreatmentEffect()
        super().__init__(estimand=ate)
        self.ps_weight = ps_weight
        self.target_reg = target_reg
        self.mlp = MultiHeadMLP(
            in_size=covariate_dim,
            width_size=width_size,
            shared_depth=shared_depth,
            nonshared_depths=(nonshared_depth, nonshared_depth, 0),
            activation=jax.nn.elu,
            key=key,
        )
        self.srr_param = jnp.zeros(())

    def forward(self, a: ArrayLike, x: ArrayLike) -> tuple[Array, Array, Array]:
        """Three-headed DragonNet architecture.

        See Figure 1 in Shi et al. (2019).

        Args:
            a: Binary treatment indicator.
            x: Covariates.
        """
        preds = self.mlp(jnp.asarray(x))
        y0, y1, ps_logits = preds[0], preds[1], preds[2]
        y_pred = jnp.where(a, y1, y0)
        ps_pred = jax.nn.sigmoid(ps_logits)
        # Trick for numerical stability found in:
        # https://github.com/claudiashi57/dragonnet/blob/2e51b2478854df389893bac85430bca5b70f5a26/src/experiment/models.py#L90C9-L90C40
        ps_pred = (ps_pred + 0.01) / 1.02
        rr_pred = jnp.where(a, 1 / ps_pred, -1 / (1 - ps_pred))
        y_tmle_pred = y_pred + self.srr_param * rr_pred
        return y_pred, rr_pred, y_tmle_pred

    def predict(self, a: ArrayLike, x: ArrayLike, y: ArrayLike) -> Predictions:
        pred, moment = jax.vmap(self.forward_and_moment)(a, x)
        y_pred, rr_pred, y_tmle_pred = pred
        y_moment, _, y_tmle_moment = moment
        y = jnp.asarray(y)
        return StandardTheoryTMLE(
            standard=StandardTheory(y, y_pred, y_moment, rr_pred),
            tmle=StandardTheory(y, y_tmle_pred, y_tmle_moment, rr_pred),
        )

    @staticmethod
    def _loss(params, static, a: Array, x: Array, y: Array) -> ObjectiveOutput:
        """See Shi et al. (2019, pp. 3-4)."""
        model: DragonNet = eqx.combine(params, static)
        y_pred, rr_pred, y_tmle = jax.vmap(model.forward)(a, x)
        mse = mean_squared_error(y, y_pred)
        ps_loss = binary_log_loss_from_rr(rr_pred)
        tmle_loss = mean_squared_error(y, y_tmle)
        loss = mse + model.ps_weight * ps_loss + model.target_reg * tmle_loss
        aux = {
            "mse": mse,
            "ps_loss": ps_loss,
            "tmle_loss": tmle_loss,
            "srr_param": model.srr_param,
            "rr_pred": jnp.mean(rr_pred),
            "outcome_ipw": jnp.mean(y * rr_pred),
            "gamma_ipw": jnp.mean(a * rr_pred),
        }
        return loss, aux
