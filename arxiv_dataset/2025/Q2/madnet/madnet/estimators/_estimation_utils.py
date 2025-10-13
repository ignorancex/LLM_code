from typing import Literal, NamedTuple

from jax import Array
from jax import numpy as jnp
from jax import scipy as jsp
from jax.typing import ArrayLike


class Estimate(NamedTuple):
    """Estimate data model.

    Asymptotically normal estimator characterised by an estimate and standard error.
    Has methods for computing Wald intervals and hypothesis tests.
    """

    estimate: Array
    standard_error: Array

    def p_value(self, hypothesis: float = 0.0) -> Array:
        """P-value for the null that the estimand has some hypothesised value

        Defaults to 0.
        """
        t = jnp.abs(self.estimate - hypothesis) / self.standard_error
        return 2 * (1 - jsp.stats.norm.cdf(t))

    def confidence_interval(self, width: float = 0.95) -> Array:
        """Confidence interval.

        The default `width=0.95` corresponds to a 95% interval.
        """
        s = jsp.stats.norm.ppf(1 - (1 - width) / 2)
        lower = self.estimate - s * self.standard_error
        upper = self.estimate + s * self.standard_error
        return jnp.array((lower, upper))

    def rescale(self, by: ArrayLike) -> "Estimate":
        """Rescale estimates.

        Useful when undoing normalisation.
        """
        return Estimate(
            estimate=jnp.multiply(by, self.estimate),
            standard_error=jnp.multiply(by, self.standard_error),
        )

    def __repr__(self) -> str:
        return (
            f"estimate: {self.estimate:.4f}, standard error: {self.standard_error:.4f}"
        )


Estimates = dict[str, Estimate]


class Undebiased(NamedTuple):
    y_pred_moment: Array

    @property
    def estimates(self) -> Estimates:
        return {"plug_in": self.plug_in}

    @property
    def plug_in(self) -> Estimate:
        """Sometimes called a naive plug-in estiamtor or direct estimator."""
        estimate, std_err = _mean_and_std_err(self.y_pred_moment)
        return Estimate(estimate, std_err)


class InverseProbabilityWeighting(NamedTuple):
    y: Array
    rr_pred: Array

    @property
    def estimates(self) -> Estimates:
        return {"ipw": self.ipw}

    @property
    def ipw(self) -> Estimate:
        """Corresponds to inverse probability weighting in the average treatment
        effect setting.
        """
        estimate, std_err = _mean_and_std_err(self.y * self.rr_pred)
        return Estimate(estimate, std_err)


class StandardTheory(NamedTuple):
    y: Array
    y_pred: Array
    y_pred_moment: Array
    rr_pred: Array

    @property
    def estimates(self) -> Estimates:
        undebiased = Undebiased(self.y_pred_moment)
        ipw = InverseProbabilityWeighting(self.y, self.rr_pred)
        standard = {"double_robust": self.double_robust}
        return {**undebiased.estimates, **ipw.estimates, **standard}

    @property
    def double_robust(self) -> Estimate:
        """One-step debiasing correction as in the double machine learning work of
        Chernozhukov et al. (2018).

        References
        ----------
        Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.,
        & Robins, J. (2018). Double/debiased machine learning for treatment and structural
        parameters. The Econometrics Journal.
        """
        pseudo_outcome = self.y_pred_moment + self.rr_pred * (self.y - self.y_pred)
        estimate, std_err = _mean_and_std_err(pseudo_outcome)
        return Estimate(estimate, std_err)


class StandardTheoryTMLE(NamedTuple):
    standard: StandardTheory
    tmle: StandardTheory

    @property
    def estimates(self) -> Estimates:
        standard = self.standard.estimates
        tmle = self.tmle.estimates
        return {**standard, **{f"{k}_tmle": v for k, v in tmle.items()}}


class TargetedLearner(NamedTuple):
    """Targeted estimator of van der Laan and Rose (2011).

    References
    ----------
    van der Laan, M. J., & Rose, S. (2011). Targeted Learning. In The Elements of
    Statistical Learning
    """

    y: Array
    y_pred: Array
    y_pred_moment: Array
    rr_pred: Array
    rr_pred_moment: Array

    @property
    def estimates(self) -> Estimates:
        standard = StandardTheory(self.y, self.y_pred, self.y_pred_moment, self.rr_pred)
        tmle = {"tmle": self.tmle}
        return {**standard.estimates, **tmle}

    @property
    def tmle(self) -> Estimate:
        """Targeted estimate.

        Mean of `targeted_y_pred_moment`, where `targeted_y_pred` solves:
            `mean(rr_pred * (y - targeted_y_pred)) = 0`
        for linear parametric submodels of the form:
            `targeted_y_pred = y_pred + coef * rr_pred`.
        """
        coef = jnp.mean((self.y - self.y_pred) * self.rr_pred) / jnp.mean(
            jnp.square(self.rr_pred)
        )
        targeted_y_pred = self.y_pred + coef * self.rr_pred
        targeted_y_pred_moment = self.y_pred_moment + coef * self.rr_pred_moment
        pseudo_outcome = targeted_y_pred_moment + self.rr_pred * (
            self.y - targeted_y_pred
        )
        estimate, std_err = _mean_and_std_err(pseudo_outcome)
        return Estimate(estimate, std_err)

    def tmle_bounded(
        self, family: Literal["gaussian", "binomial"] = "binomial"
    ) -> Estimate:
        """Targeted estimate using a logit link function in the parametric submodel.

        Recomended by van der Laan, M. J., & Rose, S. (2011). When using this
        estimator it is recommended to map the outcome on to the interval [0, 1]
        before fitting outcome prediction models.

        Uses a link function to define the parametric submodel as:
        targeted_y_pred = link(link_inverse(y_pred) + coef * rr_pred)
        """
        # TODO: requires a generalised linear model solver, e.g.
        # from tensorflow_probability.substrates.jax import glm
        # <https://www.tensorflow.org/probability/examples/Generalized_Linear_Models>
        raise NotImplementedError


class MADPredictions(NamedTuple):
    """Moment-constrained automatic debiasing predictions."""

    y_outcome: Array
    gamma_outcome: Array
    psi: Array
    y_perp: Array
    y_perp_moment: Array
    gamma_perp: Array
    gamma_perp_moment: Array
    gamma_moment: Array
    psi: Array

    @property
    def estimates(self) -> Estimates:
        psi = self.psi

        y_pred = self.y_perp + psi * self.gamma_outcome
        y_pred_moment = self.y_perp_moment + psi * self.gamma_moment
        gamma_res = self.gamma_outcome - self.gamma_perp
        gamma_violation = jnp.mean(self.gamma_perp_moment)
        correction = 1 + gamma_violation / (1 - gamma_violation)
        rr_norm_sq = jnp.mean(jnp.square(gamma_res))
        rr_pred = gamma_res / rr_norm_sq / correction
        rr_pred_moment = (
            (self.gamma_moment - self.gamma_perp_moment) / rr_norm_sq / correction
        )
        tmle = TargetedLearner(
            self.y_outcome, y_pred, y_pred_moment, rr_pred, rr_pred_moment
        )
        return {**tmle.estimates}


Predictions = (
    Undebiased
    | InverseProbabilityWeighting
    | StandardTheory
    | StandardTheoryTMLE
    | TargetedLearner
    | MADPredictions
)


def _mean_and_std_err(x: ArrayLike) -> tuple[Array, Array]:
    """Mean and standard error in the mean"""
    n = jnp.asarray(x).shape[0]
    mean = jnp.mean(x)
    mean_sq = jnp.mean(jnp.square(x))
    return mean, jnp.sqrt((mean_sq - mean**2) / n)
