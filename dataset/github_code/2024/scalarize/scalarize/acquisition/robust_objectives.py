#!/usr/bin/env python3

r"""
Distributionally robust objectives implemented as Monte-Carlo objectives.
"""

from math import ceil
from typing import Callable, List, Optional, Union

import torch
from botorch.acquisition.risk_measures import RiskMeasureMCObjective
from torch import Tensor


class ChiSquare(RiskMeasureMCObjective):
    r"""A worst-case sensitivity approximation to the chi square risk measure:

    `ChiSquare({y_i}, epsilon)
        = mean(y_i, i=1,...,n_w) - sqrt(epsilon * var(y_i, i=1,...,n_w))`.

    Equivalently, this risk measure can be formulated as a worst-case sensitivity
    approximation to the following constrained optimization problem:

    `min_q E_{q(w)}[f(x, w)]` with `Chi2(q, p) <= epsilon`,

    where `q` is a distribution over the `n_w` samples, `p` is a uniform distribution
    over the `n_w` samples and `Chi2` is the Pearson's Chi squared divergence.
    """

    def __init__(
        self,
        n_w: int,
        epsilon: float,
        preprocessing_function: Optional[Callable[[Tensor], Tensor]] = None,
        weights: Optional[Union[List[float], Tensor]] = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            n_w: The size of the `w_set` to calculate the risk measure over.
            epsilon: The radius of the divergence penalty.
            preprocessing_function: A preprocessing function to apply to the samples
                before computing the risk measure. This can be used to scalarize
                multi-output samples before calculating the risk measure.
                For constrained optimization, this should also apply
                feasibility-weighting to samples. Given a `batch x m`-dim
                tensor of samples, this should return a `batch`-dim tensor.
            weights: An optional `m`-dim tensor or list of weights for scalarizing
                multi-output samples before calculating the risk measure.
                Deprecated, use `preprocessing_function` instead.
        """
        super().__init__(
            n_w=n_w, preprocessing_function=preprocessing_function, weights=weights
        )
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        self.epsilon = epsilon

    def forward(self, samples: Tensor, X: Optional[Tensor]) -> Tensor:
        r"""Calculate the approximate chi square risk measure for some given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q`-dim tensor of chi square samples.
        """
        # `sample_shape x batch_shape x q x n_w`
        prepared_samples = self._prepare_samples(samples)
        sample_average = torch.mean(prepared_samples, dim=-1)
        # Do not use Bessel correction.
        sample_variance = torch.var(prepared_samples, dim=-1, unbiased=False)

        return sample_average - torch.sqrt(self.epsilon * sample_variance)


class TotalVariation(RiskMeasureMCObjective):
    r"""The total variation risk measure:

    `TV({y_i}, epsilon)
        = (1 - epsilon) CVaR({y_i}, 1 - epsilon) + epsilon * min(y_i, i=1,...,n_w)`,

    where `CVaR({y_i}, alpha)` computes the conditional value-at-risk---see MCVaR
    description for a more precise description.

    This class also implements the worst-case sensitivity approximation as well:

    `ApproximateTV({y_i}, epsilon)
        = mean(y_i, i=1,...,n_w)
            - epsilon * (max(y_i, i=1,...,n_w) - min(y_i, i=1,...,n_w))`.

    Equivalently, the total variation risk measure can be formulated as the solution
    to the following constrained optimization problem:

    `min_q E_{q(w)}[f(x, w)]` with `TV(q, p) <= epsilon`,

    where `q` is a distribution over the `n_w` samples, `p` is a uniform distribution
    over the `n_w` samples and `TV` is the total variation.
    """

    def __init__(
        self,
        n_w: int,
        epsilon: float,
        approximate: bool = False,
        interpolate: bool = False,
        preprocessing_function: Optional[Callable[[Tensor], Tensor]] = None,
        weights: Optional[Union[List[float], Tensor]] = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            n_w: The size of the `w_set` to calculate the risk measure over.
            epsilon: The radius of the divergence penalty.
            approximate: If True, we use the worst-case sensitivity approximation of
                the total variation risk measure, else we use the exact calculation.
            interpolate: If True, we use a linear interpolation of the discrete
                cumulative distribution function in order to compute the CVaR.
                Otherwise, we consider the standard CVaR of a discrete random
                variable.
            preprocessing_function: A preprocessing function to apply to the samples
                before computing the risk measure. This can be used to scalarize
                multi-output samples before calculating the risk measure.
                For constrained optimization, this should also apply
                feasibility-weighting to samples. Given a `batch x m`-dim
                tensor of samples, this should return a `batch`-dim tensor.
            weights: An optional `m`-dim tensor or list of weights for scalarizing
                multi-output samples before calculating the risk measure.
                Deprecated, use `preprocessing_function` instead.
        """
        super().__init__(
            n_w=n_w, preprocessing_function=preprocessing_function, weights=weights
        )
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if not approximate and epsilon > 1.0:
            raise ValueError(
                "epsilon must be less than or equal to one when using the exact "
                "method."
            )
        self.epsilon = epsilon
        self.approximate = approximate

        if not approximate and epsilon < 1.0:
            # initialize the cvar variables
            proportion = n_w * (1 - epsilon)
            self.k = ceil(proportion)
            # Uniform averaged weights.
            self.q = torch.ones(self.k) / self.k

            # Use interpolated weights.
            if interpolate and self.k - proportion > 0:
                q = torch.ones(self.k) / proportion
                q[-1] = 1 - torch.sum(q[:-1])
                self.q = q

    def forward(self, samples: Tensor, X: Optional[Tensor]) -> Tensor:
        r"""Calculate the total variation risk measure for some given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input. The samples are
                generated from `p`.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q`-dim tensor of total variation samples.
        """
        # # `sample_shape x batch_shape x q x n_w`
        prepared_samples = self._prepare_samples(samples)

        sample_min = torch.min(prepared_samples, dim=-1).values

        if self.approximate:
            sample_mean = torch.mean(prepared_samples, dim=-1)
            sample_max = torch.max(prepared_samples, dim=-1).values
            sample_risk = sample_mean - self.epsilon * (sample_max - sample_min)
        else:
            if self.epsilon != 1.0:
                top_k = torch.topk(
                    prepared_samples,
                    k=self.k,
                    largest=False,
                    dim=-1,
                ).values

                cvar = torch.sum(top_k * self.q.to(top_k), dim=-1)
            else:
                cvar = 0.0

            sample_risk = self.epsilon * sample_min + (1 - self.epsilon) * cvar

        return sample_risk


class Entropic(RiskMeasureMCObjective):
    r"""The entropic risk measure:

    `KL({y_i}, rho, epsilon)
        = - rho * epsilon - rho * log(mean(exp(y_i / rho)), i=1,...,n_w).

    This risk measure is an approximate solution to the following constrained
    optimization problem:

    `min_q E_{q(w)}[f(x, w)]` with `KL(q, p) <= epsilon`,

    where `q` is a distribution over the `n_w` samples, `p` is a uniform distribution
    over the `n_w` samples and `KL is the Kullback-Leibler divergence. The solution
    is exact for the Lagrange parameter that maximizes the risk measure:
     `sup_{rho > 0} KL({y_i}, rho, epsilon)`.
    """

    def __init__(
        self,
        n_w: int,
        epsilon: float,
        rho: float,
        preprocessing_function: Optional[Callable[[Tensor], Tensor]] = None,
        weights: Optional[Union[List[float], Tensor]] = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            n_w: The size of the `w_set` to calculate the risk measure over.
            epsilon: The radius of the divergence penalty.
            rho: A positive Lagrange parameter.
            preprocessing_function: A preprocessing function to apply to the samples
                before computing the risk measure. This can be used to scalarize
                multi-output samples before calculating the risk measure.
                For constrained optimization, this should also apply
                feasibility-weighting to samples. Given a `batch x m`-dim
                tensor of samples, this should return a `batch`-dim tensor.
            weights: An optional `m`-dim tensor or list of weights for scalarizing
                multi-output samples before calculating the risk measure.
                Deprecated, use `preprocessing_function` instead.
        """
        super().__init__(
            n_w=n_w, preprocessing_function=preprocessing_function, weights=weights
        )
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        self.epsilon = epsilon

        if rho <= 0:
            raise ValueError("rho must be positive.")
        self.rho = rho

    def forward(self, samples: Tensor, X: Optional[Tensor]) -> Tensor:
        r"""Calculate the entropic risk measure for some given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input. The samples are
                generated from `p`.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q`-dim tensor of entropic risk samples.
        """
        # `sample_shape x batch_shape x q x n_w`
        prepared_samples = self._prepare_samples(samples)
        transformed_average = torch.log(
            torch.mean(torch.exp(-prepared_samples / self.rho), dim=-1)
        )

        return -self.rho * (self.epsilon + transformed_average)


class MCVaR(RiskMeasureMCObjective):
    r"""The Mean-Conditional Value-at-Risk risk measure.

    The Conditional Value-at-Risk measures the expectation of the worst outcomes
    (small rewards or large losses) with a total probability of `alpha`.

    Assuming `y_i` is ordered from smallest to largest:

    `CVaR({y_i}, alpha) = mean(y_i, i=1,...,n_alpha)`,

    where `n_alpha <= n_w` corresponds to the `alpha` level quantile, that is
    `y_{n_alpha}` is the value for the `alpha`-quantile.

    `MCVaR({y_i}, alpha, beta)
        = (1 / beta) * mean(y_i, i=1,...,n_w) + (1 - 1 / beta) * CVaR({y_i}, alpha)`,

    where `beta` is larger than `1`.

    Note: Due to the use of a discrete `w_set` of samples, the MCVaR calculated here
        are (possibly biased) Monte-Carlo approximations of the true risk measures.
    """

    def __init__(
        self,
        alpha: float,
        n_w: int,
        beta: Optional[float] = None,
        interpolate: bool = False,
        preprocessing_function: Optional[Callable[[Tensor], Tensor]] = None,
        weights: Optional[Union[List[float], Tensor]] = None,
    ) -> None:
        r"""Transform the posterior samples to samples of a risk measure.

        Args:
            n_w: The size of the `w_set` to calculate the risk measure over.
            alpha: The risk level, float in `(0.0, 1.0]`.
            beta: The level of relaxation, float in `[1.0, inf)`. If `beta` is None,
                we compute the CVaR, which is equivalent to setting `beta = inf`. If
                `beta = 1`, we compute the Expectation.
            interpolate: If True, we use a linear interpolation of the discrete
                cumulative distribution function in order to compute the CVaR.
                Otherwise, we consider the standard CVaR of a discrete random
                variable.
            preprocessing_function: A preprocessing function to apply to the samples
                before computing the risk measure. This can be used to scalarize
                multi-output samples before calculating the risk measure.
                For constrained optimization, this should also apply
                feasibility-weighting to samples. Given a `batch x m`-dim
                tensor of samples, this should return a `batch`-dim tensor.
            weights: An optional `m`-dim tensor or list of weights for scalarizing
                multi-output samples before calculating the risk measure.
                Deprecated, use `preprocessing_function` instead.
        """
        super().__init__(
            n_w=n_w, preprocessing_function=preprocessing_function, weights=weights
        )
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0.0, 1.0]")
        if beta is not None and beta < 1:
            raise ValueError("beta must be greater than 1.")
        self.alpha = alpha
        self.beta = beta
        proportion = n_w * alpha
        self.k = ceil(proportion)
        # Uniform averaged weights.
        self.q = torch.ones(self.k) / self.k

        # Use interpolated weights.
        if interpolate and self.k - proportion > 0:
            q = torch.ones(self.k) / proportion
            q[-1] = 1 - torch.sum(q[:-1])
            self.q = q

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Calculate the MCVaR for some given samples.

        Args:
            samples: A `sample_shape x batch_shape x (q * n_w) x m`-dim tensor of
                posterior samples. The q-batches should be ordered so that each
                `n_w` block of samples correspond to the same input.
            X: A `batch_shape x q x d`-dim tensor of inputs. Ignored.

        Returns:
            A `sample_shape x batch_shape x q`-dim tensor of MCVaR samples.
        """
        # `sample_shape x batch_shape x q x n_w`
        prepared_samples = self._prepare_samples(samples)

        top_k = torch.topk(
            prepared_samples,
            k=self.k,
            largest=False,
            dim=-1,
        ).values

        cvar = torch.sum(top_k * self.q.to(top_k), dim=-1)

        mean_cvar = cvar
        if self.beta is not None:
            mean = torch.mean(prepared_samples, dim=-1)
            beta_inverse = 1 / self.beta
            mean_cvar = beta_inverse * mean + (1 - beta_inverse) * mean_cvar

        return mean_cvar
