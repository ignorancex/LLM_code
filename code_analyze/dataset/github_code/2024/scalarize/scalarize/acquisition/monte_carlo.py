#!/usr/bin/env python3

r"""
Batch acquisition functions using the reparameterization trick in combination
with (quasi) Monte-Carlo sampling.

Modified code from BoTorch: https://github.com/pytorch/botorch.

Modifications:
- Modified `qNoisyExpectedImprovement` to incorporate any pending points into
baseline points.
"""

from __future__ import annotations

import warnings

from copy import deepcopy
from typing import Any, Optional

import torch
from botorch.acquisition.cached_cholesky import CachedCholeskyMCAcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.acquisition.utils import prune_inferior_points
from botorch.exceptions.warnings import BotorchWarning
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor


class qNoisyExpectedImprovement(
    MCAcquisitionFunction, CachedCholeskyMCAcquisitionFunction
):
    r"""MC-based batch Noisy Expected Improvement.
    This function does not assume a `best_f` is known (which would require
    noiseless observations). Instead, it uses samples from the joint posterior
    over the `q` test points and previously observed points. The improvement
    over previously observed points is computed for each sample and averaged.
    `qNEI(X) = E(max(max Y - max Y_baseline, 0))`, where
    `(Y, Y_baseline) ~ f((X, X_baseline)), X = (x_1,...,x_q)`
    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qNEI = qNoisyExpectedImprovement(model, train_X, sampler)
        >>> qnei = qNEI(test_X)

    Modifications:
    1) We incorporated `X_pending` into the baseline points.
    2) We made adding baseline points optional.
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Optional[Tensor] = None,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        prune_baseline: bool = False,
        cache_root: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""q-Noisy Expected Improvement.
        Args:
            model: A fitted model.
            X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
                that have already been observed. These points are considered as
                the potential best design point.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated. Concatenated into `X` upon
                forward call. Copied and set to have no gradient.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the best point. This can significantly
                improve performance and is generally recommended. In order to
                customize pruning parameters, instead manually call
                `botorch.acquisition.utils.prune_inferior_points` on `X_baseline`
                before instantiating the acquisition function.
            cache_root: A boolean indicating whether to cache the root
                decomposition over `X_baseline` and use low-rank updates.

        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=None,
        )
        self._setup(model=model, cache_root=cache_root)
        if prune_baseline:
            X_baseline = prune_inferior_points(
                model=model,
                X=X_baseline,
                objective=objective,
                posterior_transform=posterior_transform,
                marginalize_dim=kwargs.get("marginalize_dim"),
            )
        self.register_buffer("_X_baseline", X_baseline)
        self.register_buffer("_X_baseline_and_pending", X_baseline)
        self.set_X_pending(X_pending)

    @property
    def X_baseline(self) -> Tensor:
        r"""Return X_baseline augmented with pending points."""
        return self._X_baseline_and_pending

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Informs the acquisition function about pending design points.

        Args:
            X_pending: `batch_shape x num_pending x d` Tensor with `num_pending`
            `d`-dim design points that have been submitted for evaluation but have
            not yet been evaluated.
        """
        if X_pending is None:
            self.X_pending = X_pending
            if hasattr(self, "_X_baseline"):
                X_baseline = self._X_baseline
            else:
                X_baseline = None
        else:
            if X_pending.requires_grad:
                warnings.warn(
                    "Pending points require a gradient but the acquisition function"
                    " will not provide a gradient to these points.",
                    BotorchWarning,
                )
            X_pending = X_pending.detach().clone()
            if self._X_baseline is not None:
                X_baseline = torch.cat([self._X_baseline, X_pending], dim=-2)
                num_base = self._X_baseline.shape[0]
            else:
                X_baseline = X_pending
                num_base = 0

            # Number of new points: (num_base + num_pending) - num_base.
            num_pending = X_baseline.shape[0] - num_base
            if num_pending > 0:
                self.register_buffer("_X_baseline_and_pending", X_baseline)
                # Set to None so that pending points are not concatenated in forward.
                self.X_pending = None

        # Cache the baseline samples, maximum values and the root decomposition.
        if X_baseline is not None and self._cache_root:
            self.q_in = -1
            # set baseline samples
            with torch.no_grad():
                posterior = self.model.posterior(
                    X_baseline, posterior_transform=self.posterior_transform
                )
                # Note: The root decomposition is cached in two different places. It
                # may be confusing to have two different caches, but this is not
                # trivial to change since each is needed for a different reason:
                # - LinearOperator caching to `posterior.mvn` allows for reuse within
                #  this function, which may be helpful if the same root decomposition
                #  is produced by the calls to `self.base_sampler` and
                #  `self._cache_root_decomposition`.
                # - self._baseline_L allows a root decomposition to be persisted
                #  outside this method.
                baseline_samples = self.get_posterior_samples(posterior)
            # We make a copy here because we will write an attribute `base_samples`
            # to `self.base_sampler.base_samples`, and we don't want to mutate
            # `self.sampler`.
            self.base_sampler = deepcopy(self.sampler)

            self.register_buffer("baseline_samples", baseline_samples)
            baseline_obj = self.objective(baseline_samples, X=X_baseline)
            self.register_buffer(
                "baseline_obj_max_values", baseline_obj.max(dim=-1).values
            )
            self._baseline_L = self._compute_root_decomposition(posterior=posterior)

    def _forward_cached(self, posterior: Posterior, X_full: Tensor, q: int) -> Tensor:
        r"""Compute difference objective using cached root decomposition.
        Args:
            posterior: The posterior.
            X_full: A `batch_shape x n + q x d`-dim tensor of inputs
            q: The batch size.
        Returns:
            A `sample_shape x batch_shape`-dim tensor containing the
                difference in objective under each MC sample.
        """
        # handle one-to-many input transforms
        n_w = posterior._extended_shape()[-2] // X_full.shape[-2]
        q_in = q * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        new_samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
        new_obj = self.objective(new_samples, X=X_full[..., -q:, :])
        new_obj_max_values = new_obj.max(dim=-1).values
        n_sample_dims = len(self.base_sampler.sample_shape)
        view_shape = torch.Size(
            [
                *self.baseline_obj_max_values.shape[:n_sample_dims],
                *(1,) * (new_obj_max_values.ndim - self.baseline_obj_max_values.ndim),
                *self.baseline_obj_max_values.shape[n_sample_dims:],
            ]
        )
        return new_obj_max_values - self.baseline_obj_max_values.view(view_shape)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qNoisyExpectedImprovement on the candidate set `X`.
        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.
        Returns:
            A `batch_shape'`-dim Tensor of Noisy Expected Improvement values at the
            given design points `X`, where `batch_shape'` is the broadcasted batch shape
            of model and input `X`.
        """
        q = X.shape[-2]
        if self.X_baseline is not None:
            X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        else:
            X_full = X
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = self.model.posterior(
            X_full, posterior_transform=self.posterior_transform
        )
        if self._cache_root:
            diffs = self._forward_cached(posterior=posterior, X_full=X_full, q=q)
        else:
            samples = self.get_posterior_samples(posterior)
            obj = self.objective(samples, X=X_full)
            if obj.shape[-1] == 1:
                diffs = obj[..., -1]
            else:
                diffs = (
                    obj[..., -q:].max(dim=-1).values - obj[..., :-q].max(dim=-1).values
                )

        return diffs.clamp_min(0).mean(dim=0)
