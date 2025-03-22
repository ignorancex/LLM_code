#!/usr/bin/env python3

r"""
Batch acquisition functions using the reparameterization trick in combination
with (quasi) Monte-Carlo sampling.

Modified code from BoTorch: https://github.com/pytorch/botorch.

Modifications:
- Added Posterior variance acquisition
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition.multi_objective.analytic import (
    MultiObjectiveAnalyticAcquisitionFunction,
)
from botorch.acquisition.multi_objective.objective import AnalyticMultiOutputObjective
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor


class Uncertainty(MultiObjectiveAnalyticAcquisitionFunction):
    r"""The mean trace of the posterior covariance matrix."""

    def __init__(
        self,
        model: Model,
        objective: Optional[AnalyticMultiOutputObjective] = None,
    ) -> None:
        r"""The mean trace of the posterior covariance matrix.

        Args:
            model: A fitted multi-outcome GP model.
            objective: An `AnalyticMultiOutputObjective`.

        """
        super().__init__(model=model, objective=objective)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the mean trace of the posterior covariance matrix on the
        candidate set X.

        Args:
            X: A `batch_shape x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `batch_shape'`-dim tensor of uncertainty values at the given design
                points `X`.
        """
        posterior = self.objective(self.model.posterior(X))
        average_trace = torch.mean(posterior.variance, dim=-1)
        return torch.sqrt(average_trace).squeeze(-1)
