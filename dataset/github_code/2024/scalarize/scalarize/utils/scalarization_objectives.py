#!/usr/bin/env python3

r"""
Helper utilities for constructing the scalarization-based objectives.
"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.transforms.outcome import OutcomeTransform
from torch import Tensor

from scalarize.models.transforms.outcome import Normalize
from scalarize.utils.scalarization_functions import ScalarizationFunction


def get_scalarized_samples(
    Y: Tensor,
    scalarization_fn: ScalarizationFunction,
    outcome_transform: OutcomeTransform = None,
) -> Tensor:
    r"""Compute the scalarized objective samples.

    Args:
        Y: A `num_samples x num_points x M`-dim Tensor of objective values. Note that
            we can also pass a `num_points x M`-dim Tensor. This will be expanded
            to a `1 x num_points x M`-dim Tensor.
        scalarization_fn: A scalarization function defined for `M` dimensional
            outputs. The total number of scalarization parameter configurations is
            denoted by `num_scalars`.
        outcome_transform: An outcome transform, defaults to the identity.

    Returns:
        A `num_points x (num_samples x num_scalars)`-dim Tensor containing the
            scalarized samples.
    """
    if outcome_transform is None:
        # Identity transform.
        outcome_transform = Normalize()

    # `num_samples x num_points x M`
    Z = Y.unsqueeze(0) if Y.ndim == 2 else Y

    # `num_samples x num_points x M`
    transformed_samples, _ = outcome_transform(Z)

    # `num_points x (num_samples x num_scalars)`
    scalarized_samples = torch.flatten(
        scalarization_fn(transformed_samples).movedim(0, -2), start_dim=-2, end_dim=-1
    )

    return scalarized_samples


def get_utility_mcobjective(
    scalarization_fn: ScalarizationFunction,
    outcome_transform: OutcomeTransform = None,
) -> GenericMCObjective:
    r"""Computes the Monte Carlo set utility objective.

    Args:
        scalarization_fn: A scalarization function.
        outcome_transform: An outcome transform which will be applied to the
            objective before applying the scalarization function.

    Returns:
        The Monte Carlo objective.
    """

    def scalarized_objectives(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # `Y` has shape `num_mc_samples x batch_shape x m`
        # `s_obj` has shape `batch_shape x (num_mc_samples x num_scalars)`
        s_obj = get_scalarized_samples(
            Y=Y,
            scalarization_fn=scalarization_fn,
            outcome_transform=outcome_transform,
        )
        # `s_obj` has shape `(num_mc_samples x num_scalars) x batch_shape`
        return s_obj.movedim(-1, 0)

    return GenericMCObjective(scalarized_objectives)
