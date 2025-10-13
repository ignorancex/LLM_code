#!/usr/bin/env python3

r"""
Some helper functions for the projection apps.
"""

from typing import List

import torch

from botorch.utils.transforms import unnormalize
from torch import Tensor


def textify_vector(X: Tensor, decimal: int = 10) -> str:
    r"""Turn a Tensor into text.

    E.g. textify_vector(torch.tensor([0.2, 0.4, 0.6])) = "(0.2, 0.4, 0.6)"

    Args:
        X: A (N,)-dimensional Tensor.
        decimal: The number of decimal places.

    Returns:
        The string corresponding to `X`.
    """
    return "(" + ", ".join(str(round(x.item(), decimal)) for x in X) + ")"


def compute_optimal_points(
    length: Tensor,
    ref_point: Tensor,
    weight: Tensor,
    bounds: Tensor,
) -> Tensor:
    r"""Compute the optimal points.

    Args:
        length: A (num_weights, )-dimensional Tensor containing the normalized
            lengths.
        ref_point: A (M, )-dimensional Tensor containing the normalized
            reference point.
        weight: A (num_weights, M)-dimensional Tensor containing the weight vector.
        bounds: A (2, M)-dimensional Tensor containing the bounds.

    Returns:
        A (num_weights, M)-dimensional Tensor containing the projected vector.
    """
    return unnormalize(ref_point + length.unsqueeze(-1) * weight, bounds=bounds)


def compute_projected_vector(
    X: Tensor,
    indices: List[int],
) -> Tensor:
    r"""Compute the projected vector.

    Args:
        X: A (*batch_shape, M)-dimensional Tensor.
        indices: A (P,)-dimensional Tensor containing the indices

    Returns:
        A (*batch_shape, P)-dimensional Tensor containing the projected vector.
    """
    return torch.column_stack([X[..., p] for p in indices])
