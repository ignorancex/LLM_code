#!/usr/bin/env python3

r"""
Compute a Monte Carlo estimate of the contribution values.
"""

from __future__ import annotations

from typing import Tuple

import torch
from scipy.special import binom
from torch import Size, Tensor


def shapley_values(scalarized_Y: Tensor) -> Tensor:
    r"""Computes the Monte Carlo estimate of the Shapley contribution values.

    Args:
        scalarized_Y: A `batch_shape x num_points x num_scalars`-dim tensor of
            the scalarized objective values.

    Returns:
        A `batch_shape x num_points`-dim tensor of the Shapley contribution values.
    """
    tkwargs = {"dtype": scalarized_Y.dtype, "device": scalarized_Y.device}
    num_points = scalarized_Y.shape[-2]

    # Sort the scalarized objectives.
    # `batch_shape x num_points x num_scalar`
    sorted_sY, sorted_indices = torch.sort(scalarized_Y, dim=-2, descending=False)
    sorted_indices_reverse = torch.argsort(sorted_indices, dim=-2, descending=False)

    # Initialize the auxiliary tensor.
    # `batch_shape x num_points x num_scalar x num_points`
    r = torch.zeros(torch.Size([*sorted_sY.shape, num_points]), **tkwargs)
    for i in range(num_points):
        if i == 0:
            # `batch_shape x num_points x num_scalar`
            r[..., 0] = scalarized_Y

            # `batch_shape x num_points x num_scalar`
            idx = torch.zeros(*scalarized_Y.shape, dtype=int, device=tkwargs["device"])
        else:
            # `batch_shape x num_points x num_scalar`
            mask_i = sorted_indices_reverse == (i - 1)
            idx = idx + mask_i

            # Compute the weight.
            c_i = torch.sum(
                torch.tensor(
                    [binom(i - 1, k) / binom(num_points - 1, k + 1) for k in range(i)],
                    **tkwargs,
                )
            )

            # `batch_shape x num_points x num_scalar`
            diff = scalarized_Y - sorted_sY.gather(-2, idx)

            # `batch_shape x num_points x num_scalar`
            r[..., i] = c_i * torch.clamp(diff, 0)

            idx = idx + 1

    # Compute the mean of the weighted sum,
    # and the mean over the scalarization parameters.
    # `batch_shape x num_points`
    shapley_vals = torch.mean(r, dim=(-1, -2))

    return shapley_vals


def maximal_values(scalarized_Y: Tensor) -> Tensor:
    r"""Computes the Monte Carlo estimate of the maximal contribution values.

    NOTE: The scalarized objective values should be non-negative for this to make
    sense.

    Args:
        scalarized_Y: A `batch_shape x num_points x num_scalars`-dim tensor of
            the scalarized objective values.

    Returns:
        A `batch_shape x num_points`-dim tensor of the maximal contribution values.
    """
    max_values = torch.max(scalarized_Y, dim=-2).values.unsqueeze(-2)
    diff = scalarized_Y - torch.max(scalarized_Y, dim=-2).values.unsqueeze(-2)
    diff[diff >= 0] = 1
    diff[diff < 0] = 0
    max_sY = diff * scalarized_Y
    # If there are multiple maximizers then we divide by the cardinality
    max_sY = max_sY / torch.sum(max_sY / max_values, dim=-2, keepdims=True)

    return max_sY.mean(dim=-1)


def _initialize_greedy_algorithm(
    scalarized_Y: Tensor,
) -> Tuple[int, Tensor, Size, Tensor, Tensor, Tensor]:
    r"""Initialize the Monte Carlo estimate of the greedy contribution values.

    Args:
        scalarized_Y: A `batch_shape x num_points x num_scalars`-dim tensor of
            the the scalarized objective values.

    Returns:
        A six-element tuple containing

        - num_points: The number of points.
        - scalarized_Y: A `num_configs x num_points x num_scalars`-dim Tensor
            containing the scalarized objective. This `batch_shape` has been
            flattened.
        - original_shape: The size of the final vector, `batch_shape x num_points`.
        - greedy_values: A `num_configs x num_points`-dim Tensor of zeros.
        - available_mask: A `num_configs x num_points`-dim Tensor of True values.
        - all_indices: A `num_configs x num_points`-dim Tensor of indices.
    """
    num_points = scalarized_Y.shape[-2]
    # `batch_shape x num_points`
    original_shape = scalarized_Y[..., 0].shape
    # flatten the batch shape
    # `num_configs x num_points x num_scalars`
    if scalarized_Y.ndim > 2:
        scalarized_Y = torch.flatten(scalarized_Y, start_dim=0, end_dim=-3)
    else:
        scalarized_Y = scalarized_Y.unsqueeze(0)

    # `num_configs x num_points`
    greedy_values = torch.zeros_like(scalarized_Y[..., 0])
    available_mask = torch.ones_like(greedy_values, dtype=bool)
    all_indices = torch.arange(num_points).expand(available_mask.shape)

    return (
        num_points,
        scalarized_Y,
        original_shape,
        greedy_values,
        available_mask,
        all_indices,
    )


def _batch_argmax(
    utility: Tensor, greedy_values: Tensor, all_indices: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Computes the argmax of a batch of utility values and also returns the
    corresponding mask.

    Args:
        utility: A `batch_shape x (num_points - i) x M`-dim tensor of the scalarized
            objective values.
        greedy_values: A `num_configs x num_points`-dim Tensor containing the greedy
            values.
        all_indices: A `num_configs x (num_points - i)`-dim Tensor of indices.

    Returns:
        A three-element tuple containing

        - values: A `num_configs x (num_points - i)`-dim Tensor containing the utility
            values obtained by selecting the latest point.
        - selected_mask: A `num_configs x (num_points - i)`-dim Tensor of boolean
            values. The True values represent the points that have been selected in
            the current `i`-th round.
        - utility_mask: A `num_configs x num_points`-dim Tensor of boolean values.
            The True values represent the points that have been selected in the
            current `i`-th round.
    """
    values, indices = utility.max(dim=-1)
    # `num_configs x (num_points - i)`
    selected_mask = torch.zeros_like(utility, dtype=bool)
    selected_mask[torch.arange(selected_mask.size(0)), indices] = 1

    # `num_configs x num_points`
    utility_mask = torch.zeros_like(greedy_values, dtype=bool)
    util_idx = all_indices[selected_mask]
    utility_mask[torch.arange(utility_mask.size(0)), util_idx] = 1

    return values, selected_mask, utility_mask


def _update_available_points(
    selected_mask: Tensor,
    available_mask: Tensor,
    scalarized_Y: Tensor,
    all_indices: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Updates the remaining points and indices in the greedy algorithm.

    Args:
        selected_mask: A `num_configs x num_points`-dim Tensor of boolean values. The
            True values represent the points that have been selected in the current
            round.
        available_mask: A `num_configs x num_points`-dim Tensor of boolean values.
            The True values represent the points that were still available.
        scalarized_Y: A `num_configs x num_points x num_scalars`-dim Tensor
            containing the scalarized objective values.

    Returns:
        A three-element tuple containing

        - scalarized_Y: A `num_configs x (num_points - 1) x num_scalars`-dim Tensor
            containing the remaining scalarized objective values.
        - all_indices: A `num_configs x (num_points - 1)`-dim Tensor containing the
            remaining indices.
        - available_mask: A `num_configs x (num_points - 1)`-dim Tensor of boolean
            values. The True values represent the points that are still available.
    """

    available_mask[selected_mask] = False
    # Remove the chosen index from the available indices
    # `num_configs x (num_points - 1) x num_scalars`
    new_shape = torch.Size(
        [scalarized_Y.shape[0], scalarized_Y.shape[1] - 1, scalarized_Y.shape[2]]
    )
    scalarized_Y = scalarized_Y[available_mask].reshape(new_shape)
    all_indices = all_indices[~selected_mask].reshape(new_shape[:-1])
    available_mask = available_mask[available_mask].reshape(new_shape[:-1])

    return scalarized_Y, all_indices, available_mask


def forward_greedy_values(scalarized_Y: Tensor) -> Tensor:
    r"""Computes the Monte Carlo estimate of the forward greedy contribution values.

    NOTE: The scalarized objective values should be non-negative for this to make
    sense.

    Args:
        scalarized_Y: A `batch_shape x num_points x num_scalars`-dim tensor of
            the scalarized objective values.

    Returns:
        A `batch_shape x num_points`-dim tensor of the forward greedy contribution
            values.
    """
    (
        num_points,
        scalarized_Y,
        original_shape,
        greedy_values,
        available_mask,
        all_indices,
    ) = _initialize_greedy_algorithm(scalarized_Y=scalarized_Y)

    max_values = 0
    current_values = 0

    for i in range(num_points):
        if i == 0:
            utility = scalarized_Y.mean(dim=-1)
        else:
            utility = torch.maximum(max_values.unsqueeze(-2), scalarized_Y).mean(dim=-1)

        values, selected_mask, utility_mask = _batch_argmax(
            utility=utility, greedy_values=greedy_values, all_indices=all_indices
        )

        if i == 0:
            greedy_values[utility_mask] = values
            max_values = scalarized_Y[selected_mask]
        else:
            greedy_values[utility_mask] = values - current_values
            max_values = torch.maximum(max_values, scalarized_Y[selected_mask])

        current_values = values
        (scalarized_Y, all_indices, available_mask,) = _update_available_points(
            selected_mask=selected_mask,
            available_mask=available_mask,
            scalarized_Y=scalarized_Y,
            all_indices=all_indices,
        )

    return greedy_values.view(original_shape)


def backward_greedy_values(scalarized_Y: Tensor) -> Tensor:
    r"""Computes the Monte Carlo estimate of the backward greedy contribution values.

    NOTE: The scalarized objective values should be non-negative for this to make
    sense.

    Args:
        scalarized_Y: A `batch_shape x num_points x num_scalars`-dim tensor of
            the scalarized objective values.

    Returns:
        A `batch_shape x num_points`-dim tensor of the backward greedy contribution
            values.
    """
    (
        num_points,
        scalarized_Y,
        original_shape,
        greedy_values,
        available_mask,
        all_indices,
    ) = _initialize_greedy_algorithm(scalarized_Y=scalarized_Y)

    # total utility
    current_values = scalarized_Y.max(dim=-2).values.mean(dim=-1)

    for i in range(num_points):
        if i < num_points - 1:
            # `num_configs x (num_points - i)`
            utility = torch.zeros_like(greedy_values[..., 0 : num_points - i])

            for j in range(num_points - i):
                idx = torch.cat([torch.arange(j), torch.arange(j + 1, num_points - i)])
                # `num_configs x (num_points - i - 1)`
                utility[..., j] = (
                    scalarized_Y[..., idx, :].max(dim=-2).values.mean(dim=-1)
                )

        else:
            # compute the final iteration
            # `num_configs x num_scalars`
            s_vals = scalarized_Y[available_mask]
            # `num_configs x 1`
            utility = s_vals.mean(dim=-1, keepdims=True)

        values, selected_mask, utility_mask = _batch_argmax(
            utility=utility, greedy_values=greedy_values, all_indices=all_indices
        )

        if i < num_points - 1:
            greedy_values[utility_mask] = current_values - values
        else:
            greedy_values[utility_mask] = values

        current_values = values
        (scalarized_Y, all_indices, available_mask,) = _update_available_points(
            selected_mask=selected_mask,
            available_mask=available_mask,
            scalarized_Y=scalarized_Y,
            all_indices=all_indices,
        )

    return greedy_values.view(original_shape)
