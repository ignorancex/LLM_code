#!/usr/bin/env python3

r"""
Utilities for the rocket experiment.
"""

from time import time
from typing import Any, Callable, Dict, Tuple

import numpy as np

import torch
from botorch.acquisition.risk_measures import CVaR, Expectation, WorstCase
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.transforms.input import AppendFeatures
from botorch.test_functions.base import MultiObjectiveTestProblem
from scalarize.models.transforms.outcome import Normalize
from scalarize.robust_experiment_utils import RobustSetUtility
from scalarize.utils.scalarization_functions import (
    LengthScalarization,
    ScalarizationFunction,
)
from scipy.special import gamma
from torch import Tensor


def get_objective_function(
    base_function: MultiObjectiveTestProblem,
    observation_kwargs: Dict[str, Any],
) -> Callable[[Tensor], Tuple[Tensor, Tensor, Tensor]]:
    r"""Initialize the objective function that is used for evaluation.

    Args:
        base_function: The base objective function.
        observation_kwargs: The dictionary containing the variables needed to
            generate the noises.

    Returns:
        The objective function.
    """
    tkwargs = observation_kwargs["tkwargs"]
    delta = observation_kwargs["delta"]
    mean = observation_kwargs["mean"]
    std_dev = observation_kwargs["std_dev"]

    def eval_problem(X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Generates function evaluation.

        Args:
            X: A `N x d`-dim Tensor.

        Returns:
            X: A `n x d`-dim Tensor containing the controllable inputs.
            Y: A `n x M`-dim Tensor containing the outputs.
            Z: A `n x w`-dim Tensor containing the uncontrollable inputs.
        """
        feature_noise = delta * (2 * torch.rand(X.shape, **tkwargs) - 1) + mean
        X_aug = torch.column_stack([X, feature_noise])
        Y = base_function(X_aug)
        observation_noise = torch.randn(Y.shape, **tkwargs) * std_dev
        return X, Y + observation_noise, feature_noise

    return eval_problem


def get_grid(num_dimensions: int, grid_length: int, tkwargs: Dict[str, Any]) -> Tensor:
    r"""Compute an evenly spaced grid.

    Args:
        num_dimensions: The number of dimensions.
        grid_length: The spacing on the grid.
        tkwargs: The tensor dtype to use and device to use.

    Returns:
        A `num_grid_points x num_dimensions`-dim Tensor, where `num_grid_points =
            grid_length ** num_dimensions`.
    """
    u = torch.linspace(0, 1, grid_length, **tkwargs)
    mesh = torch.meshgrid(*[u for _ in range(num_dimensions)], indexing="xy")
    reshaped_mesh = torch.column_stack(
        [m.reshape(grid_length**num_dimensions) for m in mesh]
    )
    return reshaped_mesh


def initialize_set_utility(
    base_function: MultiObjectiveTestProblem,
    data: Dict[str, Any],
) -> RobustSetUtility:
    r"""Initialize the robust utility.

    Args:
        base_function: The multi-objective function.
        data: The initial data set.

    Returns:
        The robust utility.
    """
    tkwargs = data["tkwargs"]
    alpha = data["alpha"]
    weights = data["weights"].to(**tkwargs)
    ref_point = data["ref_point"].to(**tkwargs)
    noise_grid = data["noise_grid"].to(**tkwargs)
    input_transform = AppendFeatures(feature_set=noise_grid).eval()
    outcome_transform = Normalize(bounds=None)
    s_fn = LengthScalarization(ref_points=ref_point, weights=weights)
    if alpha == 0.0:
        robust_objective = Expectation(n_w=len(noise_grid))
    elif alpha == 1.0:
        robust_objective = WorstCase(n_w=len(noise_grid))
    else:
        robust_objective = CVaR(alpha=alpha, n_w=len(noise_grid))

    set_utility = RobustSetUtility(
        eval_problem=base_function,
        scalarization_fn=s_fn,
        outcome_transform=outcome_transform,
        input_transform=input_transform,
        robust_objective=robust_objective,
    )
    return set_utility


def hv_transform(y: Tensor, num_objectives: int):
    r"""The hypervolume transformation function.

    Args:
        y: A `N x 1`-dim Tensor.
        num_objectives: The number of objectives.

    Returns:
        The hypervolume transformed objective.
    """
    hv_constant = torch.pi ** (1 / 2) / (
        2 * gamma(num_objectives / 2 + 1) ** (1 / num_objectives)
    )
    return hv_constant * torch.pow(y, num_objectives)


def get_best_points(
    current_X: Tensor,
    current_Y: Tensor,
    new_X: Tensor,
    values: Tensor,
    indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    r"""Update the best points determined by a scalarization function.

    Args:
        current_X: A `N x d`-dim Tensor of inputs.
        current_Y: A `N x 1`-dim Tensor of outputs.
        new_X: A `N x d`-dim Tensor of prospective inputs.
        values: The new maximums.
        indices: The new arg maximums.

    Returns:
        The tuple of updated points.
    """
    best_inputs = new_X[indices]
    best_values = values

    if current_X is None:
        return best_inputs, best_values
    else:
        all_best_inputs = current_X.clone()

        all_best_values = torch.stack([current_Y, best_values], dim=-1)
        all_best_values, all_best_indices = torch.max(all_best_values, dim=-1)
        all_best_inputs[all_best_indices.bool()] = best_inputs[all_best_indices.bool()]

    return all_best_inputs, all_best_values


def grid_search_optimizer(
    model: GenericDeterministicModel,
    grid: Tensor,
    step: int,
    s_fn: ScalarizationFunction,
    num_points: int,
    bounds: Tensor = None,
    maximize: bool = True,
):
    r"""Perform a scalarization-based grid search to determine the scalarized
    optimum set.

    Args:
        model: The model.
        grid: The `N x d`-dim Tensor containing the grid.
        step: The step size that is used within the grid search algorithm.
        s_fn: The scalarization function that is used to determine optimality.
        bounds: A `2 x d`-dim Tensor containing the input bounds.
        num_points: The number of optimal points to be outputted.
        maximize: If true, we consider a maximization problem.

    Returns:
        A two-element tuple containing

        - A `num_points x d`-dim Tensor containing the collection of optimal inputs.
        - A `num_points x M`-dim Tensor containing the collection of optimal
            objectives.
    """
    all_best_inputs = None
    all_best_values = None

    start_time = time()
    for i in range(0, len(grid), step):
        sample_X = grid[i : i + step]
        sample_Y = model.posterior(sample_X).mean

        values, indices = s_fn(sample_Y).max(dim=0)

        all_best_inputs, all_best_values = get_best_points(
            current_X=all_best_inputs,
            current_Y=all_best_values,
            new_X=sample_X,
            values=values,
            indices=indices,
        )
        print(f"i={i}/{len(grid)}, time={time() - start_time}")
    optimal_inputs = all_best_inputs.unique(dim=0)
    optimal_outputs = model.posterior(optimal_inputs).mean
    print(f"length={len(optimal_inputs)}")
    random_indices = np.random.choice(
        len(optimal_inputs), replace=False, size=num_points
    )
    return optimal_inputs[random_indices], optimal_outputs[random_indices]
