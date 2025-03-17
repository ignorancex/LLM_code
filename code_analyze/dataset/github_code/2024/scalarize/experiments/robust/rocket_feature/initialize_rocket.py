#!/usr/bin/env python3

r"""
Initialize the rocket experiment.
"""

import os

import numpy as np
import torch

from rocket_utils import get_grid, get_objective_function

from scalarize.experiment_utils import get_problem_bounds
from scalarize.test_functions.multi_objective import RocketInjector
from scalarize.utils.scalarization_parameters import UnitVector

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "data/initial_rocket_data.pt")

    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)

    dtype = torch.double
    device = "cpu"
    tkwargs = {"dtype": dtype, "device": device}

    alpha = 0.9
    mean = 0.5
    delta = 0.5
    num_initial_points = 5

    base_function = RocketInjector(negate=True)
    input_dim = base_function.dim
    input_bounds = base_function.bounds.to(**tkwargs)
    controllable_bounds = input_bounds[..., 0:2]

    num_objectives = base_function.num_objectives
    output_bounds = get_problem_bounds(name="rocket", tkwargs=tkwargs)

    std_dev = 0.01 * torch.sqrt(output_bounds[1] - output_bounds[0])

    # Define the function evaluation.
    observation_kwargs = {
        "delta": delta,
        "mean": mean,
        "std_dev": std_dev,
        "tkwargs": tkwargs,
    }

    eval_problem = get_objective_function(
        base_function=base_function, observation_kwargs=observation_kwargs
    )

    # Input grid.
    input_grid_length = 51
    controllable_dim = 2
    input_grid = get_grid(
        num_dimensions=controllable_dim,
        grid_length=input_grid_length,
        tkwargs=tkwargs,
    )

    # Noise grid.
    noise_grid_length = 21
    uncertainty_dim = 2
    noise_grid = get_grid(
        num_dimensions=uncertainty_dim,
        grid_length=noise_grid_length,
        tkwargs=tkwargs,
    )
    noise_grid = delta * (2 * noise_grid - 1) + mean

    # Utility function.
    weights_grid_length = 25
    unit_vector = UnitVector(
        num_objectives=num_objectives, transform_label="erf_normalize"
    )
    weights = unit_vector(
        get_grid(
            num_dimensions=num_objectives,
            grid_length=weights_grid_length,
            tkwargs=tkwargs,
        )
    )

    # Whole input grid.
    one = torch.ones_like(noise_grid)
    whole_input_grid = [
        torch.column_stack([input_grid[t : t + 1] * one, noise_grid])
        for t in range(len(input_grid))
    ]
    whole_input_grid = torch.row_stack(whole_input_grid)

    initial_indices = np.random.choice(
        len(input_grid), replace=False, size=num_initial_points
    )

    X_initial = input_grid[initial_indices]
    X, Y, X_env = eval_problem(X_initial)

    ref_point = base_function.ref_point.to(**tkwargs)

    data = {
        "problem": "rocket",
        "seed": seed,
        "tkwargs": tkwargs,
        "num_initial_points": num_initial_points,
        "input_dim": input_dim,
        "controllable_dim": controllable_dim,
        "uncertainty_dim": uncertainty_dim,
        "num_objectives": num_objectives,
        "input_bounds": input_bounds,
        "controllable_bounds": controllable_bounds,
        "output_bounds": output_bounds,
        "std_dev": std_dev,
        "mean": mean,
        "delta": delta,
        "alpha": alpha,
        "X": X,
        "Y": Y,
        "X_env": X_env,
        "weights_grid_length": weights_grid_length,
        "weights": weights,
        "ref_point": ref_point,
        "input_grid_length": input_grid_length,
        "input_grid": input_grid,
        "noise_grid_length": noise_grid_length,
        "noise_grid": noise_grid,
        "whole_input_grid": whole_input_grid,
    }

    torch.save(data, output_path)
