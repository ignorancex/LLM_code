#!/usr/bin/env python3

r"""
Get cake data.
"""

import gc
import os
import time
from math import ceil

import torch

from botorch import fit_gpytorch_mll
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.transforms import normalize, unnormalize

from scalarize.experiment_utils import initialize_model
from scalarize.utils.scalarization_functions import LengthScalarization
from scalarize.utils.scalarization_parameters import UnitVector


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    problem_name = "cake"
    data_path = os.path.join(current_dir, f"data/{problem_name}_data.pt")
    model_data_path = os.path.join(current_dir, f"data/{problem_name}_model_data.pt")
    tkwargs = {"dtype": torch.double, "device": "cpu"}

    # Initial parameters.
    seed = 0
    torch.manual_seed(seed)
    grid_length = 24
    chunk_length = 10**5
    plot_grid_length = 128

    data = torch.load(data_path)
    X = data["X"]
    Y = data["Y"]

    input_dim = X.shape[-1]
    num_objectives = Y.shape[-1]
    input_bounds = torch.row_stack([X.min(dim=0).values, X.max(dim=0).values])
    output_bounds = torch.row_stack([Y.min(dim=0).values, Y.max(dim=0).values])

    # Fit the model.
    model_kwargs = {}
    mll, model = initialize_model(train_x=X, train_y=Y, **model_kwargs)
    fit_gpytorch_mll(mll)

    u = torch.linspace(0, 1, grid_length, **tkwargs)
    input_dim_eff = input_dim - 1
    mesh = torch.meshgrid(*[u for _ in range(input_dim_eff)], indexing="xy")
    reshaped_mesh = torch.column_stack(
        [m.reshape(grid_length**input_dim_eff) for m in mesh]
    )

    # Get simplex points.
    sorted_mesh, _ = torch.sort(reshaped_mesh, dim=-1)
    sorted_mesh = torch.unique(sorted_mesh, dim=0)
    aug_sorted_mesh = torch.cat(
        [
            torch.zeros(len(sorted_mesh), 1, **tkwargs),
            sorted_mesh,
            torch.ones(len(sorted_mesh), 1, **tkwargs),
        ],
        dim=-1,
    )
    simplex_grid = aug_sorted_mesh[..., 1:] - aug_sorted_mesh[..., :-1]

    # Get feasible points.
    flour_condition = torch.logical_and(
        simplex_grid[..., 0:2].sum(dim=-1) >= 0.2,
        simplex_grid[..., 0:2].sum(dim=-1) <= 0.4,
    )
    sugar_condition = torch.logical_and(
        simplex_grid[..., 2] >= 0.15,
        simplex_grid[..., 2] <= 0.35,
    )
    add_condition = simplex_grid[..., 3:6].sum(dim=-1) >= 0.20

    feasibility_mask = torch.logical_and(
        torch.logical_and(flour_condition, sugar_condition), add_condition
    )

    x_grid = simplex_grid[feasibility_mask]
    num_points = len(x_grid)

    model_data = {
        "x_grid": x_grid,
        "tkwargs": tkwargs,
        "grid_length": grid_length,
        "plot_grid_length": plot_grid_length,
        "seed": seed,
    }

    torch.save(model_data, model_data_path)
    ##############################################################################
    start_time = time.time()
    model_mean = torch.zeros(num_points, num_objectives, **tkwargs)
    model_variance = torch.zeros(num_points, num_objectives, **tkwargs)

    index = 0
    iteration = 0
    num_iterations = ceil(num_points / chunk_length)

    start_time = time.time()
    while (index + 1) <= num_points:
        loop_time = time.time()
        next_index = index + chunk_length
        posterior = model.posterior(x_grid[index:next_index])
        model_mean[index:next_index] = posterior.mean
        model_variance[index:next_index] = posterior.variance

        index = next_index
        iteration = iteration + 1

        model_data["mean"] = model_mean
        model_data["variance"] = model_variance

        del posterior
        gc.collect()
        torch.cuda.empty_cache()

        print(
            f"index={iteration}/{num_iterations}, "
            f"loop_time={time.time() - loop_time}, "
            f"time_taken={time.time() - start_time}"
        )

        torch.save(model_data, model_data_path)

    ##############################################################################
    model_data["pareto_mask"] = is_non_dominated(model_mean)

    model_output_bounds = torch.row_stack(
        [model_mean.min(dim=0).values, model_mean.max(dim=0).values]
    )

    torch.save(model_data, model_data_path)

    ##############################################################################
    unit_vector = UnitVector(num_objectives=num_objectives, transform_label="polar")

    utopia = model_mean.max(dim=0).values
    nadir = model_mean.min(dim=0).values
    output_range = utopia - nadir

    normalized_model_mean = normalize(model_mean, model_output_bounds)
    normalized_nadir = normalize(nadir.unsqueeze(0), model_output_bounds)

    u = torch.linspace(0, 1, plot_grid_length, **tkwargs)
    WX, WY = torch.meshgrid(u, u, indexing="xy")
    Z = torch.zeros(plot_grid_length * plot_grid_length, num_objectives, **tkwargs)

    for i in range(plot_grid_length):
        for j in range(plot_grid_length):
            index = i * plot_grid_length + j
            Wij = torch.column_stack([WX[i, j], WY[i, j]])

            weights = unit_vector(Wij)
            s_fn = LengthScalarization(weights=weights, ref_points=normalized_nadir)
            pareto_point = (
                normalized_nadir
                + s_fn(normalized_model_mean).max(dim=0).values.unsqueeze(-1) * weights
            )
            Z[index, ...] = unnormalize(pareto_point, model_output_bounds)

    model_data["nadir"] = nadir
    model_data["utopia"] = utopia
    model_data["mean_surface"] = Z
    torch.save(model_data, model_data_path)
