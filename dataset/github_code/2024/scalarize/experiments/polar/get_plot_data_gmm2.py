#!/usr/bin/env python3

r"""
Extract the necessary model data from the GMM2 problem for plotting.
"""

import os

import torch

from botorch.test_functions.multi_objective import GMM
from botorch.utils.multi_objective import is_non_dominated

from scalarize.utils.scalarization_functions import LengthScalarization

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    problem_name = "gmm2"
    data_path = os.path.join(current_dir, f"data/{problem_name}_data.pt")
    length_path = os.path.join(current_dir, f"data/{problem_name}_lengths.pt")
    plot_data_path = os.path.join(current_dir, f"data/{problem_name}_plot_data.pt")

    # Load the data.
    data = torch.load(data_path)
    X = data["X"]
    Y = data["Y"]
    grid_length = data["grid_length"]
    ref_point = data["ref_point"]
    tkwargs = data["tkwargs"]

    # Load the lengths.
    length_data = torch.load(length_path)
    pareto_lengths = length_data["pareto_lengths"]
    model_lengths = length_data["lengths"]
    weights = length_data["weights"]
    scalarization_fn = LengthScalarization(weights=weights, ref_points=ref_point)

    # Compute the objective values.
    num_objectives = 2
    problem = GMM(num_objectives=num_objectives, negate=True)

    u = torch.linspace(0, 1, grid_length, **tkwargs)
    X1, X2 = torch.meshgrid(u, u, indexing="xy")
    all_X = torch.column_stack(
        [X1.reshape(grid_length**2), X2.reshape(grid_length**2)]
    )

    obj_bounds = data["objective_bounds"]
    fX = problem(all_X).to(**tkwargs)
    fX = (fX - obj_bounds[0]) / (obj_bounds[1] - obj_bounds[0])

    pareto_mask = is_non_dominated(fX)
    not_pareto_mask = torch.logical_not(pareto_mask)

    pareto_fX = fX[pareto_mask]
    not_pareto_fX = fX[not_pareto_mask]

    # Get the Pareto fronts.
    interpolated_front = ref_point + pareto_lengths.unsqueeze(-1) * weights

    sample_lengths = torch.zeros(len(Y), len(weights), **tkwargs)
    for t in range(len(Y)):
        Yt = Y[0 : t + 1]
        sample_lengths[t, ...] = scalarization_fn(Yt).max(dim=-2).values
    sample_fronts = ref_point + sample_lengths.unsqueeze(-1) * weights

    mean_lengths = model_lengths.mean(dim=-2)
    mean_fronts = ref_point + mean_lengths.unsqueeze(-1) * weights

    q95_lengths = model_lengths.quantile(q=0.95, dim=-2)
    q95_fronts = ref_point + q95_lengths.unsqueeze(-1) * weights

    q05_lengths = model_lengths.quantile(q=0.05, dim=-2)
    q05_fronts = ref_point + q05_lengths.unsqueeze(-1) * weights

    plot_data = {
        "X": X,
        "Y": Y,
        "fX": fX,
        "pareto_fX": pareto_fX,
        "not_pareto_fX": not_pareto_fX,
        "ref_point": ref_point,
        "interpolated_front": interpolated_front,
        "sample_fronts": sample_fronts,
        "mean_fronts": mean_fronts,
        "q95_fronts": q95_fronts,
        "q05_fronts": q05_fronts,
    }

    torch.save(plot_data, plot_data_path)
