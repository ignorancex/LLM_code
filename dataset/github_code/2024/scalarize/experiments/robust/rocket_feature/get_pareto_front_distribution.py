#!/usr/bin/env python3

r"""
Compute the Pareto front distribution associated with a data set of runs.
"""

import os
import sys

import time

from math import ceil

import torch

from botorch.models.transforms.input import AppendFeatures
from botorch.utils.transforms import normalize, unnormalize

from scalarize.test_functions.multi_objective import RocketInjector
from scalarize.utils.scalarization_functions import LengthScalarization
from scalarize.utils.scalarization_parameters import UnitVector


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    initial_data_path = os.path.join(current_dir, "data/initial_rocket_data.pt")
    algo = sys.argv[1]

    num_iterations = 60
    grid_length = 128
    num_seeds = 100

    initial_data = torch.load(initial_data_path)
    tkwargs = initial_data["tkwargs"]
    output_path = os.path.join(current_dir, f"data/rocket_cvar_data_{algo}.pt")

    problem = RocketInjector(negate=True)
    noise_grid = initial_data["noise_grid"]
    nadir = initial_data["ref_point"]
    weights = initial_data["weights"]
    alpha = initial_data["alpha"]
    num_initial_points = initial_data["num_initial_points"]
    controllable_dim = initial_data["controllable_dim"]
    num_objectives = initial_data["num_objectives"]
    output_bounds = initial_data["output_bounds"]
    n_w = len(noise_grid)

    input_transform = AppendFeatures(feature_set=noise_grid).eval()
    cvar_num = n_w - ceil(n_w * alpha) + 1
    unit_vector = UnitVector(num_objectives=num_objectives, transform_label="polar")

    u = torch.linspace(0, 1, grid_length, **tkwargs)
    WX, WY = torch.meshgrid(u, u, indexing="xy")

    normalized_nadir = normalize(nadir, output_bounds)

    utilities = torch.zeros(num_seeds, num_iterations + 1, **tkwargs)
    Xs = torch.zeros(
        num_seeds, num_initial_points + num_iterations, controllable_dim, **tkwargs
    )
    Ys = torch.zeros(
        num_seeds, num_initial_points + num_iterations, num_objectives, **tkwargs
    )
    Zs = torch.zeros(num_seeds, grid_length, grid_length, num_objectives, **tkwargs)
    best_lengths = torch.zeros(num_seeds, len(weights), **tkwargs)

    main_s_fn = LengthScalarization(weights=weights, ref_points=nadir)

    start_time = time.time()
    for j in range(num_seeds):
        seed = j + 1
        data_path = os.path.join(
            current_dir, f"data/{algo}/{str(seed).zfill(4)}_{algo}.pt"
        )
        data = torch.load(data_path)
        utilities[j] = data["utilities"]

        Xs[j, ...] = data["X"]
        Ys[j, ...] = data["Y"]

        X_collection = [input_transform(x.unsqueeze(0)) for x in Xs[j, ...]]
        Y_collection = [problem(x) for x in X_collection]
        normalized_Y_collection = [normalize(y, output_bounds) for y in Y_collection]

        Z = torch.zeros(grid_length * grid_length, num_objectives, **tkwargs)
        for row in range(grid_length):
            for column in range(grid_length):
                index = row * grid_length + column
                Wij = torch.column_stack([WX[row, column], WY[row, column]])
                weights = unit_vector(Wij)

                s_fn = LengthScalarization(weights=weights, ref_points=normalized_nadir)

                cvar_length = None
                for y in normalized_Y_collection:
                    cvar_length_x = (
                        s_fn(y)
                        .topk(cvar_num, dim=0, largest=False)
                        .values.mean(dim=0)
                        .unsqueeze(-1)
                    )
                    if cvar_length is None:
                        cvar_length = cvar_length_x
                    else:
                        cvar_length = torch.maximum(cvar_length, cvar_length_x)

                pareto_point = normalized_nadir + cvar_length * weights
                Z[index, ...] = unnormalize(pareto_point, output_bounds)
                best_lengths[j] = torch.maximum(
                    best_lengths[j],
                    main_s_fn(Z[index, ...]),
                )

        Zs[j, ...] = Z.reshape(grid_length, grid_length, num_objectives)
        print(
            f"algo={algo},"
            f"seed={seed}/{num_seeds},"
            f"time_taken={time.time()-start_time:.2f}"
        )

        algo_data = {
            "grid_length": grid_length,
            "Xs": Xs,
            "Ys": Ys,
            "pareto_fronts": Zs,
            "utilities": utilities,
            "best_lengths": best_lengths,
        }
        torch.save(algo_data, output_path)
