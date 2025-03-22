#!/usr/bin/env python3

r"""
Computing the lengths for the GP samples associated with the DTLZ2 problem.
"""

import os
import sys

import time

import torch
from botorch.test_functions.multi_objective import DTLZ2

from botorch.utils.multi_objective import is_non_dominated
from scalarize.utils.scalarization_functions import LengthScalarization
from scalarize.utils.scalarization_parameters import UnitVector


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    num_weights = int(sys.argv[1])
    problem_name = "dtlz2"
    data_path = os.path.join(current_dir, f"data/{problem_name}_data.pt")
    sample_path = os.path.join(current_dir, f"data/{problem_name}_model_samples.pt")
    length_path = os.path.join(current_dir, f"data/{problem_name}_lengths.pt")

    # Load the data.
    data = torch.load(data_path)
    grid_length = data["grid_length"]
    ref_point = data["ref_point"]
    tkwargs = data["tkwargs"]

    # Load the samples.
    sample_data = torch.load(sample_path)
    samples = sample_data["samples"]
    num_samples = sample_data["num_samples"]
    num_evaluations = sample_data["num_evaluations"]

    # Compute the objective values.
    num_objectives = 2
    problem = DTLZ2(dim=3, negate=True)

    u = torch.linspace(0, 1, grid_length, **tkwargs)
    X1, X2, X3 = torch.meshgrid(u, u, u, indexing="xy")
    all_X = torch.column_stack(
        [
            X1.reshape(grid_length**3),
            X2.reshape(grid_length**3),
            X3.reshape(grid_length**3),
        ]
    )

    obj_bounds = data["objective_bounds"]
    fX = problem(all_X).to(**tkwargs)
    fX = (fX - obj_bounds[0]) / (obj_bounds[1] - obj_bounds[0])
    pareto_mask = is_non_dominated(fX)
    pX = all_X[is_non_dominated(fX)]
    pfX = fX[is_non_dominated(fX)]

    # Length scalarization.
    t = torch.linspace(0, 1, num_weights, **tkwargs).unsqueeze(-1)
    unit_vector = UnitVector(num_objectives=num_objectives, transform_label="polar")
    weights = unit_vector(t)
    scalarization_fn = LengthScalarization(weights=weights, ref_points=ref_point)

    pareto_lengths = scalarization_fn(fX).max(dim=-2).values

    all_lengths = torch.zeros(num_evaluations, num_samples, num_weights, **tkwargs)
    time_taken = torch.zeros(num_evaluations, **tkwargs)
    start_time = time.time()
    for t in range(num_evaluations):
        loop_time = time.time()
        all_lengths[t, ...] = scalarization_fn(samples[t, ...]).max(dim=-2).values
        time_taken[t] = time.time() - loop_time

        length_data = {
            "weights": weights,
            "ref_point": ref_point,
            # "all_X": all_X,
            # "all_Y": fX,
            # "pareto_set": pX,
            # "pareto_front": pfX,
            "pareto_lengths": pareto_lengths,
            "lengths": all_lengths,
            "time_taken": time_taken,
        }

        torch.save(length_data, length_path)
