#!/usr/bin/env python3

r"""
Compute the solution to the rocket problem and the best utilities.
"""

import os
import time

import torch

from rocket_utils import hv_transform, initialize_set_utility
from scalarize.test_functions.multi_objective import RocketInjector

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "data/pareto_rocket_data.pt")
    initial_data_path = os.path.join(current_dir, "data/initial_rocket_data.pt")

    initial_data = torch.load(initial_data_path)
    tkwargs = initial_data["tkwargs"]
    input_grid = initial_data["input_grid"]
    base_function = RocketInjector(negate=True)
    num_objectives = base_function.num_objectives
    set_utility = initialize_set_utility(base_function=base_function, data=initial_data)

    start_time = time.time()
    step_size = 1
    wall_time = torch.zeros(len(input_grid), **tkwargs)
    for i in range(0, len(input_grid), step_size):
        set_utility(input_grid[i : i + step_size])
        best_hypervolume = hv_transform(
            y=set_utility.best_values,
            num_objectives=num_objectives,
        ).mean(dim=0)
        wall_time[i] = time.time() - start_time

        print(
            f"hv_i={i+1}/{len(input_grid)}, "
            f"best_val={best_hypervolume}, "
            f"time_elapsed={time.time()-start_time:.2f}"
        )

        data = {
            "problem": "rocket",
            "best_inputs": set_utility.best_inputs,
            "best_lengths": set_utility.best_values,
            "best_hypervolume": best_hypervolume,
            "ref_point": initial_data["ref_point"],
            "weights": initial_data["weights"],
            "wall_time": wall_time,
        }

        torch.save(data, output_path)
