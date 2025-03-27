#!/usr/bin/env python3

r"""
Run the rocket experiment with random search.
"""

import os
import sys
from time import time

import numpy as np
import torch

from rocket_utils import get_objective_function, hv_transform, initialize_set_utility
from scalarize.test_functions.multi_objective import RocketInjector


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    label = "sobol"
    seed = int(float(sys.argv[1]))
    output_path = os.path.join(
        current_dir, "data", label, f"{str(seed).zfill(4)}_{label}.pt"
    )
    initial_data_path = os.path.join(current_dir, "data/initial_rocket_data.pt")
    num_iterations = 60

    initial_data = torch.load(initial_data_path)
    tkwargs = initial_data["tkwargs"]
    input_grid = initial_data["input_grid"]

    X = initial_data["X"]
    Y = initial_data["Y"]
    X_env = initial_data["X_env"]
    base_function = RocketInjector(negate=True)
    num_objectives = base_function.num_objectives

    torch.manual_seed(seed)
    np.random.seed(seed)

    eval_problem = get_objective_function(
        base_function=base_function,
        observation_kwargs=initial_data,
    )

    set_utility = initialize_set_utility(base_function=base_function, data=initial_data)

    for i in range(len(X)):
        set_utility(X[i : i + 1])

    utilities = torch.zeros(num_iterations + 1, **tkwargs)
    utilities[0] = hv_transform(
        y=set_utility.best_values,
        num_objectives=num_objectives,
    ).mean(dim=0)

    wall_time = torch.zeros(num_iterations, **tkwargs)
    start_time = time()
    random_indices = np.random.choice(
        len(input_grid), replace=False, size=num_iterations
    )
    for t in range(num_iterations):
        new_x, new_y, new_x_env = eval_problem(
            input_grid[random_indices[t]].unsqueeze(0)
        )
        set_utility(new_x)
        utilities[t + 1] = hv_transform(
            y=set_utility.best_values,
            num_objectives=num_objectives,
        ).mean(dim=0)

        X = torch.cat([X, new_x], dim=0)
        Y = torch.cat([Y, new_y], dim=0)
        X_env = torch.cat([X_env, new_x_env], dim=0)
        wall_time[t] = time() - start_time
        print(
            f"t={t+1}/{num_iterations}, "
            f"x={new_x}, "
            f"hv={utilities[t + 1]}, "
            f"time_elapsed={wall_time[t]:.2f}"
        )

        data = {
            "problem": "rocket",
            "algo": label,
            "seed": seed,
            "num_iterations": num_iterations,
            "X": X,
            "Y": Y,
            "X_env": X_env,
            "utilities": utilities,
            "wall_time": wall_time,
            "np_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
        }
        torch.save(data, output_path)
