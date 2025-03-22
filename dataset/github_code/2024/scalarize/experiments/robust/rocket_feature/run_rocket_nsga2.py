#!/usr/bin/env python3

r"""
Run the rocket experiment with NSGA2.
"""

import os
import sys
from time import time

import numpy as np
import torch

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from rocket_utils import get_objective_function, hv_transform, initialize_set_utility

from scalarize.test_functions.multi_objective import RocketInjector


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    label = "nsga2"
    seed = int(float(sys.argv[1]))
    output_path = os.path.join(
        current_dir, "data", label, f"{str(seed).zfill(4)}_{label}.pt"
    )
    initial_data_path = os.path.join(current_dir, "data/initial_rocket_data.pt")
    num_iterations = 60
    pop_size = 10
    num_gen = int(num_iterations / pop_size) + 1

    initial_data = torch.load(initial_data_path)
    tkwargs = initial_data["tkwargs"]
    input_grid = initial_data["input_grid"]
    controllable_dim = initial_data["controllable_dim"]
    input_grid_length = initial_data["input_grid_length"]

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

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=1.0, eta=3.0, repair=RoundingRepair()),
        mutation=PM(prob=1.0, eta=3.0, repair=RoundingRepair()),
    )

    Xs = []
    Ys = []
    X_envs = []

    class NumpyRocket(Problem):
        r"""Numpy version of the discretized rocket problem."""

        def __init__(self):
            r"""Discretized rocket problem."""
            super().__init__(
                n_var=controllable_dim,
                n_obj=num_objectives,
                n_ieq_constr=0,
                xl=0.0,
                xu=input_grid_length - 1,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            xt = torch.tensor(x, **tkwargs) / (input_grid_length - 1)
            xt, yt, xt_env = eval_problem(xt)
            Xs.append(xt)
            Ys.append(yt)
            X_envs.append(xt_env)

            out["F"] = -yt.detach().numpy()

    pymoo_problem = NumpyRocket()

    res = minimize(
        pymoo_problem,
        algorithm,
        ("n_gen", num_gen),
        verbose=False,
    )

    Xs = torch.row_stack(Xs)
    Ys = torch.row_stack(Ys)
    X_envs = torch.row_stack(X_envs)

    start_time = time()
    for t in range(num_iterations):
        new_x = Xs[t : t + 1]
        new_y = Ys[t : t + 1]
        new_env_x = X_envs[t : t + 1]
        set_utility(new_x)
        utilities[t + 1] = hv_transform(
            y=set_utility.best_values,
            num_objectives=num_objectives,
        ).mean(dim=0)

        X = torch.cat([X, new_x], dim=0)
        Y = torch.cat([Y, new_y], dim=0)
        X_env = torch.cat([X_env, new_env_x], dim=0)
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
            "pop_size": pop_size,
            "num_gen": num_gen,
            "X": X,
            "Y": Y,
            "X_env": X_env,
            "utilities": utilities,
            "wall_time": wall_time,
            "np_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
        }
        torch.save(data, output_path)
