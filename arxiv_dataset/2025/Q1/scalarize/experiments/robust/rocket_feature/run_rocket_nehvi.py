#!/usr/bin/env python3

r"""
Run the rocket experiment with the standard expected hypervolume improvement.
"""

import functools
import gc
import os
import sys
from time import time

import numpy as np
import torch

from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.sampling.normal import SobolQMCNormalSampler

from rocket_utils import get_objective_function, hv_transform, initialize_set_utility

from scalarize.experiment_utils import initialize_model
from scalarize.test_functions.multi_objective import RocketInjector


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    label = "nehvi"
    seed = int(float(sys.argv[1]))
    output_path = os.path.join(
        current_dir, "data", label, f"{str(seed).zfill(4)}_{label}.pt"
    )
    initial_data_path = os.path.join(current_dir, "data/initial_rocket_data.pt")
    num_iterations = 60

    initial_data = torch.load(initial_data_path)
    tkwargs = initial_data["tkwargs"]
    input_grid = initial_data["input_grid"]
    whole_input_grid = initial_data["whole_input_grid"]

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

    num_samples = 128
    step_size = 1000
    acq_ref_points = torch.zeros(num_iterations, num_objectives, **tkwargs)

    wall_time = torch.zeros(num_iterations, **tkwargs)
    cache_root = True
    prune_baseline = True
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))

    start_time = time()
    for t in range(num_iterations):
        X_aug = torch.column_stack([X, X_env])
        mll, model = initialize_model(train_x=X_aug, train_y=Y)
        fit_gpytorch_mll(mll)

        X_baseline = X_aug
        # Estimate reference point.
        ymin = Y.min(dim=0).values
        ymax = Y.max(dim=0).values
        acq_ref_points[t] = ymin - 0.1 * (ymax - ymin)

        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=acq_ref_points[t].tolist(),
            X_baseline=X_baseline,
            prune_baseline=prune_baseline,
            sampler=sampler,
            cache_root=cache_root,
        )
        # Grid search.
        acq_values = torch.zeros(len(whole_input_grid), **tkwargs)
        for i in range(0, len(whole_input_grid), step_size):
            X_i = whole_input_grid[i : i + step_size].unsqueeze(-2)
            new_batch = acq_func(X_i)
            acq_values[i : i + step_size] = new_batch.clone().detach()

            print(
                f"t={t + 1}/{num_iterations}, "
                f"i={i + 1}/{len(whole_input_grid)}, "
                f"time_elapsed={time() - start_time:.2f}"
            )

        del acq_func, mll, model
        gc.collect()
        torch.cuda.empty_cache()

        bools = [whole_input_grid[..., :2] == x for x in X]
        reduced_bools = [b.all(dim=-1) for b in bools]
        mask = ~functools.reduce(np.logical_or, reduced_bools).to(bool)

        best_index = acq_values[mask].argmax()
        acq_x = whole_input_grid[..., :2][mask][best_index]

        new_x, new_y, new_env_x = eval_problem(acq_x.unsqueeze(0))
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
            f"time_elapsed={time() - start_time:.2f}"
        )

        data = {
            "problem": "rocket",
            "algo": label,
            "seed": seed,
            "num_iterations": num_iterations,
            "num_samples": num_samples,
            "acq_ref_points": acq_ref_points,
            "X": X,
            "Y": Y,
            "X_env": X_env,
            "utilities": utilities,
            "wall_time": wall_time,
            "np_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
        }

        torch.save(data, output_path)
