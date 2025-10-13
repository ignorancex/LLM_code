#!/usr/bin/env python3

r"""
Run the rocket experiment with the robust random scalarization approach with the
upper confidence bound strategy.
"""

import functools
import gc
import os
import sys
from time import time

import numpy as np
import torch

from botorch import fit_gpytorch_mll
from botorch.acquisition.risk_measures import CVaR
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.transforms.input import AppendFeatures
from botorch.sampling.stochastic_samplers import StochasticSampler

from rocket_utils import get_objective_function, hv_transform, initialize_set_utility

from scalarize.acquisition.monte_carlo import qNoisyExpectedImprovement
from scalarize.experiment_utils import initialize_model
from scalarize.models.transforms.outcome import Normalize
from scalarize.robust_experiment_utils import get_robust_utility_mc_objective
from scalarize.test_functions.multi_objective import RocketInjector
from scalarize.utils.scalarization_functions import LengthScalarization
from scalarize.utils.scalarization_objectives import get_utility_mcobjective


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    label = "str_ucb"
    seed = int(float(sys.argv[1]))
    output_path = os.path.join(
        current_dir, "data", label, f"{str(seed).zfill(4)}_{label}.pt"
    )
    initial_data_path = os.path.join(current_dir, "data/initial_rocket_data.pt")
    num_iterations = 60

    initial_data = torch.load(initial_data_path)
    tkwargs = initial_data["tkwargs"]
    input_grid = initial_data["input_grid"]
    noise_grid = initial_data["noise_grid"]
    weights = initial_data["weights"]
    alpha = initial_data["alpha"]

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

    beta = 2.0
    step_size = 100
    sampled_weights = torch.zeros(num_iterations, num_objectives, **tkwargs)
    acq_ref_points = torch.zeros(num_iterations, num_objectives, **tkwargs)

    wall_time = torch.zeros(num_iterations, **tkwargs)
    start_time = time()

    random_weight_indices = np.random.choice(
        len(weights), replace=False, size=num_iterations
    )

    num_noise_samples = 256
    num_samples = 1
    cache_root = False
    prune_baseline = True
    sampler = StochasticSampler(sample_shape=torch.Size([num_samples]))
    outcome_transform = Normalize(bounds=None)

    for t in range(num_iterations):
        random_noise_indices = np.random.choice(
            len(noise_grid), replace=False, size=num_noise_samples
        )
        noise_grid_sample = noise_grid[random_noise_indices]
        acq_input_transform = AppendFeatures(feature_set=noise_grid_sample).eval()
        acq_robust_objective = CVaR(alpha=alpha, n_w=len(noise_grid_sample))

        X_aug = torch.column_stack([X, X_env])
        mll, model = initialize_model(
            train_x=X_aug, train_y=Y, input_transform=acq_input_transform
        )
        fit_gpytorch_mll(mll)

        def f_ucb(X):
            posterior = model.posterior(X)
            mean = posterior.mean
            variance = posterior.variance
            return mean + beta * torch.sqrt(variance)

        acq_model = GenericDeterministicModel(f=f_ucb, num_outputs=model.num_outputs)
        X_baseline = X

        # Estimate reference point.
        ymin = Y.min(dim=0).values
        ymax = Y.max(dim=0).values
        acq_ref_points[t] = ymin - 0.1 * (ymax - ymin)

        sampled_weights[t] = weights[random_weight_indices[t]]
        acq_s_fn = LengthScalarization(
            ref_points=acq_ref_points[t], weights=sampled_weights[t]
        )

        util_obj = get_utility_mcobjective(
            scalarization_fn=acq_s_fn,
            outcome_transform=outcome_transform,
        )
        mc_obj = get_robust_utility_mc_objective(
            scalarization_objective=util_obj,
            robust_objective=acq_robust_objective,
            include_augmentation=False,
            model=model,
            tkwargs=tkwargs,
        )

        acq_func = qNoisyExpectedImprovement(
            model=acq_model,
            objective=mc_obj,
            X_baseline=X_baseline,
            sampler=sampler,
            prune_baseline=prune_baseline,
            cache_root=cache_root,
        )
        # Grid search.
        acq_values = torch.zeros(len(input_grid), **tkwargs)
        for i in range(0, len(input_grid), step_size):
            new_batch = acq_func(input_grid[i : i + step_size].unsqueeze(-2))
            acq_values[i : i + step_size] = new_batch.clone().detach()

            print(
                f"t={t + 1}/{num_iterations}, "
                f"i={i + 1}/{len(input_grid)}, "
                f"time_elapsed={time() - start_time:.2f}"
            )

            del new_batch
            gc.collect()
            torch.cuda.empty_cache()

        del acq_func, mll, model
        gc.collect()
        torch.cuda.empty_cache()

        bools = [input_grid == x for x in X]
        reduced_bools = [b.all(dim=-1) for b in bools]
        mask = ~functools.reduce(np.logical_or, reduced_bools).to(bool)

        best_index = acq_values[mask].argmax()
        acq_x = input_grid[mask][best_index]

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
            f"hv={utilities[t+1]}, "
            f"time_elapsed={wall_time[t]:.2f}"
        )

        data = {
            "problem": "rocket",
            "algo": label,
            "seed": seed,
            "num_iterations": num_iterations,
            "beta": beta,
            "num_samples": num_samples,
            "num_noise_samples": num_noise_samples,
            "random_weight_indices": random_weight_indices,
            "sampled_weights": sampled_weights,
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
