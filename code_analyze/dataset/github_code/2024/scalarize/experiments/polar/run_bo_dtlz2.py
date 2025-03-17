#!/usr/bin/env python3

r"""
A Bayesian optimization loop on the DTLZ2 problem.
"""

import os
import sys

import time

import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)

from botorch.models import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.test_functions.multi_objective import DTLZ2

from scalarize.experiment_utils import initialize_model

from torch import Tensor


def set_model(X: Tensor, Y: Tensor) -> ModelListGP:
    r"""Set the GP model with fixed parameters.

    Args:
        X: A `N x d`-dim Tensor of inputs.
        Y: A `N x M`-dim Tensor of outputs.

    Returns:
        A fitted model.
    """
    mll, model = initialize_model(train_x=X, train_y=Y, **model_kwargs)

    model_1_parameters = {
        "lengthscale": [0.8, 0.8, 0.8],
        "noise": [1.0000e-04],
    }
    model_2_parameters = {
        "lengthscale": [0.8, 0.8, 0.8],
        "noise": [1.0000e-04],
    }

    model.models[0].covar_module.base_kernel.lengthscale = model_1_parameters[
        "lengthscale"
    ]
    model.models[1].covar_module.base_kernel.lengthscale = model_2_parameters[
        "lengthscale"
    ]
    model.models[0].likelihood.noise = model_1_parameters["noise"]
    model.models[1].likelihood.noise = model_2_parameters["noise"]
    model.eval()
    return model


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    grid_length = int(sys.argv[1])
    problem_name = "dtlz2"
    data_path = os.path.join(current_dir, f"data/{problem_name}_data.pt")
    tkwargs = {"dtype": torch.double, "device": "cpu"}
    model_kwargs = {"use_model_list": True, "use_fixed_noise": False}

    # Initial parameters.
    num_initial = 20
    num_iterations = 80
    num_bo_samples = 64
    seed = 0
    torch.manual_seed(seed)

    # Get normalized observations.
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

    obj_bounds = torch.tensor([[-1.5, -1.5], [0.0, 0.0]], **tkwargs)
    fX = problem(all_X).to(**tkwargs)
    fX = (fX - obj_bounds[0]) / (obj_bounds[1] - obj_bounds[0])

    # Initial sample.
    selected_indices = torch.zeros(num_initial + num_iterations, **tkwargs)
    initial_indices = torch.randint(high=len(all_X), size=(num_initial,))
    X = all_X[initial_indices]
    Y = fX[initial_indices]
    selected_indices[:num_initial] = initial_indices

    # Bayesian optimization.
    ref_point = -0.05 * torch.ones(2, **tkwargs)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_bo_samples]))

    time_taken = torch.zeros(num_iterations, **tkwargs)
    start_time = time.time()
    for t in range(num_iterations):
        loop_time = time.time()
        model = set_model(X=X, Y=Y)

        acq = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),
            X_baseline=X,
            prune_baseline=True,
            sampler=sampler,
            cache_root=True,
        )
        acqX = acq(all_X.unsqueeze(-2))
        index = acqX.argmax()
        new_X = all_X[index]
        new_Y = fX[index]

        X = torch.row_stack([X, new_X])
        Y = torch.row_stack([Y, new_Y])
        selected_indices[num_initial + t] = index

        time_taken[t] = time.time() - loop_time

        print(
            f"t={t + 1}/{num_iterations}, "
            f"loop_time={time.time() - loop_time}, "
            f"time elapsed={time.time() - start_time}"
        )

        data = {
            "problem": problem_name,
            "algo": "nehvi",
            "tkwargs": tkwargs,
            "model_kwargs": model_kwargs,
            "objective_bounds": obj_bounds,
            "num_initial": num_initial,
            "num_iterations": num_iterations,
            "grid_length": grid_length,
            "num_bo_samples": num_bo_samples,
            "ref_point": ref_point,
            "X": X,
            "Y": Y,
            "selected_indices": selected_indices,
            "time_taken": time_taken,
        }

        torch.save(data, data_path)
