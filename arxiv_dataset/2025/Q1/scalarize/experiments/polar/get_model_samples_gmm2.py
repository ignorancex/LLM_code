#!/usr/bin/env python3

r"""
Sampling from a GP model that was trained on data from the GMM2 problem.
"""

import os
import sys

import time

import numpy as np
import torch

from botorch.models import ModelListGP

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
        "lengthscale": [0.1, 0.1],
        "noise": [1.0000e-04],
    }
    model_2_parameters = {
        "lengthscale": [0.1, 0.1],
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
    num_samples = int(sys.argv[1])
    problem_name = "gmm2"
    data_path = os.path.join(current_dir, f"data/{problem_name}_data.pt")
    sample_path = os.path.join(current_dir, f"data/{problem_name}_model_samples.pt")

    # Load the data.
    data = torch.load(data_path)
    X = data["X"]
    Y = data["Y"]
    grid_length = data["grid_length"]
    ref_point = data["ref_point"]
    tkwargs = data["tkwargs"]
    model_kwargs = data["model_kwargs"]
    num_initial = data["num_initial"]
    num_iterations = data["num_iterations"]
    num_evaluations = num_initial + num_iterations
    num_objectives = 2

    u = torch.linspace(0, 1, grid_length, **tkwargs)
    X1, X2 = torch.meshgrid(u, u, indexing="xy")
    all_X = torch.column_stack(
        [
            X1.reshape(grid_length**2),
            X2.reshape(grid_length**2),
        ]
    )

    # Get the samples.
    sample_seed = 0
    num_evaluations = num_iterations + num_initial
    all_samples = torch.zeros(
        num_evaluations, num_samples, len(all_X), num_objectives, **tkwargs
    )

    time_taken = torch.zeros(num_evaluations, **tkwargs)
    start_time = time.time()
    for t in range(num_evaluations):
        loop_time = time.time()
        torch.manual_seed(sample_seed)
        model = set_model(X=X[: t + 1], Y=Y[: t + 1])

        for j in range(2):
            posterior = model.models[j](all_X)
            mean = posterior.mean.detach().numpy()
            cov = posterior.covariance_matrix.detach().numpy()
            outcome_transform = model.models[j].outcome_transform
            samples = torch.tensor(
                np.random.multivariate_normal(mean, cov, size=num_samples), **tkwargs
            )
            if t > 0:
                t_samples, _ = outcome_transform.untransform(samples)
            else:
                t_samples = samples
            all_samples[t, ..., j] = t_samples

        time_taken[t] = time.time() - loop_time

        print(
            f"t={t + 1}/{num_evaluations}, "
            f"loop_time={time.time() - loop_time}, "
            f"time elapsed={time.time() - start_time}"
        )

        sample_data = {
            "problem": problem_name,
            "num_evaluations": num_initial + num_iterations,
            "num_samples": num_samples,
            "sample_seed": sample_seed,
            "samples": all_samples,
            "time_taken": time_taken,
        }

        torch.save(sample_data, sample_path)
