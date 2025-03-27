#!/usr/bin/env python3

r"""
The main script used to run the experiments.

"""

import errno
import gc
import json
import os
import sys
from time import time
from typing import Any, Dict, Optional, Tuple

import gpytorch.settings as gpt_settings

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.utils import draw_sobol_samples
from botorch.utils.transforms import unnormalize

from scalarize.experiment_utils import (
    generate_initial_data,
    get_acquisition_function,
    get_problem,
    get_set_utility,
    initialize_model,
)

from torch import Tensor

scalarization_functions_list = [
    "hypervolume",
    "length",
    "linear",
    "ks",
    "d1",
    "igd",
    "r2",
]

supported_labels = [
    "sobol",
    "eui",
    "eui-rg-0.1",
    "eui-rg-0.2",
    "eui-rg-0.3",
    "eui-rg-0.4",
    "eui-rg-0.5",
    "eui-rg-0.6",
    "eui-rg-0.7",
    "eui-rg-0.8",
    "eui-rg-0.9",
    "eui-rg-1.0",
    "eui-thresh-0.1",
    "eui-thresh-0.2",
    "eui-thresh-0.3",
    "eui-thresh-0.4",
    "eui-thresh-0.5",
    "eui-thresh-0.6",
    "eui-thresh-0.7",
    "eui-thresh-0.8",
    "eui-thresh-0.9",
    "eui-mc-1",
    "eui-mc-2",
    "eui-mc-4",
    "eui-mc-8",
    "eui-mc-16",
    "eui-mc-32",
    "eui-mc-64",
    "eui-mc-128",
    "eui-mc-256",
    "eui-fs-1",
    "eui-fs-2",
    "eui-fs-4",
    "eui-fs-8",
    "eui-fs-16",
    "eui-fs-32",
    "eui-fs-64",
    "eui-fs-128",
    "eui-fs-256",
    "eui-ts",
    "eui-ucb",
    "resi",
    "resi-ts",
    "resi-ucb",
    "ehvi",
    "nehvi",
    "parego",
    "nparego",
]


def main(
    seed: int,
    label: str,
    input_dict: Optional[Dict[str, Any]],
    mode: Optional[str],
    output_path: str,
    num_iterations: int,
    num_initial_points: int,
    function_name: str,
    optimization_kwargs: Dict[str, Any],
    scalarization_kwargs: Dict[str, Any],
    util_kwargs: Dict[str, Any],
    sampling_kwargs_util: Dict[str, Any],
    sampling_kwargs_acq: Dict[str, Any],
    acq_kwargs: Dict[str, Any],
    model_kwargs: Optional[Dict[str, Any]] = None,
    save_frequency: int = 5,
    dtype: torch.dtype = torch.double,
    device: torch.device = torch.device("cpu"),
) -> None:
    r"""Run the BO loop for a given number of iterations. Supports restarting of
    prematurely killed experiments.

    Args:
        seed: The experiment seed.
        label: The algorithm to use.
        input_dict: If continuing an existing experiment, this is the output saved by
            the incomplete run.
        mode: Should be `-a` if appending outputs to an existing experiment.
        output_path: The path for the output file.
        num_iterations: Number of iterations of the BO loop to perform.
        num_initial_points: Number of initial evaluations to use.
        function_name: The name of the test function to use.
        optimization_kwargs: Arguments passed to `optimize_acqf`. Includes
            `num_restarts` and `raw_samples` and other optional arguments.
        scalarization_kwargs: The arguments determining the scalarization function.
            Must include the `name` of the scalarization function.
        util_kwargs: The arguments for the utility function.
        sampling_kwargs_util: The arguments determining the utility function used to
            assess performance.
        sampling_kwargs_acq: The arguments determining the utility function used to
            set the acquisition function.
        acq_kwargs: The arguments determining the utility function used in
            the acquisition function.
        model_kwargs: Arguments for `initialize_model`.
        save_frequency: How often to save the output.
        dtype: The tensor dtype to use.
        device: The device to use.
    """
    assert label in supported_labels, "Label is not supported!"
    scalarization_label = scalarization_kwargs["label"]
    assert (
        scalarization_label in scalarization_functions_list
    ), "Scalarization function is not supported"

    torch.manual_seed(seed)
    np.random.seed(seed)

    tkwargs = {"dtype": dtype, "device": device}
    model_kwargs = model_kwargs or {}

    # Get the objective function
    base_function = get_problem(name=function_name, tkwargs=tkwargs)
    base_function.to(**tkwargs)

    # Set default optimization parameters.
    optimization_kwargs.setdefault("num_restarts", 20)
    optimization_kwargs.setdefault("raw_samples", 1024)
    options = optimization_kwargs.get("options")
    if options is None:
        options = {}
        optimization_kwargs["options"] = options
    options.setdefault("batch_limit", 5)
    options.setdefault("maxiter", 200)

    # Get the bounds
    bounds = base_function.bounds.to(**tkwargs)
    standard_bounds = torch.ones(2, bounds.shape[-1], **tkwargs)
    standard_bounds[0] = 0

    acq_bounds = standard_bounds

    # Define the evaluation.
    def eval_problem(X: Tensor) -> Tuple[Tensor, Tensor]:
        X_eval = unnormalize(X, bounds)
        Y = base_function(X_eval)
        return X, Y

    # Get the initial data.
    X, Y = generate_initial_data(
        n=num_initial_points,
        eval_problem=eval_problem,
        bounds=standard_bounds,
        tkwargs=tkwargs,
    )

    set_utility = get_set_utility(
        function_name=function_name,
        scalarization_kwargs=scalarization_kwargs,
        util_kwargs=util_kwargs,
        sampling_kwargs=sampling_kwargs_util,
        tkwargs=tkwargs,
        estimate_utility=False,
        data=None,
        model_kwargs=None,
        acq_kwargs=None,
    )

    # Set some counters to keep track of things.
    batch_size = 1
    start_time = time()
    existing_iterations = 0
    wall_time = torch.zeros(num_iterations, dtype=dtype)
    utility_time = torch.zeros(num_iterations, dtype=dtype)

    # If in the "append" mode, load the existing outputs.
    if input_dict is not None:
        assert torch.allclose(X, input_dict["X"][: X.shape[0]].to(**tkwargs))
        assert torch.allclose(Y, input_dict["Y"][: Y.shape[0]].to(**tkwargs))
        if mode == "-a":
            # Adding iterations to existing output.
            assert input_dict["label"] == label
            existing_iterations = torch.div(
                input_dict["X"].shape[0] - X.shape[0], batch_size, rounding_mode="floor"
            )
            if existing_iterations >= num_iterations:
                raise ValueError("Existing output has as many or more iterations!")

            wall_time[:existing_iterations] = (
                input_dict["wall_time"].cpu().to(dtype=dtype)
            )

            utility_time[:existing_iterations] = (
                input_dict["utility_time"].cpu().to(dtype=dtype)
            )

            X = input_dict["X"].to(**tkwargs)
            Y = input_dict["Y"].to(**tkwargs)
            all_set_utility = input_dict["all_set_utility"]
            # all_optimal_values = input_dict["all_optimal_values"]
            # all_optimal_inputs = input_dict["all_optimal_inputs"]

            # Update the internal state of the set utility.
            set_utility(X)
            compute_util_time = time() - start_time

            # Load current torch and numpy random seed state.
            np_random_state = input_dict["np_random_state"]
            torch_random_state = input_dict["torch_random_state"]

            np.random.set_state(np_random_state)
            torch.random.set_rng_state(torch_random_state)

            start_time = start_time - float(input_dict["wall_time"][-1])
        else:
            # This should never happen!
            raise RuntimeError("Mode unsupported!")
    else:
        all_set_utility = torch.tensor([set_utility(X)], **tkwargs)
        # all_optimal_values = [set_utility.best_values]
        # all_optimal_inputs = [set_utility.best_inputs]
        compute_util_time = time() - start_time

    for i in range(existing_iterations, num_iterations):
        print(
            f"Starting label {label}, "
            f"seed {seed}, "
            f"iteration {i}, "
            f"time: {time()-start_time}, "
            f"current set utility: {all_set_utility[-1]}."
        )
        # Fit the model.
        mll, model = initialize_model(train_x=X, train_y=Y, **model_kwargs)
        fit_gpytorch_mll(mll)

        # Set the input
        if label == "sobol":
            candidates = (
                draw_sobol_samples(
                    bounds=standard_bounds,
                    n=1,
                    q=batch_size,
                )
                .squeeze(0)
                .to(**tkwargs)
            )

        else:
            with gpt_settings.cholesky_max_tries(6):
                acq_func = get_acquisition_function(
                    name=function_name,
                    iteration=i,
                    num_iterations=num_iterations,
                    label=label,
                    model=model,
                    X_baseline=X,
                    scalarization_kwargs=scalarization_kwargs,
                    sampling_kwargs=sampling_kwargs_acq,
                    acq_kwargs=acq_kwargs,
                    util_kwargs=util_kwargs,
                    tkwargs=tkwargs,
                    Y_baseline=Y,
                    bounds=acq_bounds,
                )
                torch.cuda.empty_cache()

                acq_candidates, _ = optimize_acqf(
                    acq_function=acq_func,
                    bounds=acq_bounds,
                    q=batch_size,
                    **optimization_kwargs,
                )

                candidates = acq_candidates

            # Free memory.
            del acq_func, mll, model
            gc.collect()
            torch.cuda.empty_cache()

        print("candidates={}".format(candidates))
        # Get the new observations and update the data.
        new_x, new_y = eval_problem(candidates)
        # Note that the saved X is the normalized.
        X = torch.cat([X, new_x], dim=0)
        Y = torch.cat([Y, new_y], dim=0)
        wall_time[i] = time() - start_time - compute_util_time

        init_util_time = time()
        new_set_utility = torch.tensor([set_utility(new_x)], **tkwargs)

        all_set_utility = torch.cat([all_set_utility, new_set_utility], dim=0)
        # all_optimal_values = all_optimal_values + [set_utility.best_values]
        # all_optimal_inputs = all_optimal_inputs + [set_utility.best_inputs]

        compute_util_time = time() - init_util_time
        utility_time[i] = compute_util_time

        # Periodically save the output.
        if num_iterations % save_frequency == 0:
            np_random_state = np.random.get_state()
            torch_random_state = torch.random.get_rng_state()

            output_dict = {
                "label": label,
                "X": X.cpu(),
                "Y": Y.cpu(),
                "wall_time": wall_time[: i + 1],
                "utility_time": utility_time[: i + 1],
                "all_set_utility": all_set_utility,
                # "all_optimal_values": all_optimal_values,
                # "all_optimal_inputs": all_optimal_inputs,
                "np_random_state": np_random_state,
                "torch_random_state": torch_random_state,
            }
            torch.save(output_dict, output_path)

    np_random_state = np.random.get_state()
    torch_random_state = torch.random.get_rng_state()
    # Save the final output.
    output_dict = {
        "label": label,
        "X": X.cpu(),
        "Y": Y.cpu(),
        "wall_time": wall_time,
        "utility_time": utility_time,
        "all_set_utility": all_set_utility,
        # "all_optimal_values": all_optimal_values,
        # "all_optimal_inputs": all_optimal_inputs,
        "np_random_state": np_random_state,
        "torch_random_state": torch_random_state,
    }
    torch.save(output_dict, output_path)
    return


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, sys.argv[1])
    config_path = os.path.join(exp_dir, "config.json")
    label = sys.argv[2]
    seed = int(float(sys.argv[3]))
    last_arg = sys.argv[4] if len(sys.argv) > 4 else None
    output_path = os.path.join(exp_dir, label, f"{str(seed).zfill(4)}_{label}.pt")
    input_dict = None
    mode = None
    if os.path.exists(output_path):
        if last_arg and last_arg in ["-a", "-f"]:
            mode = last_arg
            if last_arg == "-f":
                print("Overwriting the existing output!")
            elif last_arg == "-a":
                print(
                    "Appending iterations to existing output!"
                    "Warning: If parameters other than `iterations` have "
                    "been changed, this will corrupt the output!"
                )
                input_dict = torch.load(output_path)
            else:
                raise RuntimeError
        else:
            print(
                "The output file exists for this experiment & seed!"
                "Pass -f as the 4th argument to overwrite!"
                "Pass -a as the 4th argument to add more iterations!"
            )
            quit()
    elif not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    main(
        seed=seed,
        label=label,
        input_dict=input_dict,
        mode=mode,
        output_path=output_path,
        **kwargs,
    )
