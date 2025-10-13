#!/usr/bin/env python3

r"""
The main script used to run the robust experiments.

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

from scalarize.experiment_utils import initialize_model

from scalarize.robust_experiment_utils import (
    generate_perturbed_initial_data,
    get_acquisition_function_robust,
    get_decision_input,
    get_environment_acquisition,
    get_input_transform,
    get_perturbations,
    get_problem_variables,
    get_robust_set_utility,
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

environment_settings = [
    "general",
    "simulated",
]

input_transforms = [
    "input",
    "feature",
]

robust_objectives_list = [
    "ChiSquare",
    "CVaR",
    "Expectation",
    "KL",
    "MCVaR",
    "TotalVariation",
    "WorstCase",
]

supported_labels = [
    "sobol",
    "eui",
    "eui-ucb",
    "eui-ts",
    "resi",
    "resi-ucb",
    "resi-ts",
    "aresi-ucb",
    "robust-eui",
    "robust-eui-ucb",
    "robust-eui-ts",
    "robust-resi",
    "robust-resi-ucb",
    "robust-resi-ts",
    "ehvi",
    "nehvi",
    "parego",
    "nparego",
]

robust_acquisition_list = [
    "robust-eui",
    "robust-eui-ucb",
    "robust-eui-ts",
    "robust-resi",
    "robust-resi-ucb",
    "robust-resi-ts",
]

augumented_acquisition_list = [
    "aresi-ucb",
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
    environment_kwargs: Dict[str, Any],
    input_transform_kwargs: Dict[str, Any],
    robust_kwargs: Dict[str, Any],
    perturbation_kwargs_util: Dict[str, Any],
    perturbation_kwargs_acq: Dict[str, Any],
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
        environment_kwargs: The arguments determining the environment. In the
            `general` setting, the perturbation is set by the environment, whereas in
             the `simulated` setting, the perturbation is set by the decision maker.
             Whether the perturbations are observed by the decision maker is
             determined by the boolean `observed`.
        input_transform_kwargs: The arguments determining the input transform.
        robust_kwargs: The arguments determining the robust objective.
        perturbation_kwargs_util: The arguments determining the perturbations used to
            assess the performance.
        perturbation_kwargs_acq: The arguments determining the perturbations used to
            set the acquisition function.
        model_kwargs: Arguments for `initialize_model`. The input transform is added
            later in the main.
        save_frequency: How often to save the output.
        dtype: The tensor dtype to use.
        device: The device to use.
    """
    assert label in supported_labels, "Label is not supported!"
    scalarization_label = scalarization_kwargs["label"]
    assert (
        scalarization_label in scalarization_functions_list
    ), "Scalarization function is not supported"

    robust_label = robust_kwargs["label"]
    assert robust_label in robust_objectives_list, "Robust objective is not supported"

    environment_setting = environment_kwargs["setting"]
    assert (
        environment_setting in environment_settings
    ), "Environment type is not supported"
    environment_kwargs.setdefault("observed", True)
    observed_perturbations = environment_kwargs["observed"]

    input_transform_label = input_transform_kwargs["label"]
    assert input_transform_label in input_transforms, "Input transform is not supported"
    use_input_transform = label in robust_acquisition_list

    # When we have a feature input transform and do not observe the feature, then
    # we do not use the input transform.
    if use_input_transform:
        if (
            input_transform_label == "feature"
            and environment_setting == "general"
            and not observed_perturbations
        ):
            raise ValueError(
                "We cannot compute the robust acquisition function in the general "
                "unobserved setting with feature perturbations."
            )

    torch.manual_seed(seed)
    np.random.seed(seed)

    tkwargs = {"dtype": dtype, "device": device}

    # Get the problem variables.
    problem_variables = get_problem_variables(
        function_name=function_name,
        tkwargs=tkwargs,
        environment_kwargs=environment_kwargs,
        input_transform_kwargs=input_transform_kwargs,
    )
    base_function = problem_variables["base_function"]

    # Get the indices.
    permute_indices = problem_variables["permute_indices"]

    # Get the additional problem-dependent permutation kwargs.
    perturbation_dim = problem_variables["perturbation_dim"]
    perturbation_bounds = problem_variables["perturbation_bounds"]

    perturbation_kwargs_util.setdefault("dimension", perturbation_dim)
    perturbation_kwargs_util.setdefault("bounds", perturbation_bounds)

    perturbation_kwargs_acq.setdefault("dimension", perturbation_dim)
    perturbation_kwargs_acq.setdefault("bounds", perturbation_bounds)

    # Get the bounds.
    decision_dim = problem_variables["decision_dim"]
    controllable_dim = problem_variables["controllable_dim"]

    # Bounds for the controllable variable.
    standard_decision_bounds = problem_variables["standard_decision_bounds"]
    standard_problem_bounds = problem_variables["standard_problem_bounds"]
    standard_controllable_bounds = problem_variables["standard_controllable_bounds"]

    acq_bounds = standard_problem_bounds
    if input_transform_label == "feature" and label in robust_acquisition_list:
        acq_bounds = standard_decision_bounds

    # Set default optimization parameters.
    optimization_kwargs.setdefault("num_restarts", 20)
    optimization_kwargs.setdefault("raw_samples", 1024)
    options = optimization_kwargs.get("options")
    if options is None:
        options = {}
        optimization_kwargs["options"] = options
    options.setdefault("batch_limit", 5)
    options.setdefault("maxiter", 200)

    # Define the evaluation.
    def eval_problem(X: Tensor) -> Tuple[Tensor, Tensor]:
        if environment_setting == "general":
            # Incorporate perturbations into the input.
            perturbation_set = get_perturbations(
                n_w=len(X),
                perturbation_kwargs=perturbation_kwargs_util,
                tkwargs=tkwargs,
            )
            if input_transform_label == "feature":
                X_augmented = torch.cat([X, perturbation_set], dim=-1)
            elif input_transform_label == "input":
                input_transform = get_input_transform(
                    input_transform_kwargs=input_transform_kwargs,
                    perturbation_set=perturbation_set,
                ).eval()
                X_augmented = input_transform(X)
                # Have to extract the necessary inputs.
                X_augmented = torch.stack(
                    [X_augmented[j * len(X) + j] for j in range(len(X))]
                )
            else:
                raise ValueError("The input transform is not supported!")
        elif environment_setting == "simulated":
            # The user decides the input perturbation.
            X_augmented = X
        else:
            raise ValueError("The environment setting is not supported!")

        X_evaluation = unnormalize(
            X_augmented[..., permute_indices], base_function.bounds
        )

        Y = base_function(X_evaluation)
        if observed_perturbations:
            X_observed = X_augmented
        else:
            X_observed = X

        return X_observed, Y, X_augmented

    # Get the initial data.
    # Note that when the feature perturbation is used, then the feature input is
    # appended to the end: `X = [X_input, X_feature]`, similarly with `Z`.
    X, Y, Z = generate_perturbed_initial_data(
        n=num_initial_points,
        eval_problem=eval_problem,
        bounds=standard_controllable_bounds,
        tkwargs=tkwargs,
    )

    set_utility = get_robust_set_utility(
        function_name=function_name,
        scalarization_kwargs=scalarization_kwargs,
        util_kwargs=util_kwargs,
        sampling_kwargs=sampling_kwargs_util,
        robust_kwargs=robust_kwargs,
        environment_kwargs=environment_kwargs,
        input_transform_kwargs=input_transform_kwargs,
        perturbation_kwargs=perturbation_kwargs_util,
        tkwargs=tkwargs,
        estimate_utility=False,
        data=None,
        model_kwargs=None,
        acq_kwargs=None,
    )

    # Get the model arguments. Note that is important to do this after generating
    # the initial data, otherwise, we have inconsistencies between the initial data
    # across the different seeds.
    model_kwargs = model_kwargs or {}
    if use_input_transform:
        # Note that the perturbations used for the acquisition function is different
        # from the perturbations used for the performance assessment.
        num_perturbations = perturbation_kwargs_acq["num_perturbations"]
        perturbation_set_acq = get_perturbations(
            n_w=num_perturbations,
            perturbation_kwargs=perturbation_kwargs_acq,
            tkwargs=tkwargs,
        )

        input_transform_acq = get_input_transform(
            input_transform_kwargs=input_transform_kwargs,
            perturbation_set=perturbation_set_acq,
        )
    else:
        input_transform_acq = None
    model_kwargs["input_transform"] = input_transform_acq
    if "ts" in label:
        model_kwargs["use_model_list"] = False

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
        assert torch.allclose(Z, input_dict["Z"][: Z.shape[0]].to(**tkwargs))
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
            Z = input_dict["Z"].to(**tkwargs)
            all_set_utility = input_dict["all_set_utility"]
            # all_optimal_values = input_dict["all_optimal_values"]
            # all_optimal_inputs = input_dict["all_optimal_inputs"]

            # Update the internal state of the set utility.
            Z_decision = get_decision_input(
                X=Z,
                input_transform_label=input_transform_label,
                decision_dim=decision_dim,
            )
            set_utility(Z_decision)
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
        Z_decision = get_decision_input(
            X=Z,
            input_transform_label=input_transform_label,
            decision_dim=decision_dim,
        )

        all_set_utility = torch.tensor([set_utility(Z_decision)], **tkwargs)
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

        mll, model = initialize_model(train_x=X, train_y=Y, **model_kwargs)
        fit_gpytorch_mll(mll)

        X_baseline = X
        if label in robust_acquisition_list and input_transform_label == "feature":
            X_baseline = X[..., 0:decision_dim]

        # Set the input.
        if label == "sobol":
            candidates = (
                draw_sobol_samples(
                    bounds=standard_controllable_bounds,
                    n=1,
                    q=batch_size,
                )
                .squeeze(0)
                .to(**tkwargs)
            )

        else:
            with gpt_settings.cholesky_max_tries(6):

                acq_func = get_acquisition_function_robust(
                    name=function_name,
                    iteration=i,
                    num_iterations=num_iterations,
                    label=label,
                    model=model,
                    X_baseline=X_baseline,
                    scalarization_kwargs=scalarization_kwargs,
                    sampling_kwargs=sampling_kwargs_acq,
                    acq_kwargs=acq_kwargs,
                    util_kwargs=util_kwargs,
                    input_transform_kwargs=input_transform_kwargs,
                    robust_kwargs=robust_kwargs,
                    perturbation_kwargs=perturbation_kwargs_acq,
                    tkwargs=tkwargs,
                    Y_baseline=Y,
                    bounds=acq_bounds,
                )

                acq_candidates, _ = optimize_acqf(
                    acq_function=acq_func,
                    bounds=acq_bounds,
                    q=batch_size,
                    **optimization_kwargs,
                )

                candidates = acq_candidates[..., 0:controllable_dim]

                if (
                    label in robust_acquisition_list
                    and input_transform_label == "feature"
                    and environment_setting == "simulated"
                ):
                    # Optimize for the environment variable separately.
                    env_acq_func = get_environment_acquisition(
                        model=model,
                        name=function_name,
                        scalarization_kwargs=scalarization_kwargs,
                        acq_kwargs=acq_kwargs,
                        util_kwargs=util_kwargs,
                        tkwargs=tkwargs,
                        X_baseline=X_baseline,
                        Y_baseline=Y,
                        bounds=acq_bounds,
                    )
                    # This assumes a batch size of one.
                    fixed_features = {i: value for i, value in enumerate(candidates[0])}

                    aug_candidates, _ = optimize_acqf(
                        acq_function=env_acq_func,
                        fixed_features=fixed_features,
                        bounds=standard_problem_bounds,
                        q=batch_size,
                        **optimization_kwargs,
                    )
                    candidates = aug_candidates
                    del env_acq_func

            # Free memory.
            del acq_func, mll, model
            gc.collect()
            torch.cuda.empty_cache()

        print("candidates={}".format(candidates))
        # Get the new observations and update the data.
        new_x, new_y, new_z = eval_problem(candidates)
        # Note that the saved X is the normalized.
        X = torch.cat([X, new_x], dim=0)
        Y = torch.cat([Y, new_y], dim=0)
        Z = torch.cat([Z, new_z], dim=0)
        wall_time[i] = time() - start_time - compute_util_time

        init_util_time = time()
        new_z_eff = get_decision_input(
            X=new_z,
            input_transform_label=input_transform_label,
            decision_dim=decision_dim,
        )
        new_set_utility = torch.tensor([set_utility(new_z_eff)], **tkwargs)

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
                "Z": Z.cpu(),
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
        "Z": Z.cpu(),
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
