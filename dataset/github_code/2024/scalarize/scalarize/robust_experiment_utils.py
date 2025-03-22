#!/usr/bin/env python3

r"""
Utilities for the robust experiments.
"""
from copy import deepcopy

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    UnstandardizeAnalyticMultiOutputObjective,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.risk_measures import (
    Expectation,
    RiskMeasureMCObjective,
    VaR,
    WorstCase,
)
from botorch.models import ModelListGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.models.transforms.input import (
    AppendFeatures,
    InputPerturbation,
    InputTransform,
)
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler

from botorch.utils.gp_sampling import get_gp_samples
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import (
    draw_sobol_normal_samples,
    draw_sobol_samples,
    sample_simplex,
)
from botorch.utils.transforms import unnormalize
from torch import Tensor
from torch.distributions.beta import Beta
from torch.nn import Module

from scalarize.acquisition.monte_carlo import qNoisyExpectedImprovement

from scalarize.acquisition.robust_objectives import (
    ChiSquare,
    Entropic,
    MCVaR,
    TotalVariation,
)

from scalarize.experiment_utils import (
    get_acquisition_outcome_transform,
    get_problem,
    get_problem_outcome_transform,
    get_problem_reference_point,
    get_reference_point,
    get_scalarization_function,
    get_uncertainty_acquisition,
    initialize_model,
)

from scalarize.utils.scalarization_functions import ScalarizationFunction
from scalarize.utils.scalarization_objectives import (
    get_scalarized_samples,
    get_utility_mcobjective,
)

input_transform_dict = {
    "input": InputPerturbation,
    "feature": AppendFeatures,
}

robust_objectives_dict = {
    "ChiSquare": ChiSquare,
    "CVaR": MCVaR,
    "Expectation": Expectation,
    "MCVaR": MCVaR,
    "TotalVariation": TotalVariation,
    "KL": Entropic,
    "WorstCase": WorstCase,
    "VaR": VaR,
}

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


def generate_perturbed_initial_data(
    n: int,
    eval_problem: Callable[[Tensor], Tensor],
    bounds: Tensor,
    tkwargs: Dict[str, Any],
) -> Tuple[Tensor, Tensor]:
    r"""Generates the initial data for the robust experiments.

    Args:
        n: Number of training points.
        eval_problem: The callable used to evaluate the objective function.
        bounds: A `2 x d`-dim Tensor containing the bounds to generate the training
            points.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        train_x_observed: A `n x d_obs`-dim Tensor containing the observed inputs.
        train_y: A `n x M`-dim Tensor containing the training outputs.
        train_x_evaluated: A `n x d`-dim Tensor containing the evaluated inputs.
    """
    decision_input = (
        draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(-2).to(**tkwargs)
    )
    train_x_observed, train_y, train_x_evaluated = eval_problem(decision_input)
    return train_x_observed, train_y, train_x_evaluated


def get_problem_variables(
    function_name: str,
    tkwargs: Dict[str, Any],
    environment_kwargs: Dict[str, Any],
    input_transform_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    r"""Get the problem variables.

    Args:
        function_name: The name of the objective function.
        tkwargs: The tensor dtype to use and device to use.
        environment_kwargs: Arguments describing the environment.
        input_transform_kwargs: Arguments for the input transform.

    Returns:
        A dictionary containing the problem variables.

        - "base_function": The base function.
        - "decision_bounds": The `2 x decision_dim`-dim Tensor containing the bounds
            for the decision input.
        - "environment_bounds": The `2 x environment_dim`-dim Tensor containing the
            bounds for the environment variable.
        - "perturbation_bounds": The `2 x perturbation_dim`-dim Tensor containing the
            bounds for the perturbation. Note that this is set in the normalized
            space.
        - "controllable_bounds": The `2 x controllable_dim`-dim Tensor containing
            the bounds for the controllable input. This depends on the environment
            setting.
        - "standard_decision_bounds": The `2 x decision_dim`-dim Tensor containing
            the standardized decision bounds.
        - "standard_problem_bounds": The `2 x (decision_dim + environment_dim)`-dim
            Tensor containing the standardized problem bounds.
        - "standard_controllable_bounds": The `2 x controllable_dim`-dim Tensor
            containing the standardized problem bounds.
        - "feature_indices": The `feature_dim`-dim Tensor containing the indices for
            the features. Defaults to empty Tensor.
        - "decision_indices": The `decision_dim`-dim Tensor containing the indices
            for the decision variables.
        - "permute_indices": The `d`-dim Tensor containing the indices for
            re-organising the concatenated tensor:
            `X_concat = torch.cat([X_decision, X_feature], dim=-1)`.
        - "decision_dim": The dimension of the decision variable.
        - "feature_dim": The dimension of the feature variable.
        - "permutation_dim": The dimension of the permutations variable.
        - "controllable_dim": The dimension of the controllable variables.
    """
    input_transform_label = input_transform_kwargs["label"]
    environment_setting = environment_kwargs["setting"]

    base_function = get_problem(name=function_name, tkwargs=tkwargs)
    base_function.to(**tkwargs)

    # Get the dimension of the features.
    input_transform_kwargs.setdefault("feature_indices", [])
    feature_indices = torch.tensor(
        input_transform_kwargs["feature_indices"], dtype=torch.long
    )
    feature_dim = len(feature_indices)
    if len(feature_indices.unique()) != feature_dim:
        raise ValueError("Elements of `feature_indices` tensor must be unique!")

    decision_indices = torch.tensor(
        [i for i in range(base_function.dim) if i not in feature_indices],
        dtype=torch.long,
    )

    # get the indices to reconstruct the input `X` from the concatenated input
    # `X_concat = torch.cat([X_decision, X_feature], dim=-1)`
    permute_indices = torch.tensor(
        [i for i in range(base_function.dim)], dtype=torch.long
    )
    if feature_dim != 0:
        for i, idx in enumerate(decision_indices):
            permute_indices[idx] = i
        for j, idx in enumerate(feature_indices):
            permute_indices[idx] = i + j + 1

    # Need to pass a list of indices in order
    if feature_dim == 0 and input_transform_label == "feature":
        raise ValueError(
            "Need to specify the feature indices for the feature perturbation."
        )

    if input_transform_label == "input":
        decision_bounds = base_function.bounds
        environment_bounds = None
        controllable_bounds = base_function.bounds
    elif input_transform_label == "feature":
        decision_bounds = base_function.bounds[:, decision_indices]
        environment_bounds = base_function.bounds[:, feature_indices]

        controllable_bounds = base_function.bounds[:, decision_indices]
        if environment_setting == "simulated":
            controllable_bounds = torch.cat(
                [decision_bounds, environment_bounds], dim=-1
            )

    else:
        raise ValueError("Perturbation type is not supported!")

    decision_dim = decision_bounds.shape[-1]
    perturbation_dim = decision_dim if input_transform_label == "input" else feature_dim
    controllable_dim = controllable_bounds.shape[-1]

    standard_decision_bounds = torch.ones(2, decision_dim, **tkwargs)
    standard_decision_bounds[0] = 0
    standard_problem_bounds = torch.ones(2, base_function.dim, **tkwargs)
    standard_problem_bounds[0] = 0
    perturbation_bounds = torch.ones(2, perturbation_dim, **tkwargs)
    perturbation_bounds[0] = 0
    standard_controllable_bounds = torch.ones(2, controllable_dim, **tkwargs)
    standard_controllable_bounds[0] = 0

    problem_variables = {
        "base_function": base_function,
        "decision_bounds": decision_bounds,
        "environment_bounds": environment_bounds,
        "perturbation_bounds": perturbation_bounds,
        "controllable_bounds": controllable_bounds,
        "standard_decision_bounds": standard_decision_bounds,
        "standard_problem_bounds": standard_problem_bounds,
        "standard_controllable_bounds": standard_controllable_bounds,
        "feature_indices": feature_indices,
        "decision_indices": decision_indices,
        "permute_indices": permute_indices,
        "decision_dim": decision_dim,
        "feature_dim": feature_dim,
        "perturbation_dim": perturbation_dim,
        "controllable_dim": controllable_dim,
    }

    return problem_variables


def get_decision_input(
    X: Tensor,
    input_transform_label: str,
    decision_dim: Tensor,
) -> Tensor:
    r"""Get the decision input.

    Args:
        X: A `batch_shape x q x d`-dim input.
        input_transform_label: The label for the input transformation.
        decision_dim: The number of decision variables.

    Returns:
        An `batch_shape x q x decision_dim`-dim tensor of the decision inputs.
    """
    if input_transform_label == "input":
        return X
    elif input_transform_label == "feature":
        return X[..., 0:decision_dim]
    else:
        raise ValueError("The input transform is not supported!")


def get_perturbations(
    n_w: int,
    perturbation_kwargs: Dict[str, Any],
    tkwargs: Dict[str, Any],
    **kwargs,
) -> Tensor:
    r"""Generate an `n_w x perturbation_dim`-dim tensor of perturbations.

    NOTE: The perturbation settings are set in the normalized space.

    Args:
        n_w: Number of perturbations to generate.
        perturbation_kwargs: Arguments that are needed to set the perturbations.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        A `n_w x perturbation_dim`-dim tensor of perturbations.
    """
    method = perturbation_kwargs["method"]
    perturbation_dim = perturbation_kwargs["dimension"]
    bounds = perturbation_kwargs["bounds"]

    if method == "uniform":
        # Uniform(mean - delta, mean + delta).
        mean = perturbation_kwargs.get("mean")
        if mean is None:
            raise ValueError(f"mean is required for {method} perturbations.")
        mean = torch.tensor(mean, **tkwargs)

        delta = perturbation_kwargs.get("delta")
        if delta is None:
            raise ValueError(f"delta is required for {method} perturbations.")
        delta = torch.tensor(delta, **tkwargs)

        base_samples = draw_sobol_samples(bounds=bounds, n=n_w, q=1)
        perturbations = mean + (base_samples.squeeze(1) - 0.5) * (2 * delta)
    elif method == "truncated-normal":
        # TruncatedNormal(loc=mean, scale=std_dev, min=0.0, max=1.0).

        mean = perturbation_kwargs.get("mean")
        if mean is None:
            raise ValueError(f"mean is required for {method} perturbations.")
        mean = torch.tensor(mean, **tkwargs)

        std_dev = perturbation_kwargs.get("std_dev")
        if std_dev is None:
            raise ValueError(f"std_dev is required for {method} perturbations.")
        std_dev = torch.tensor(std_dev, **tkwargs)

        base_samples = draw_sobol_normal_samples(d=perturbation_dim, n=n_w, **tkwargs)
        perturbations = torch.clamp(mean + base_samples * std_dev, 0.0, 1.0)
    elif method == "beta":
        # Beta(shape=alpha, shape=beta).
        alpha = perturbation_kwargs.get("alpha")
        if alpha is None:
            raise ValueError(f"alpha is required for {method} perturbations.")
        alpha = torch.tensor(alpha, **tkwargs)

        beta = perturbation_kwargs.get("beta")
        if beta is None:
            raise ValueError(f"beta is required for {method} perturbations.")
        beta = torch.tensor(beta, **tkwargs)
        random_variable = Beta(alpha, beta)
        perturbations = random_variable.rsample(sample_shape=(n_w,))

    else:
        raise ValueError(f"Unknown method: {method}!")

    return perturbations


def get_robust_objective(
    n_w: int,
    robust_kwargs: Dict[str, Any],
) -> Callable[[Tensor], Tensor]:
    r"""Compute the robust objective.

    Args:
        n_w: The number of perturbations.
        robust_kwargs: The arguments for the robust objective.

    Return:
        A function that when given a Tensor of shape `num_configs x (q x n_w) x M`
            returns the robust value Tensor of shape `num_configs x q`.
    """
    label = robust_kwargs["label"]
    if label in robust_objectives_dict.keys():
        other_kwargs = {k: v for k, v in robust_kwargs.items() if k not in ["label"]}
        return robust_objectives_dict[label](n_w=n_w, **other_kwargs)
    else:
        raise ValueError(f"Unknown robust objective: {label}!")


def get_input_transform(
    input_transform_kwargs: Dict[str, Any],
    perturbation_set: Tensor,
) -> InputTransform:
    r"""Compute the input transform.

    Args:
        input_transform_kwargs: The arguments for the input transform.
        perturbation_set: A `n_w x perturbation_dim`-dim tensor of perturbations.

    Return:
        An input transform in training mode.
    """
    input_transform_label = input_transform_kwargs["label"]
    other_kwargs = {
        k: v
        for k, v in input_transform_kwargs.items()
        if k not in ["label", "feature_indices"]
    }
    if input_transform_label == "input":
        input_transform = InputPerturbation(
            perturbation_set=perturbation_set, **other_kwargs
        )
    elif input_transform_label == "feature":
        input_transform = AppendFeatures(feature_set=perturbation_set, **other_kwargs)
    else:
        raise ValueError(
            f"The input transform '{input_transform_label}' is not supported!"
        )

    return input_transform


class RobustSetUtility(Module):
    r"""A helper class that computes the Monte Carlo approximation of the robust set
    utility function:

    `U(f(X)) = E_{p(theta)}[max_{x in X} inf_q E_{q(w)}[s_{theta}(f(x, w))]]`,

    where `X` is a finite set and `s_{theta}` is a scalarization function
    parameterised by `theta`.
    """

    def __init__(
        self,
        eval_problem: Callable[[Tensor], Tensor],
        scalarization_fn: ScalarizationFunction,
        outcome_transform: OutcomeTransform,
        input_transform: InputTransform,
        robust_objective: RiskMeasureMCObjective,
    ) -> None:
        r"""Compute a Monte Carlo estimate of the robust set utility.

        Args:
            eval_problem: The true function without any input or observation noise.
            scalarization_fn: The scalarization function.
            outcome_transform: The outcome transform.
            input_transform: The input transform.
            robust_objective: The robust objective.
        """
        super().__init__()

        self.eval_problem = eval_problem
        self.scalarization_fn = scalarization_fn
        self.outcome_transform = outcome_transform
        self.input_transform = input_transform
        self.robust_objective = robust_objective
        self.best_values = None
        self.best_inputs = None

    def forward(self, new_X: Tensor) -> float:
        r"""Calculate the resulting Monte Carlo approximation of the robust set
        utility by including `new_X` into the set.

        Args:
            new_X: `q x d`-dim tensor of candidate points.

        Returns:
            The estimate of the robust set utility of all points evaluated so far up.
        """
        # `(num_perturbations * q) x num_scalar`
        scalarized_objectives = get_scalarized_samples(
            Y=self.eval_problem(self.input_transform(new_X)),
            scalarization_fn=self.scalarization_fn,
            outcome_transform=self.outcome_transform,
        )

        # `num_scalar x (num_perturbations * q) x 1`
        scalarized_objectives = scalarized_objectives.movedim(-1, 0).unsqueeze(-1)
        # `num_scalar x q`
        robust_values = self.robust_objective(scalarized_objectives)
        # `num_scalar`
        best_values, best_indices = robust_values.max(dim=-1)

        # `num_scalar x d`
        best_inputs = new_X[best_indices]

        if self.best_values is None:
            self.best_values = best_values
            self.best_inputs = best_inputs
        else:
            all_best_values = torch.stack([self.best_values, best_values], dim=-1)
            all_best_values, all_best_indices = torch.max(all_best_values, dim=-1)
            self.best_values = all_best_values

            self.best_inputs[all_best_indices.bool()] = best_inputs[
                all_best_indices.bool()
            ]

        # Compute the mean over the scalarization parameters.
        return self.best_values.mean(dim=0)


def get_robust_set_utility(
    function_name: str,
    scalarization_kwargs: Dict[str, Any],
    util_kwargs: Dict[str, Any],
    sampling_kwargs: Dict[str, Any],
    robust_kwargs: Dict[str, Any],
    environment_kwargs: Dict[str, Any],
    input_transform_kwargs: Dict[str, Any],
    perturbation_kwargs: Dict[str, Any],
    tkwargs: Dict[str, Any],
    estimate_utility: bool = False,
    data: Dict[str, Any] = None,
    model_kwargs: Dict[str, Any] = None,
    acq_kwargs: Dict[str, Any] = None,
    seed: int = 0,
) -> RobustSetUtility:
    r"""Get the robust set utility.

    Args:
        function_name: The name of the objective function.
        scalarization_kwargs: Arguments for the scalarization function.
        util_kwargs: Arguments for the utility.
        sampling_kwargs: Arguments used to determine the sampling used in the
            scalarization function and perturbations.
        robust_kwargs: Arguments for the robust objective.
        environment_kwargs: Arguments for the environment.
        input_transform_kwargs: Arguments for the input transform.
        perturbation_kwargs: Arguments for the perturbations.
        tkwargs: Arguments for tensors, dtype and device.
        estimate_utility: If True, then we estimate the outcome transform.
        data: The data that is used to estimate the utility.
        model_kwargs: The arguments to fit the model.
        acq_kwargs: The arguments for the acquisition functions.
        seed: The default seed.

    Returns:
        The set utility.
    """
    # Get the problem variables
    problem_variables = get_problem_variables(
        function_name=function_name,
        tkwargs=tkwargs,
        environment_kwargs=environment_kwargs,
        input_transform_kwargs=input_transform_kwargs,
    )
    base_function = problem_variables["base_function"]
    permute_indices = problem_variables["permute_indices"]

    # Get dimension and bounds.
    perturbation_dim = problem_variables["perturbation_dim"]
    perturbation_bounds = problem_variables["perturbation_bounds"]
    perturbation_kwargs.setdefault("dimension", perturbation_dim)
    perturbation_kwargs.setdefault("bounds", perturbation_bounds)

    bounds = base_function.bounds
    num_objectives = base_function.num_objectives

    # Define the perfect evaluation.
    def eval_problem_noiseless(X: Tensor) -> Tensor:
        X = unnormalize(X[..., permute_indices], bounds)
        fX = base_function.evaluate_true(X)
        Y = -fX if base_function.negate else fX
        return Y

    num_perturbations = perturbation_kwargs["num_perturbations"]

    # Ensure consistency of set utility performance metric across seeds by using same
    # Monte Carlo samples.

    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    perturbation_set = get_perturbations(
        n_w=num_perturbations,
        perturbation_kwargs=perturbation_kwargs,
        tkwargs=tkwargs,
    )

    # Note that we put input transform in evaluation mode.
    input_transform = get_input_transform(
        input_transform_kwargs=input_transform_kwargs,
        perturbation_set=perturbation_set,
    ).eval()

    robust_objective = get_robust_objective(
        n_w=num_perturbations,
        robust_kwargs=robust_kwargs,
    )

    if estimate_utility:

        # Fit the model.
        mll, model = initialize_model(
            train_x=data["X"],
            train_y=data["Y"],
            **model_kwargs,
        )
        fit_gpytorch_mll(mll)

        # Note that if `ignore_perturbations=True`, then this method does not
        # currently support model estimates of the outcome transform. Therefore,
        # we have to use a data-based estimate instead.
        outcome_transform, _ = get_acquisition_outcome_transform(
            name=function_name,
            model=model,
            scalarization_kwargs=scalarization_kwargs,
            acq_kwargs=acq_kwargs,
            util_kwargs=util_kwargs,
            tkwargs=tkwargs,
            X_baseline=data["X"],
            Y_baseline=data["Y"],
            bounds=None,
        )
    else:
        outcome_transform = get_problem_outcome_transform(
            name=function_name,
            scalarization_kwargs=scalarization_kwargs,
            util_kwargs=util_kwargs,
            tkwargs=tkwargs,
        )

    reference_point = get_problem_reference_point(
        name=function_name,
        scalarization_kwargs=scalarization_kwargs,
        util_kwargs=util_kwargs,
        tkwargs=tkwargs,
    )

    util_scalarization_fn = get_scalarization_function(
        num_objectives=num_objectives,
        name=function_name,
        scalarization_kwargs=scalarization_kwargs,
        util_kwargs=util_kwargs,
        sampling_kwargs=sampling_kwargs,
        outcome_transform=outcome_transform,
        tkwargs=tkwargs,
        ref_points=reference_point,
    )

    set_utility = RobustSetUtility(
        eval_problem=eval_problem_noiseless,
        scalarization_fn=util_scalarization_fn,
        outcome_transform=outcome_transform,
        input_transform=input_transform,
        robust_objective=robust_objective,
    )

    torch.random.set_rng_state(old_state)

    return set_utility


def get_uncertainty_penalty(
    X: Tensor,
    model: Model,
    objective_bounds: Tensor,
    tkwargs: Dict[str, Any],
) -> Tensor:
    r"""Get the uncertainty penalty values.

    Args:
        X: A `batch_shape x q x d`-dim Tensor.
        model: The model including input transform.
        objective_bounds: The `2 x M`-dim Tensor containing the estimated bounds for
            the objectives.
        tkwargs: The tensor dtype to use and device to use.

    Returns:
        A `batch_shape x (q * n_w)`-dim Tensor containing the uncertainty values.
    """
    num_objectives = objective_bounds.shape[-1]
    normalize_otf = UnstandardizeAnalyticMultiOutputObjective(
        Y_mean=torch.zeros(num_objectives, **tkwargs),
        Y_std=1 / (objective_bounds[1] - objective_bounds[0]),
    )
    posterior = normalize_otf(model.posterior(X))
    average_trace = torch.mean(posterior.variance, dim=-1)

    return torch.sqrt(average_trace)


def get_robust_utility_mc_objective(
    scalarization_objective: GenericMCObjective,
    robust_objective: GenericMCObjective,
    include_augmentation: bool = False,
    model: Optional[Model] = None,
    objective_bounds: Optional[Tensor] = None,
    beta: float = 0.0,
    tkwargs: Optional[Dict[str, Any]] = None,
) -> GenericMCObjective:
    r"""Get the robust utility objective.

    Args:
        scalarization_objective: The scalarization objective.
        robust_objective: The robust objective.
        include_augmentation: If True, we include an augmentation term to the
            scalarized objective. This comes in the form of an uncertainty penalty.
        model: The model used to compute the augmentation term.
        objective_bounds: A `2 x M`-dim Tensor containing the estimated
            objective bounds, which is used to compute the augmentation term.
        beta: The multiplier for the augmentation term.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        The robust utility objective.
    """
    if include_augmentation:

        def robust_mc_objectives(Y: Tensor, X: Tensor) -> Tensor:
            # `Y` has shape `num_mc_samples x batch_shape x (q x n_w) x M`
            # `penalty` has shape `batch_shape x q`
            penalty = get_uncertainty_penalty(
                X=X,
                model=model,
                objective_bounds=objective_bounds,
                tkwargs=tkwargs,
            )
            # `s_obj` has shape
            # `(num_mc_samples x num_scalar) x batch_shape x (q x n_w)`
            s_obj = scalarization_objective(Y) + beta * penalty
            # `r_obj` has shape `(num_mc_samples x num_scalar) x batch_shape x q`
            r_obj = robust_objective(s_obj.unsqueeze(-1))
            return r_obj

    else:

        def robust_mc_objectives(Y: Tensor, X: Tensor) -> Tensor:
            # `Y` has shape `num_mc_samples x batch_shape x (q x n_w) x M`
            # `s_obj` has shape
            # `(num_mc_samples x num_scalar) x batch_shape x (q x n_w)`
            s_obj = scalarization_objective(Y)
            # `r_obj` has shape `(num_mc_samples x num_scalar) x batch_shape x q`
            r_obj = robust_objective(s_obj.unsqueeze(-1))
            return r_obj

    return GenericMCObjective(robust_mc_objectives)


def get_augmented_utility_mc_objective(
    scalarization_objective: GenericMCObjective,
    model: Optional[Model] = None,
    objective_bounds: Optional[Tensor] = None,
    beta: float = 0.0,
    tkwargs: Optional[Dict[str, Any]] = None,
) -> GenericMCObjective:
    r"""Get the augmented utility objective.

    Args:
        scalarization_objective: The scalarization objective.
        model: The model used to compute the augmentation term.
        objective_bounds: A `2 x M`-dim Tensor containing the estimated
            objective bounds, which is used to compute the augmentation term.
        beta: The multiplier for the augmentation term.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        The augmented utility objective.
    """

    def augmented_mc_objectives(Y: Tensor, X: Tensor) -> Tensor:
        # `Y` has shape `num_mc_samples x batch_shape x (q x n_w) x M`
        # `penalty` has shape `batch_shape x q`
        penalty = get_uncertainty_penalty(
            X=X,
            model=model,
            objective_bounds=objective_bounds,
            tkwargs=tkwargs,
        )

        # `s_obj` has shape `(num_mc_samples x num_scalar) x batch_shape x q`
        s_obj = scalarization_objective(Y) + beta * penalty
        return s_obj

    return GenericMCObjective(augmented_mc_objectives)


def get_model_without_input_transform(
    model: Model,
) -> Model:
    r"""Get a copy of a model without input transformations.

    Args:
        model: The model.

    Returns:
        The model without input transformations..
    """
    # Get copy of model without input transform.
    model_without_input_transform = deepcopy(model)
    if not isinstance(model, ModelListGP):
        intf = getattr(model, "input_transform", None)
        if intf is not None:
            del model_without_input_transform.input_transform
    else:
        for model_i in model_without_input_transform.models:
            intf = getattr(model_i, "input_transform", None)
            if intf is not None:
                del model_i.input_transform

    return model_without_input_transform


def get_acquisition_function_robust(
    name: str,
    iteration: int,
    num_iterations: int,
    label: str,
    model: Model,
    X_baseline: Tensor,
    scalarization_kwargs: Dict[str, Any],
    sampling_kwargs: Dict[str, Any],
    acq_kwargs: Dict[str, Any],
    util_kwargs: Dict[str, Any],
    input_transform_kwargs: Dict[str, Any],
    robust_kwargs: Dict[str, Any],
    perturbation_kwargs: Dict[str, Any],
    tkwargs: Dict[str, Any],
    Y_baseline: Optional[Tensor] = None,
    bounds: Optional[Tensor] = None,
) -> AcquisitionFunction:
    r"""Initialize the acquisition function for the robust problem.

    Args:
        name: The name of the problem function.
        iteration: The iteration number.
        num_iterations: The number of iterations.
        label: The name of the acquisition function.
        model: The model.
        X_baseline: A `num_baseline x d`-dim Tensor containing the baseline inputs.
        scalarization_kwargs: Arguments used to determine the scalarization function.
        sampling_kwargs: Arguments for the sampling used to set the scalarization
            function.
        acq_kwargs: Arguments for the acquisition functions.
        util_kwargs: Arguments for the utility.
        input_transform_kwargs: Arguments for the input transform.
        robust_kwargs: Arguments for the robust objective.
        perturbation_kwargs: Arguments for the perturbations.
        tkwargs: The tensor dtype to use and device to use.
        Y_baseline: A `num_baseline x M`-dim Tensor containing the baseline outputs.
        bounds: A `2 x d`-dim Tensor containing the bounds of the inputs. Note that
            this is the standardised bounds.

    Returns:
        The acquisition function.
    """
    num_objectives = model.num_outputs
    estimate_bounds = acq_kwargs.get("estimate_the_bounds", True)
    use_input_transform = label in robust_acquisition_list
    num_perturbations = perturbation_kwargs["num_perturbations"]
    generate_new_perturbations = acq_kwargs.get("generate_new_perturbations", False)
    is_augmented = True if label in augumented_acquisition_list else False
    ucb_beta = acq_kwargs.get("beta", 2.0)
    augmented_beta = 0.0
    if is_augmented:
        augmented_beta = acq_kwargs.get("beta", 2.0)
        ucb_beta = 0.0

    outcome_transform, objective_bounds = get_acquisition_outcome_transform(
        name=name,
        model=model,
        scalarization_kwargs=scalarization_kwargs,
        acq_kwargs=acq_kwargs,
        util_kwargs=util_kwargs,
        tkwargs=tkwargs,
        X_baseline=X_baseline,
        Y_baseline=Y_baseline,
        bounds=bounds,
    )

    # Set the baseline objective values to be the objective ranges if available.
    Y_baseline_or_bounds = objective_bounds
    if estimate_bounds:
        if label in ["nparego"]:
            with torch.no_grad():
                Y_baseline_or_bounds = model.posterior(X_baseline).mean
        else:
            Y_baseline_or_bounds = Y_baseline

    if use_input_transform:
        if generate_new_perturbations:
            perturbation_set = get_perturbations(
                n_w=num_perturbations,
                perturbation_kwargs=perturbation_kwargs,
                tkwargs=tkwargs,
            )

            input_transform = get_input_transform(
                input_transform_kwargs=input_transform_kwargs,
                perturbation_set=perturbation_set,
            )

            if not isinstance(model, ModelListGP):
                model.input_transform = input_transform
            else:
                for model_i in model.models:
                    model_i.input_transform = input_transform

    if "-ts" in label:
        acq_model = get_gp_samples(
            model=model,
            num_outputs=model.num_outputs,
            n_samples=1,
        )
        num_samples = 1
        cache_root = False
        # NOTE: we get an input transform runtime warning for the robust acquisition
        # functions, when we try to compute the posterior of the baseline points in
        # the `prune_inferior_points` method.
        prune_baseline = True
        sampler = StochasticSampler(sample_shape=torch.Size([num_samples]))
    elif "-ucb" in label:

        def f_ucb(X):
            posterior = model.posterior(X)
            mean = posterior.mean
            variance = posterior.variance
            return mean + ucb_beta * torch.sqrt(variance)

        acq_model = GenericDeterministicModel(f=f_ucb, num_outputs=model.num_outputs)
        num_samples = 1
        cache_root = False
        prune_baseline = True
        sampler = StochasticSampler(sample_shape=torch.Size([num_samples]))
    else:
        acq_model = model
        num_samples = acq_kwargs["num_samples"]
        cache_root = True
        prune_baseline = True
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))

    if ("eui" in label) or ("resi" in label):
        if "resi" in label:
            sampling_kwargs["num_weights"] = 1

        scalarization_fn = get_scalarization_function(
            name=name,
            num_objectives=num_objectives,
            scalarization_kwargs=scalarization_kwargs,
            util_kwargs=util_kwargs,
            sampling_kwargs=sampling_kwargs,
            outcome_transform=outcome_transform,
            tkwargs=tkwargs,
            model=model,
            X_baseline=X_baseline,
        )

        util_obj = get_utility_mcobjective(
            scalarization_fn=scalarization_fn,
            outcome_transform=outcome_transform,
        )

        if "robust" in label:
            robust_obj = get_robust_objective(
                n_w=num_perturbations,
                robust_kwargs=robust_kwargs,
            )

            robust_mc_obj = get_robust_utility_mc_objective(
                scalarization_objective=util_obj,
                robust_objective=robust_obj,
                include_augmentation=is_augmented,
                model=model,
                objective_bounds=objective_bounds,
                beta=augmented_beta,
                tkwargs=tkwargs,
            )
            mc_obj = robust_mc_obj
        else:
            if is_augmented:
                mc_obj = get_augmented_utility_mc_objective(
                    scalarization_objective=util_obj,
                    model=model,
                    objective_bounds=objective_bounds,
                    beta=augmented_beta,
                    tkwargs=tkwargs,
                )
            else:
                mc_obj = util_obj

        acq = qNoisyExpectedImprovement(
            model=acq_model,
            objective=mc_obj,
            X_baseline=X_baseline,
            sampler=sampler,
            prune_baseline=prune_baseline,
            cache_root=cache_root,
        )

    elif "ehvi" in label:
        ref_point = get_reference_point(
            Y_baseline=Y_baseline_or_bounds,
            use_utopia=False,
        ).squeeze(0)

        if "nehvi" in label:
            acq = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point.tolist(),
                X_baseline=X_baseline,
                prune_baseline=prune_baseline,
                sampler=sampler,
                cache_root=cache_root,
            )

        else:
            partitioning = FastNondominatedPartitioning(
                ref_point=ref_point, Y=Y_baseline
            )

            acq = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point.tolist(),
                partitioning=partitioning,
                sampler=sampler,
            )

    elif "parego" in label:
        weights = sample_simplex(model.num_outputs, **tkwargs).squeeze()

        objective = GenericMCObjective(
            get_chebyshev_scalarization(weights=weights, Y=Y_baseline_or_bounds)
        )

        if "nparego" in label:
            acq = qNoisyExpectedImprovement(
                model=model,
                objective=objective,
                X_baseline=X_baseline,
                sampler=sampler,
                prune_baseline=prune_baseline,
                cache_root=cache_root,
            )
        else:
            acq = qExpectedImprovement(
                model=model,
                objective=objective,
                best_f=max(objective(Y_baseline)),
                sampler=sampler,
            )
    else:
        raise ValueError("The label is not supported!")

    return acq


def get_environment_acquisition(
    model: Model,
    name: str,
    scalarization_kwargs: Dict[str, Any],
    acq_kwargs: Dict[str, Any],
    util_kwargs: Dict[str, Any],
    tkwargs: Dict[str, Any],
    X_baseline: Tensor,
    Y_baseline: Tensor,
    bounds: Tensor,
) -> AcquisitionFunction:
    r"""Construct the environmental acquisition function.

    Args:
        model: A fitted model.
        name: The name of the problem function.
        scalarization_kwargs: The arguments used to determine the scalarization
            function.
        acq_kwargs: The arguments for the acquisition functions.
        util_kwargs: Arguments for the utility.
        tkwargs: The tensor dtype to use and device to use.
        X_baseline: A `num_baseline x d`-dim Tensor containing the baseline inputs.
        Y_baseline: A `num_baseline x d`-dim Tensor containing the baseline outputs.
        bounds: A `2 x d`-dim Tensor containing the bounds of the inputs. Note that
            this is the standardised bounds.

    Return:
        The uncertainty acquisition function.
    """

    model_without_input_transform = get_model_without_input_transform(model=model)

    _, objective_bounds = get_acquisition_outcome_transform(
        name=name,
        model=model_without_input_transform,
        scalarization_kwargs=scalarization_kwargs,
        acq_kwargs=acq_kwargs,
        util_kwargs=util_kwargs,
        tkwargs=tkwargs,
        X_baseline=X_baseline,
        Y_baseline=Y_baseline,
        bounds=bounds,
    )

    model_without_input_transform = get_model_without_input_transform(model=model)

    uncertainty_acq = get_uncertainty_acquisition(
        model=model_without_input_transform,
        objective_bounds=objective_bounds,
        tkwargs=tkwargs,
    )

    return uncertainty_acq
