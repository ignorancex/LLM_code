#!/usr/bin/env python3

r"""
The utilities that are used to run the experiments.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from botorch.acquisition.penalized import PenalizedAcquisitionFunction
from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import (
    ChainedOutcomeTransform,
    OutcomeTransform,
    Standardize,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.test_functions.multi_objective import (
    BraninCurrin,
    CarSideImpact,
    DTLZ2,
    GMM,
    Penicillin,
    VehicleSafety,
    ZDT1,
    ZDT3,
)
from botorch.utils.gp_sampling import get_gp_samples
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import draw_sobol_samples, sample_simplex
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor
from torch.nn import Module

from scalarize.acquisition.analytic import Uncertainty
from scalarize.acquisition.monte_carlo import qNoisyExpectedImprovement

from scalarize.data import get_rocket_pf, get_vehicle_pf, get_zdt3_pf
from scalarize.models.transforms.outcome import GaussianQuantile, Normalize
from scalarize.test_functions.multi_objective import (
    CabDesign,
    FourBarTrussDesign,
    MarineDesign,
    ResourcePlanning,
    RocketInjector,
    WeldedBeam,
)
from scalarize.utils.sampling import (
    sample_ordered_simplex,
    sample_ordered_unit_vector,
    sample_permutations,
)
from scalarize.utils.scalarization_functions import (
    ChebyshevScalarization,
    HypervolumeScalarization,
    KSScalarization,
    LengthScalarization,
    LinearScalarization,
    LpScalarization,
    ScalarizationFunction,
)
from scalarize.utils.scalarization_objectives import (
    get_scalarized_samples,
    get_utility_mcobjective,
)
from scalarize.utils.transformations import (
    estimate_bounds,
    get_baseline_candidates,
    get_kernel_density_statistics,
)

scalarization_functions_dict = {
    "d1": ChebyshevScalarization,
    "igd": LpScalarization,
    "hypervolume": HypervolumeScalarization,
    "length": LengthScalarization,
    "linear": LinearScalarization,
    "ks": KSScalarization,
    "r2": ChebyshevScalarization,
}

weight_distribution_dict = {
    "simplex": sample_ordered_simplex,
    "unit-vector": sample_ordered_unit_vector,
}

problem_dict = {
    "bc": BraninCurrin,
    "cab": CabDesign,
    "carside": CarSideImpact,
    "dtlz2_3": DTLZ2,
    "dtlz2_4": DTLZ2,
    "dtlz2_5": DTLZ2,
    "dtlz2_6": DTLZ2,
    "dtlz2_7": DTLZ2,
    "dtlz2_8": DTLZ2,
    "gmm2": GMM,
    "gmm3": GMM,
    "gmm4": GMM,
    "marine": MarineDesign,
    "penicillin": Penicillin,
    "planning": ResourcePlanning,
    "rocket": RocketInjector,
    "truss": FourBarTrussDesign,
    "vehicle": VehicleSafety,
    "beam": WeldedBeam,
    "zdt1_4": ZDT1,
    "zdt3_2": ZDT3,
    "zdt3_4": ZDT3,
}

bounds_dict = {
    "bc": [[-306, -13.8], [-0.39, -1.19]],
    "cab": [
        [-42.0, -1.05, -0.95, -0.8, -1.5, -1.15, -1.11, -1.05, -1.05],
        [-16.0, -0.25, -0.35, -0.33, -0.64, -0.66, -0.89, -0.83, -0.82],
    ],
    "carside": [[-41, -4.5, -13.1, -12.1], [-17.5, -3.6, -11, -0]],
    "dtlz2_3": [[-2, -2], [-0, -0]],
    "dtlz2_4": [[-2, -2], [-0, -0]],
    "dtlz2_5": [[-2, -2], [-0, -0]],
    "dtlz2_6": [[-2, -2], [-0, -0]],
    "dtlz2_7": [[-2, -2], [-0, -0]],
    "dtlz2_8": [[-2, -2], [-0, -0]],
    "gmm2": [[-0, -0], [0.706, 0.706]],
    "gmm3": [[-0, -0, -0], [0.706, 0.706, 0.90]],
    "gmm4": [[-0, -0, -0, -0], [0.706, 0.706, 0.90, 0.90]],
    "marine": [[265, -20000, -30500, -15], [2550, -4000, -2100, -0]],
    "penicillin": [[-0, -83, -570], [20, -0, -1]],
    "planning": [
        [-83100.0, -1350.0, -2860000.0, -16100000.0, -355000.0, -98200.0],
        [-63800.0, -30.0, -285000.0, -183800.00, -15.0, -0.0],
    ],
    "rocket": [[-1.0, -1.25, -1.1], [-0.01, -0.005, 0.41]],
    "truss": [[-3000.0, -0.05], [-1240.0, -0.0]],
    "vehicle": [[-1705, -11.7, -0.26], [-1650, -6.1, -0.04]],
    "beam": [[-330.0, -17500.0, -400000000.0], [-0.015, -0.0004, 0.0]],
    "zdt1_4": [[-1, -10], [-0, -0]],
    "zdt3_2": [[-1, -1], [-0, 0.7725]],
    "zdt3_4": [[-1, -1], [-0, 0.7725]],
}

problem_kwargs_dict = {
    "gmm2": {"num_objectives": 2},
    "gmm3": {"num_objectives": 3},
    "gmm4": {"num_objectives": 4},
    "dtlz2_3": {"dim": 3},
    "dtlz2_4": {"dim": 4},
    "dtlz2_5": {"dim": 5},
    "dtlz2_6": {"dim": 6},
    "dtlz2_7": {"dim": 7},
    "dtlz2_8": {"dim": 8},
    "zdt1_4": {"dim": 4},
    "zdt3_2": {"dim": 2},
    "zdt3_4": {"dim": 4},
}

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


def generate_initial_data(
    n: int,
    eval_problem: Callable[[Tensor], Tensor],
    bounds: Tensor,
    tkwargs: Dict[str, Any],
) -> Tuple[Tensor, Tensor]:
    r"""Generates the initial data for the experiments.

    Args:
        n: Number of training points.
        eval_problem: The callable used to evaluate the objective function.
        bounds: A `2 x d`-dim Tensor containing the bounds to generate the training
            points.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        train_x: A `n x d`-dim Tensor containing the training inputs.
        train_y: A `n x M`-dim Tensor containing the training outputs.
    """
    train_input = draw_sobol_samples(bounds=bounds, n=n, q=1).squeeze(-2).to(**tkwargs)
    train_x, train_y = eval_problem(train_input)
    return train_x, train_y


def initialize_model(
    train_x: Tensor,
    train_y: Tensor,
    use_model_list: bool = True,
    use_fixed_noise: bool = False,
    input_transform: Optional[InputTransform] = None,
) -> Tuple[
    Union[ExactMarginalLogLikelihood, SumMarginalLogLikelihood],
    Union[FixedNoiseGP, SingleTaskGP, ModelListGP],
]:
    r"""Constructs the model and its MLL.

    Args:
        train_x: A `n x d`-dim Tensor containing the training inputs.
        train_y: A `n x M`-dim Tensor containing the training outputs.
        use_model_list: If True, returns a ModelListGP with models for each outcome.
        use_fixed_noise: If True, assumes noise-free outcomes and uses FixedNoiseGP.
        input_transform: An optional input transform.

    Returns:
        The MLL and the model. Note: the model is not trained!
    """
    base_model_class = FixedNoiseGP if use_fixed_noise else SingleTaskGP
    # Put input transform in training mode.
    if input_transform is not None:
        input_transform = input_transform.train()

    # define models for objective and constraint
    if use_fixed_noise:
        train_Yvar = torch.full_like(train_y, 1e-7) * train_y.std(dim=0).pow(2)

    if use_model_list:
        model_kwargs = []
        for i in range(train_y.shape[-1]):
            model_kwargs.append(
                {
                    "train_X": train_x,
                    "train_Y": train_y[..., i : i + 1],
                    "outcome_transform": Standardize(m=1),
                    "input_transform": input_transform,
                }
            )
            if use_fixed_noise:
                model_kwargs[i]["train_Yvar"] = train_Yvar[..., i : i + 1]
        models = [base_model_class(**model_kwargs[i]) for i in range(train_y.shape[-1])]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model
    else:
        model_kwargs = {
            "train_X": train_x,
            "train_Y": train_y,
            "outcome_transform": Standardize(m=train_y.shape[-1]),
            "input_transform": input_transform,
        }
        if use_fixed_noise:
            model_kwargs["train_Yvar"] = train_Yvar

        model = base_model_class(**model_kwargs)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model


class SetUtility(Module):
    r"""A helper class that computes the Monte Carlo approximation of the set
    utility function:

    `U(f(X)) = E_{p(theta)}[max_{x in X} s_{theta}(f(x))]`,

    where `X` is a finite set and `s_{theta}` is a scalarization function
    parameterised by `theta`.
    """

    def __init__(
        self,
        eval_problem: Callable[[Tensor], Tensor],
        scalarization_fn: ScalarizationFunction,
        outcome_transform: OutcomeTransform,
    ) -> None:
        r"""Compute a Monte Carlo estimate of the Bayes utility.

        Args:
            eval_problem: The true function without any input or observation noise.
            scalarization_fn: The scalarization function.
            outcome_transform: The outcome transform.
        """
        super().__init__()

        self.eval_problem = eval_problem
        self.scalarization_fn = scalarization_fn
        self.outcome_transform = outcome_transform
        self.best_values = None
        self.best_inputs = None

    def forward(self, new_X: Tensor) -> float:
        r"""Calculate the resulting Monte Carlo approximation of the set utility by
        including `new_X` into the set.

        Args:
            new_X: `q x d`-dim tensor of candidate points.

        Returns:
            The estimate of the set utility of all points evaluated so far up.
        """
        # `q x num_scalars`
        scalarized_objectives = get_scalarized_samples(
            Y=self.eval_problem(new_X),
            scalarization_fn=self.scalarization_fn,
            outcome_transform=self.outcome_transform,
        )
        # `num_scalars`
        best_values, best_indices = scalarized_objectives.max(dim=0)
        # `num_scalars x d`
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


def get_weights(
    num_objectives: int,
    label: str,
    num_weights: int,
    tkwargs: Dict[str, Any],
    ordered: bool = False,
    descending: bool = True,
    importance_order: Optional[List[int]] = None,
    distributional_preference: bool = False,
    rho: Optional[List[float]] = None,
) -> Tensor:
    r"""Sample from the weight distribution.

    Args:
        num_objectives: The number of objectives.
        label: The name of the weight distribution.
        num_weights: The number of weights.
        tkwargs: The dtype and device to use.
        ordered: If True, we sampled from the ordered weight distribution.
        descending: If True, we order in descending order else we order in
            ascending order.
        importance_order: This is a list containing the indices arranged in terms of
            importance from high to low. For example, if we have three objectives,
            and have the ordering obj_2 > obj_3 > obj_1, then we set
            `importance_order = [1, 2, 0]`.
        distributional_preference: If True, we sample the importance order using
            the permutation distribution.
        rho: This is a list containing the probability weights used to sample from
            the permutation distribution.

    Returns:
        A `num_weights x M`-Tensor containing the weights.
    """
    # Sample the weight vector.
    weight_distribution = weight_distribution_dict[label]

    # TODO: This currently only handles one preference. Maybe try to extend this to
    #   work for a list of preference orders.
    weights = weight_distribution(
        d=num_objectives,
        n=num_weights,
        ordered=ordered,
        descending=descending,
        **tkwargs,
    )
    if importance_order is not None:
        if len(importance_order) != num_objectives:
            raise ValueError(
                f"The order list is incorrect, expected {num_objectives} objectives "
                f"but got {len(importance_order)} indices!"
            )
        weights = weights[:, importance_order]

    if distributional_preference:
        if rho is None or len(rho) != num_objectives:
            raise ValueError(
                f"Need to specify a suitable list of {num_objectives} weights for "
                f"the distributional preference sampling."
            )

        permutations = sample_permutations(
            weights=torch.tensor(rho, **tkwargs),
            n=num_weights,
            **tkwargs,
        )
        weights = torch.row_stack(
            [weights[i, perm] for i, perm in enumerate(permutations)]
        )

    return weights


def get_bounds_estimate(
    model: Optional[Model] = None,
    X_baseline: Optional[Tensor] = None,
    Y_baseline: Optional[Tensor] = None,
) -> Tensor:
    r"""Estimate the bounds. Defaults to using the data estimate when available.

    Args:
        model: The model.
        X_baseline: A `n x d`-dim Tensor containing the observed values.
        Y_baseline: A `n x M`-dim Tensor containing the observed values.

    Returns:
        A `2 x M`-Tensor containing the estimated bounds.
    """
    # Over-estimate the utopia or nadir
    if Y_baseline is not None:
        # Add 10% overestimation like infer_reference_point
        bounds = estimate_bounds(
            Y_baseline=Y_baseline,
            eta=0.5,
            kappa=0.1,
        )

    else:
        bounds = estimate_bounds(
            model=model,
            X_baseline=X_baseline,
            kappa=3.0,
        )

    return bounds


def get_reference_point(
    model: Optional[Model] = None,
    X_baseline: Optional[Tensor] = None,
    Y_baseline: Optional[Tensor] = None,
    use_utopia: bool = True,
) -> Tensor:
    r"""Generate the reference point using the bounds estimate.

    Args:
        model: The model.
        X_baseline: A `n x d`-dim Tensor containing the observed values.
        Y_baseline: A `n x M`-dim Tensor containing the observed values.
        use_utopia: If True, the reference point is the utopia, else we consider the
            nadir.

    Returns:
        A `num_ref_points x M`-Tensor containing the reference points.
    """

    # Over-estimate the utopia or nadir
    bounds = get_bounds_estimate(
        model=model,
        X_baseline=X_baseline,
        Y_baseline=Y_baseline,
    )
    if use_utopia:
        ref_points = bounds[1:, :]
    else:
        ref_points = bounds[0:1, :]

    return ref_points


def get_preference_multiplier_transform(
    scalarization_kwargs: Dict[str, Any],
    util_kwargs: Dict[str, Any],
    tkwargs: Dict[str, Any],
) -> OutcomeTransform:
    r"""Get the preference multiplier transform.

    Args:
        scalarization_kwargs: Arguments for the scalarization function.
        util_kwargs: Arguments for the utility.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        The multiplier outcome transform.
    """
    preference_weights = util_kwargs.get("preference_weights", None)
    descending = scalarization_kwargs.get("descending", True)
    if preference_weights is None:
        # identity transform
        return Normalize()
    else:
        num_objectives = len(preference_weights)
        bounds = torch.zeros(2, num_objectives, **tkwargs)
        if descending:
            bounds[1] = 1.0 / torch.tensor(preference_weights, **tkwargs)
        else:
            bounds[1] = torch.tensor(preference_weights, **tkwargs)

    return Normalize(bounds=bounds)


def get_scalarization_function(
    name: str,
    num_objectives: int,
    scalarization_kwargs: Dict[str, Any],
    util_kwargs: Dict[str, Any],
    sampling_kwargs: Dict[str, Any],
    outcome_transform: OutcomeTransform,
    tkwargs: Dict[str, Any],
    weights: Optional[Tensor] = None,
    ref_points: Optional[Tensor] = None,
    model: Optional[Model] = None,
    X_baseline: Optional[Tensor] = None,
    Y_baseline: Optional[Tensor] = None,
) -> ScalarizationFunction:
    r"""Compute the scalarization function.

    Args:
        name: The name of the problem function.
        num_objectives: The number of objectives.
        scalarization_kwargs: Arguments for the scalarization function.
        util_kwargs: Arguments for the utility.
        sampling_kwargs: Arguments used to determine the Monte Carlo samples for the
            scalarization function.
        outcome_transform: The outcome transform.
        tkwargs: The tensor dtype to use and device to use.
        weights: A `num_weights x M`-Tensor containing the weights.
        ref_points: A `num_ref_points x M`-Tensor containing the reference points.
        model: The model.
        X_baseline: A `n x d`-dim Tensor containing the observed values.
        Y_baseline: A `n x M`-dim Tensor containing the observed values.

    Returns:
        The scalarization function.
    """

    scalarization_label = scalarization_kwargs["label"]
    s_fn = scalarization_functions_dict[scalarization_label]
    descending = scalarization_functions_dict.get("descending", True)
    transform_reference_point = sampling_kwargs.get("transform_reference_point", False)

    # Get other sampling arguments
    s_kwargs = {
        k: v
        for k, v in sampling_kwargs.items()
        if k not in ["transform_reference_point"]
    }

    # Get the weights
    if weights is None:
        if scalarization_label == "igd":
            # use vector of ones
            weights = torch.ones((1, num_objectives), **tkwargs)
        elif scalarization_label == "d1":
            weights = torch.ones((1, num_objectives), **tkwargs) / num_objectives
        elif scalarization_label == "ks":
            weights = None
        else:
            weights = get_weights(
                num_objectives=num_objectives,
                tkwargs=tkwargs,
                descending=descending,
                **s_kwargs,
            )

    # Get the reference point
    if ref_points is None:
        use_utopia = scalarization_kwargs.get("use_utopia", False)

        if scalarization_label == "igd" or scalarization_label == "d1":
            all_ref_points = get_problem_reference_point(
                name=name,
                scalarization_kwargs=scalarization_kwargs,
                util_kwargs=util_kwargs,
                tkwargs=tkwargs,
            )
            # sample uniformly from the ref points if `num_ref_points` given.
            num_ref_points = sampling_kwargs.get("num_ref_points", False)
            if num_ref_points:
                ref_points = all_ref_points[
                    torch.randint(0, len(all_ref_points), (num_ref_points,))
                ]
            else:
                ref_points = all_ref_points

        elif scalarization_label == "ks":
            ref_points = torch.zeros(2, 1, **tkwargs)
            ref_points[1] = 1.0

        else:
            # Note that this defaults to using the model estimate.
            ref_points = get_reference_point(
                model=model,
                X_baseline=X_baseline,
                Y_baseline=Y_baseline,
                use_utopia=use_utopia,
            )

    if transform_reference_point:
        ref_points, _ = outcome_transform(ref_points)

    # Get other scalarization function arguments
    s_fn_kwargs = {
        k: v
        for k, v in scalarization_kwargs.items()
        if k not in ["label", "use_utopia", "descending"]
    }

    if scalarization_label == "ks":
        # assumes that the reference point is of shape `2 x M`
        scalarization_fn = s_fn(
            utopia_points=ref_points[1, :], nadir_points=ref_points[0, :]
        )
    else:
        scalarization_fn = s_fn(weights=weights, ref_points=ref_points, **s_fn_kwargs)

    return scalarization_fn


def get_ei_multiplier(
    label: str,
    iteration: int,
    num_iterations: int,
    bounds: Tensor,
) -> float:
    r"""Get the expected utility improvement penalty parameter.

    Args:
        label: The label of the acquisition function.
        iteration: The iteration number.
        num_iterations: The total number of iterations.
        bounds: A `2 x M`-dim Tensor containing the bounds for the objectives.

    Returns:
        A non-negative float.
    """
    # get the `p` value after `rg-` or `thresh`
    if "rg" in label:
        p_vals = re.findall(r"%s(\d+\.\d+)" % "rg-", label)
        if len(p_vals) > 0:
            p = float(p_vals[0])
        else:
            p = 0.0
    elif "thresh" in label:
        p_vals = re.findall(r"%s(\d+\.\d+)" % "thresh-", label)
        if len(p_vals) > 0:
            p = float(p_vals[0])
        else:
            p = 0.0
    else:
        p = 0.0

    uniform_rv = torch.rand(1)
    # TODO: make default multiplier depend on the problem settings.
    # ranges = bounds[1] - bounds[0]
    default_multiplier = 1.0

    if "thresh" in label:
        multiplier = 0.0 if iteration >= p * num_iterations else default_multiplier
    elif "rg" in label:
        multiplier = 0.0 if uniform_rv > p else default_multiplier
    else:
        multiplier = 0.0

    return multiplier


def get_acquisition_outcome_transform(
    name: str,
    model: Model,
    scalarization_kwargs: Dict[str, Any],
    acq_kwargs: Dict[str, Any],
    util_kwargs: Dict[str, Any],
    tkwargs: Dict[str, Any],
    X_baseline: Tensor,
    Y_baseline: Tensor,
    bounds: Tensor,
) -> OutcomeTransform:
    r"""Get the outcome transform for the acquisition function and also the estimated
    bounds of the objectives.

    Args:
        name: The name of the objective function.
        model: The model.
        scalarization_kwargs: Arguments for the scalarization function.
        acq_kwargs: Arguments for the acquisition functions.
        util_kwargs: Arguments for the utility.
        tkwargs: Arguments for tensors, dtype and device.
        X_baseline: A `num_baseline x d`-dim Tensor containing the baseline inputs.
        Y_baseline: A `num_baseline x M`-dim Tensor containing the baseline outputs.
        bounds: A `2 x d`-dim Tensor containing the bounds of the inputs.

    Returns:
        outcome_transform: An outcome transform.
        bounds: An `2 x M`-dim Tensor containing the objective bounds.
    """
    estimate_the_bounds = acq_kwargs.get("estimate_the_bounds", True)

    # Get outcome transform
    if estimate_the_bounds:
        otf_label = acq_kwargs.get("outcome_transform", None)
    else:
        otf_label = "normalize-exact"

    obj_bounds = None
    if otf_label is None:
        otf_label = "identity"
        otf = Normalize()

    elif otf_label == "normalize-observations":
        obj_bounds = estimate_bounds(
            Y_baseline=Y_baseline,
            eta=0.4,
            kappa=None,
        )
        otf = Normalize(bounds=obj_bounds)
    elif otf_label == "normalize-model":
        obj_bounds = estimate_bounds(
            model=model,
            X_baseline=X_baseline,
            kappa=3.0,
        )
        otf = Normalize(bounds=obj_bounds)
    elif otf_label == "quantile-observations":
        means, variances = get_kernel_density_statistics(Y=Y_baseline)
        otf = GaussianQuantile(means=means, variances=variances)
    elif otf_label == "quantile-model":
        posterior = model.posterior(X_baseline)
        means = posterior.mean
        variances = posterior.variance
        otf = GaussianQuantile(means=means, variances=variances)
    elif otf_label == "normalize-exact":
        otf = get_problem_normalize_transform(
            name=name,
            util_kwargs=util_kwargs,
            tkwargs=tkwargs,
        )
    else:
        raise ValueError("Outcome transform is not supported!")

    if obj_bounds is None:
        obj_bounds = get_bounds_estimate(
            model=model,
            X_baseline=X_baseline,
            Y_baseline=Y_baseline,
        )

    # Apply the multipliers on the objectives
    multiplier_transform = get_preference_multiplier_transform(
        scalarization_kwargs=scalarization_kwargs,
        util_kwargs=acq_kwargs,
        tkwargs=tkwargs,
    )

    otf_dict = {otf_label: otf, "multiplier_transform": multiplier_transform}
    outcome_transform = ChainedOutcomeTransform(**otf_dict)

    return outcome_transform, obj_bounds


def get_uncertainty_acquisition(
    model: Model,
    objective_bounds: Tensor,
    tkwargs: Dict[str, Any],
) -> AcquisitionFunction:
    r"""Get the uncertainty acquisition function.

    Args:
        model: The model.
        objective_bounds: The `2 x M`-dim Tensor containing the estimated bounds for
            the objectives.
        tkwargs: The tensor dtype to use and device to use.

    Returns:
        The uncertainty acquisition function.
    """
    num_objectives = objective_bounds.shape[-1]
    normalize_otf = UnstandardizeAnalyticMultiOutputObjective(
        Y_mean=torch.zeros(num_objectives, **tkwargs),
        Y_std=1 / (objective_bounds[1] - objective_bounds[0]),
    )
    uncertainty_acq = Uncertainty(
        model=model,
        objective=normalize_otf,
    )
    return uncertainty_acq


def get_acquisition_function(
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
    tkwargs: Dict[str, Any],
    Y_baseline: Optional[Tensor] = None,
    bounds: Optional[Tensor] = None,
) -> AcquisitionFunction:
    r"""Initialize the acquisition function.

    Args:
        name: The name of the problem function.
        iteration: The iteration number.
        num_iterations: The number of iterations.
        label: The name of the acquisition function.
        model: The model.
        X_baseline: A `num_baseline x d`-dim Tensor containing the baseline inputs.
        scalarization_kwargs: Arguments for the scalarization function.
        sampling_kwargs: Arguments for the sampling used to set the scalarization
            function.
        acq_kwargs: Arguments for the acquisition functions.
        util_kwargs: Arguments for the utility.
        tkwargs: The tensor dtype to use and device to use.
        Y_baseline: A `num_baseline x M`-dim Tensor containing the baseline outputs.
        bounds: A `2 x d`-dim Tensor containing the bounds of the inputs. Note that
            this is the standardised bounds.

    Returns:
        The acquisition function.
    """
    num_objectives = model.num_outputs
    estimate_bounds = acq_kwargs.get("estimate_the_bounds", True)

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

    if "-ts" in label:
        acq_model = get_gp_samples(
            model=model, num_outputs=model.num_outputs, n_samples=1
        )
        num_samples = 1
        cache_root = False
        prune_baseline = True
        sampler = StochasticSampler(sample_shape=torch.Size([num_samples]))
    elif "-ucb" in label:
        ucb_beta = acq_kwargs.get("beta", 2.0)

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
        if "fs-" in label:
            # get the numbers after `fs-`
            num_samples = re.findall(r"%s(\d+)" % "fs-", label)
            if len(num_samples) > 0:
                acq_kwargs["num_samples"] = int(num_samples[0])
            else:
                raise ValueError("The number of function samples is not supported!")

        num_samples = acq_kwargs["num_samples"]

        cache_root = True
        prune_baseline = True
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))

    if ("eui" in label) or ("resi" in label):
        if "resi" in label:
            sampling_kwargs["num_weights"] = 1
        elif "mc-" in label:
            # get the numbers after `mc-`
            num_weights = re.findall(r"%s(\d+)" % "mc-", label)
            if len(num_weights) > 0:
                sampling_kwargs["num_weights"] = int(num_weights[0])
            else:
                raise ValueError("The number of MC samples is not supported!")

        ei_multiplier = get_ei_multiplier(
            label=label,
            iteration=iteration,
            num_iterations=num_iterations,
            bounds=objective_bounds,
        )

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

        mc_obj = get_utility_mcobjective(
            scalarization_fn=scalarization_fn,
            outcome_transform=outcome_transform,
        )

        eui_acq = qNoisyExpectedImprovement(
            model=acq_model,
            objective=mc_obj,
            X_baseline=X_baseline,
            sampler=sampler,
            prune_baseline=prune_baseline,
            cache_root=cache_root,
        )

        uncertainty_acq = get_uncertainty_acquisition(
            model=model,
            objective_bounds=objective_bounds,
            tkwargs=tkwargs,
        )

        acq = PenalizedAcquisitionFunction(
            raw_acqf=eui_acq,
            penalty_func=uncertainty_acq,
            regularization_parameter=-ei_multiplier,
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


def get_noise_std(name: str) -> float:
    r"""Get the standard deviation of the noise.

    Args:
        name: The name of the objective function.

    Returns:
        The problem.
    """
    if "0.01" in name:
        noise_std = 0.01
    elif "0.02" in name:
        noise_std = 0.02
    elif "0.05" in name:
        noise_std = 0.05
    elif "0.1" in name:
        noise_std = 0.1
    elif "0.15" in name:
        noise_std = 0.15
    elif "0.2" in name:
        noise_std = 0.2
    elif "0.25" in name:
        noise_std = 0.2
    elif "0.3" in name:
        noise_std = 0.2
    else:
        noise_std = 0

    return noise_std


def get_problem(name: str, tkwargs: Dict[str, Any]) -> MultiObjectiveTestProblem:
    r"""Initialize the test function.

    Args:
        name: The name of the objective function.
        tkwargs: The tensor dtype to use and device to use.

    Returns:
        The problem.
    """
    if name in problem_dict.keys():
        problem = problem_dict[name]
        problem_kwargs = {}
        if name in problem_kwargs_dict.keys():
            problem_kwargs = problem_kwargs_dict[name]

        if "std" in name:
            bounds = get_problem_bounds(name=name, tkwargs=tkwargs)
            ranges = bounds[1] - bounds[0]
            return problem(
                negate=True,
                noise_std=get_noise_std(name=name) * ranges,
                **problem_kwargs,
            )
        else:
            return problem(negate=True, **problem_kwargs)
    else:
        raise ValueError(f"Unknown function name: {name}!")


def get_problem_reference_point(
    name: str,
    scalarization_kwargs: Dict[str, Any],
    util_kwargs: Dict[str, Any],
    tkwargs: Dict[str, Any],
) -> Tensor:
    r"""Get the reference point for each problem. This is typically set in the
    transformed objective space i.e. the space after applying the outcome transform.

    Args:
        name: The name of the objective function.
        scalarization_kwargs: Arguments for the scalarization function.
        util_kwargs: Arguments for the utility.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        A `1 x M`-dim Tensor containing the reference point.
    """
    label = scalarization_kwargs.get("label", None)
    if name == "bc":
        return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "zdt1" in name:
        return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "zdt3" in name:
        otf = get_problem_normalize_transform(
            name=name,
            util_kwargs=util_kwargs,
            tkwargs=tkwargs,
        )
        if "igd" in label:
            upper_pf = get_zdt3_pf(upper=True)
            ref_points, _ = otf(upper_pf)
            return ref_points
        elif "d1" in label:
            lower_pf = get_zdt3_pf(upper=False)
            ref_points, _ = otf(lower_pf)
            return ref_points
        elif "r2" in label:
            use_utopia = scalarization_kwargs.get("use_utopia", True)
            if use_utopia:
                return 1.1 * torch.ones(1, 1, **tkwargs)
            else:
                return -0.1 * torch.ones(1, 1, **tkwargs)
        else:
            return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "dtlz2" in name:
        otf = get_problem_normalize_transform(
            name=name,
            util_kwargs=util_kwargs,
            tkwargs=tkwargs,
        )
        if "igd" in label:
            t = torch.linspace(0, 0.5, 101)
            upper_pf = -torch.column_stack(
                [torch.cos(t * torch.pi / 2), torch.sin(t * torch.pi / 2)]
            )
            ref_points, _ = otf(upper_pf)
            return ref_points
        elif "d1" in label:
            t = torch.linspace(0.5, 1, 101)
            lower_pf = -torch.column_stack(
                [torch.cos(t * torch.pi / 2), torch.sin(t * torch.pi / 2)]
            )
            ref_points, _ = otf(lower_pf)
            return ref_points
        elif "r2" in label:
            use_utopia = scalarization_kwargs.get("use_utopia", True)
            if use_utopia:
                return 1.1 * torch.ones(1, 1, **tkwargs)
            else:
                return -0.1 * torch.ones(1, 1, **tkwargs)
        else:
            return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "gmm" in name:
        return -0.1 * torch.ones(1, 1, **tkwargs)
    elif name == "penicillin":
        if "igd" in label or "d1" in label:
            ref_points = torch.tensor(
                [
                    [5.0, -0.0, -75.0],
                    [7.0, -8.0, -125.0],
                    [9.0, -16.0, -175.0],
                    [11.0, -24.0, -225.0],
                    [13.0, -32.0, -275.0],
                ],
                **tkwargs,
            )
            return ref_points
        else:
            return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "vehicle" in name:
        otf = get_problem_normalize_transform(
            name=name,
            util_kwargs=util_kwargs,
            tkwargs=tkwargs,
        )
        if "igd" in label:
            upper_pf = get_vehicle_pf(upper=True)
            ref_points, _ = otf(upper_pf)
            return ref_points
        elif "d1" in label:
            lower_pf = get_vehicle_pf(upper=False)
            ref_points, _ = otf(lower_pf)
            return ref_points
        elif "r2" in label:
            use_utopia = scalarization_kwargs.get("use_utopia", True)
            if use_utopia:
                return 1.1 * torch.ones(1, 1, **tkwargs)
            else:
                return -0.1 * torch.ones(1, 1, **tkwargs)
        else:
            return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "carside" in name:
        return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "marine" in name:
        return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "rocket" in name:
        otf = get_problem_normalize_transform(
            name=name,
            util_kwargs=util_kwargs,
            tkwargs=tkwargs,
        )
        if "igd" in label:
            upper_pf = get_rocket_pf(upper=True)
            ref_points, _ = otf(upper_pf)
            return ref_points
        elif "d1" in label:
            lower_pf = get_rocket_pf(upper=False)
            ref_points, _ = otf(lower_pf)
            return ref_points
        elif "r2" in label:
            use_utopia = scalarization_kwargs.get("use_utopia", True)
            if use_utopia:
                return 1.1 * torch.ones(1, 1, **tkwargs)
            else:
                return -0.1 * torch.ones(1, 1, **tkwargs)
        else:
            return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "truss" in name:
        if "r2" in label:
            use_utopia = scalarization_kwargs.get("use_utopia", True)
            if use_utopia:
                return 1.1 * torch.ones(1, 1, **tkwargs)
            else:
                return -0.1 * torch.ones(1, 1, **tkwargs)
        else:
            return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "planning" in name:
        if "ks" in label:
            ref_points = torch.zeros(2, 1, **tkwargs)
            ref_points[1] = 1.0
            return ref_points
        else:
            return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "cab" in name:
        if "ks" in label:
            ref_points = torch.zeros(2, 1, **tkwargs)
            ref_points[1] = 1.0
            return ref_points
        else:
            return -0.1 * torch.ones(1, 1, **tkwargs)
    elif "beam" in name:
        return -0.1 * torch.ones(1, 1, **tkwargs)
    else:
        raise ValueError(f"Unknown function name: {name}!")


def get_problem_bounds(
    name: str,
    tkwargs: Dict[str, Any],
) -> Tensor:
    r"""Get the objective bounds transform.

    Args:
        name: The name of the objective function.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        A `2 x M`-dim Tensor containing bounds for the objectives.
    """
    if name in bounds_dict.keys():
        bounds = torch.tensor(bounds_dict[name], **tkwargs)
    else:
        raise ValueError(f"Unknown function name: {name}!")

    return bounds


def get_problem_normalize_transform(
    name: str,
    util_kwargs: Dict[str, Any],
    tkwargs: Dict[str, Any],
) -> OutcomeTransform:
    r"""Get the normalize transform.

    Args:
        name: The name of the objective function.
        util_kwargs: Arguments for the utility.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        The normalize outcome transform.
    """
    otf_label = util_kwargs.get("outcome_transform", None)

    if otf_label == "normalize":
        bounds = get_problem_bounds(name=name, tkwargs=tkwargs)
        otf = Normalize(bounds=bounds)
    elif otf_label == "quantile":
        # estimate the quantile transformation using the kernel density estimate
        base_function = get_problem(name=name, tkwargs=tkwargs)
        num_samples = util_kwargs.get("num_samples", 2**20)
        seed = util_kwargs.get("seed", 0)
        X = get_baseline_candidates(
            bounds=base_function.bounds.to(**tkwargs),
            seed=seed,
            num_samples=num_samples,
        )
        fX = base_function.evaluate_true(X)
        Y = -fX if base_function.negate else fX
        means, variances = get_kernel_density_statistics(Y=Y)
        otf = GaussianQuantile(means=means, variances=variances)

    else:
        raise ValueError(f"Unknown outcome transform: {otf_label}!")

    return otf


def get_problem_outcome_transform(
    name: str,
    scalarization_kwargs: Dict[str, Any],
    util_kwargs: Dict[str, Any],
    tkwargs: Dict[str, Any],
) -> OutcomeTransform:
    r"""Get the outcome transform for each problem.

    Args:
        name: The name of the objective function.
        scalarization_kwargs: Arguments for the scalarization function.
        util_kwargs: Arguments for the utility.
        tkwargs: Arguments for tensors, dtype and device.

    Returns:
        The outcome transform.
    """

    multiplier_transform = get_preference_multiplier_transform(
        scalarization_kwargs=scalarization_kwargs,
        util_kwargs=util_kwargs,
        tkwargs=tkwargs,
    )

    otf_label = util_kwargs.get("outcome_transform", None)
    otf = get_problem_normalize_transform(
        name=name,
        util_kwargs=util_kwargs,
        tkwargs=tkwargs,
    )

    otf_dict = {otf_label: otf, "multiplier_transform": multiplier_transform}

    return ChainedOutcomeTransform(**otf_dict)


def get_set_utility(
    function_name: str,
    scalarization_kwargs: Dict[str, Any],
    util_kwargs: Dict[str, Any],
    sampling_kwargs: Dict[str, Any],
    tkwargs: Dict[str, Any],
    estimate_utility: bool = False,
    data: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    acq_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> SetUtility:
    r"""Get the set utility.

    Args:
        function_name: The name of the objective function.
        scalarization_kwargs: Arguments for the scalarization function.
        util_kwargs: Arguments for the utility.
        sampling_kwargs: Arguments used to determine the sampling used in the
            scalarization function
        tkwargs: Arguments for tensors, dtype and device.
        estimate_utility: If True, then we estimate the outcome transform.
        data: The data that is used to estimate the utility.
        model_kwargs: Arguments used to fit the model.
        acq_kwargs: Arguments for the acquisition functions.
        seed: The default seed.

    Returns:
        The set utility.
    """
    # Get the objective function
    base_function = get_problem(name=function_name, tkwargs=tkwargs)
    base_function.to(**tkwargs)
    num_objectives = base_function.num_objectives

    # Get the bounds
    bounds = base_function.bounds.to(**tkwargs)

    # Define the perfect evaluation.
    def eval_problem_noiseless(X: Tensor) -> Tensor:
        X = unnormalize(X, bounds)
        fX = base_function.evaluate_true(X)
        Y = -fX if base_function.negate else fX
        return Y

    # Ensure consistency of set utility performance metric across seeds by using same
    # Monte Carlo samples.

    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    if estimate_utility:
        # Fit the model.
        mll, model = initialize_model(
            train_x=data["X"], train_y=data["Y"], **model_kwargs
        )
        fit_gpytorch_mll(mll)

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

    set_utility = SetUtility(
        eval_problem=eval_problem_noiseless,
        scalarization_fn=util_scalarization_fn,
        outcome_transform=outcome_transform,
    )

    torch.random.set_rng_state(old_state)

    return set_utility
