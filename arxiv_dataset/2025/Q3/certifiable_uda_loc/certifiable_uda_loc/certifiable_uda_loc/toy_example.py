from typing import Dict, List, Tuple

import dill as pickle
import numpy as np

import itertools

from py_factor_graph.factor_graph import FactorGraphData
import py_factor_graph.data_associations as data_associations
from scipy.optimize import minimize


def minimize_sum_quadratic_forms_fg(
    prior_parameter_set_list: List[data_associations.ToyExamplePrior],
):
    """
    minimize (x-x_1)^\trans Q_1 (x-x_1) + (x-x_2)^\trans Q_1 (x-x_2)
    Q_1 (x-x_1) + Q_2 (x-x_2) = 0
    (Q_1+Q_2)x = Q1x_1 +Q2x_2
    """

    def f(x):
        val = 0.0
        for p in prior_parameter_set_list:
            dx = (x - p.center).reshape(-1, 1)
            Q = np.atleast_2d(p.Q)
            val = val + (dx.T @ Q @ dx).squeeze() + p.offset
        return val

    initial_guess = prior_parameter_set_list[0].center
    result = minimize(f, initial_guess, method="BFGS")
    return result.fun, result.x


def setup_case(
    dims: int,
    n_comp: int,
    n_factors: int,
    center_spread: float,
    offset_spread: float,
    positive_offsets: bool,
    minimum_offset_spread: float,
) -> FactorGraphData:

    c_vars_list_of_lists: List[List[str]] = []
    for lv_factor in range(n_factors):
        c_vars_list_of_lists.append([f"c{lv_factor}{lv1}" for lv1 in range(n_comp)])

    continuous_var_name = "x"
    continuous_variable = data_associations.VectorVariable(
        name=continuous_var_name, dims=dims
    )
    boolean_variables = [
        [
            data_associations.BooleanVariable(
                name=f"c{lv_factor}{lv1}", true_value=None
            )
            for lv1 in range(n_comp)
        ]
        for lv_factor in range(n_factors)
    ]
    factors = []
    for lv_factor in range(n_factors):
        Q_orig_list = []
        for lv1 in range(n_comp):
            Q_orig_list.append(np.eye(dims))
        center_list = [np.random.random(dims) * center_spread for _ in range(n_comp)]
        offset_list = [np.random.random() * offset_spread for _ in range(n_comp)]
        if positive_offsets:
            offset_list = [np.abs(offset) for offset in offset_list]
            min_offset = np.min(offset_list)
            offset_list = [offset - min_offset for offset in offset_list]
        for lv1 in range(1, len(offset_list)):
            if np.abs(offset_list[lv1] - offset_list[lv1 - 1]) < minimum_offset_spread:
                offset_list[lv1] = offset_list[lv1 - 1] + minimum_offset_spread
        component_measurements = [
            data_associations.ToyExamplePrior(
                name=continuous_var_name, center=center, offset=offset, Q=Q
            )
            for center, offset, Q in zip(center_list, offset_list, Q_orig_list)
        ]
        factor = data_associations.UnknownDataAssociationMeasurement(
            measurement_list=component_measurements,
            boolean_variables=boolean_variables[lv_factor],
        )
        factors.append(factor)

    flat_boolean_variables = [var for sublist in boolean_variables for var in sublist]
    fg = FactorGraphData(
        dimension=dims,
        vector_variables=[continuous_variable],
        boolean_variables=flat_boolean_variables,
        unknown_data_association_measurements=factors,
    )
    return fg


def set_ground_truth_values(
    fg: FactorGraphData,
) -> Tuple[float, np.ndarray, List[int]]:
    # We will have many different combinations of PriorParameters.
    num_components_per_factor = [
        len(factor.measurement_list)
        for factor in fg.unknown_data_association_measurements
    ]
    n_factors = len(fg.unknown_data_association_measurements)
    val_opt = 1e16
    x_opt = None
    components_opt = None
    for active_components in itertools.product(
        *[range(n) for n in num_components_per_factor]
    ):
        # print(active_components)
        prior_parameter_set: List[data_associations.ToyExamplePrior] = []
        for lv_fac in range(n_factors):
            active_component_idx = active_components[lv_fac]
            prior_parameter_set.append(
                fg.unknown_data_association_measurements[lv_fac].measurement_list[
                    active_component_idx
                ]
            )

        val_check, x_check = minimize_sum_quadratic_forms_fg(prior_parameter_set)
        if val_check < val_opt:
            val_opt = val_check
            x_opt = x_check
            components_opt = active_components
    fg.vector_variables[0].true_value = x_opt.reshape(-1, 1)

    for lv_fac in range(n_factors):
        active_component_idx = components_opt[lv_fac]
        for lv1 in range(num_components_per_factor[lv_fac]):
            fg.unknown_data_association_measurements[lv_fac].boolean_variables[
                lv1
            ].true_value = (lv1 == active_component_idx)

    return fg
