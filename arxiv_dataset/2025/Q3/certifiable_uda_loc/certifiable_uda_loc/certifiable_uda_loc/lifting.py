from certifiable_uda_loc.utils.matrix_utils import initialize_matrix
from py_factor_graph.data_associations import (
    ToyExamplePrior,
    VectorVariable,
    BooleanVariable,
    UnknownDataAssociationMeasurement,
)
import py_factor_graph.data_associations as pyfg_da
import py_factor_graph.measurements as pyfg_measurements
import certifiable_uda_loc.constraints.constraints_common as constraints_common
import certifiable_uda_loc.variables as variables

from py_factor_graph.variables import (
    POSE_VARIABLE_TYPES,
    LANDMARK_VARIABLE_TYPES,
    PoseVariable2D,
    PoseVariable3D,
)
from certifiable_uda_loc.monomials import generate_mixture_monomials
from py_factor_graph.factor_graph import FactorGraphData
import attr
from typing import List, Union, Tuple, Dict
from poly_matrix import PolyMatrix
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
import numpy as np
import itertools

import certifiable_uda_loc.constraints.constraints_boolean as constraints_boolean
from certifiable_uda_loc.utils.string_utils import (
    var_to_colvar,
    get_var_column_names,
    colvar_to_var,
    bool_cont_var_name,
    bool_cont_var_name_to_vars,
)
from certifiable_uda_loc.utils.matrix_utils import np_to_polymatrix
from collections import namedtuple
import certifiable_uda_loc.variables as variables

QuadraticForm = namedtuple("QuadraticForm", "Q_xi b_xi c_xi")


def get_column_names_from_factor(
    factor: Union[ToyExamplePrior, pyfg_da.RotationPrior2D, pyfg_da.PositionPrior2D]
):
    if any(
        [
            isinstance(factor, factor_type)
            for factor_type in [ToyExamplePrior, pyfg_da.PositionPrior2D]
        ]
    ):
        return [factor.name], [factor.name]
    if any(
        [isinstance(factor, factor_type) for factor_type in [pyfg_da.RotationPrior2D]]
    ):
        return [factor.name], [var_to_colvar(factor.name, lv1) for lv1 in range(2)]
    if isinstance(factor, pyfg_da.PosePrior2D):
        rot_name, pos_name = variables.rotation_position_names_from_pose(factor.name)
        return [rot_name, pos_name], [
            var_to_colvar(rot_name, 0),
            var_to_colvar(rot_name, 1),
            pos_name,
        ]
    if isinstance(factor, pyfg_measurements.PoseToLandmarkMeasurement2D):
        rot_name, pos_name = variables.rotation_position_names_from_pose(
            factor.pose_name
        )
        landmark_name = factor.landmark_name
        return [rot_name, pos_name, landmark_name], [
            var_to_colvar(rot_name, 0),
            var_to_colvar(rot_name, 1),
            pos_name,
            landmark_name,
        ]
    if isinstance(factor, pyfg_measurements.PoseMeasurement2D):
        rot_name1, pos_name1 = variables.rotation_position_names_from_pose(
            factor.base_pose
        )
        rot_name2, pos_name2 = variables.rotation_position_names_from_pose(
            factor.to_pose
        )
        opt_variable_name_list = [rot_name1, pos_name1, rot_name2, pos_name2]
        col_list = [
            var_to_colvar(rot_name1, 0),
            var_to_colvar(rot_name1, 1),
            pos_name1,
        ] + [
            var_to_colvar(rot_name2, 0),
            var_to_colvar(rot_name2, 1),
            pos_name2,
        ]

        return opt_variable_name_list, col_list
    if isinstance(factor, pyfg_measurements.PoseToKnownLandmarkMeasurement2D):
        rot_name1, pos_name1 = variables.rotation_position_names_from_pose(
            factor.pose_name
        )
        return [rot_name1, pos_name1], [
            var_to_colvar(rot_name1, 0),
            var_to_colvar(rot_name1, 1),
            pos_name1,
        ]
    if isinstance(factor, pyfg_measurements.KnownPoseToLandmarkMeasurement2D):
        landmark_name = factor.landmark_name
        return [landmark_name], [landmark_name]


def lift_quadratic_prior(
    col_name, x_check, hom_var: variables.HomogenizationVariable
) -> PolyMatrix:
    lifted_variables = hom_var.column_names() + [col_name]
    Q = initialize_poly_matrix(lifted_variables)
    Q[hom_var.col_names[0], hom_var.col_names[0]] = x_check.T @ x_check
    Q[col_name, col_name] = 1
    Q[hom_var.col_names[0], col_name] = -x_check[0]
    Q[hom_var.col_names[1], col_name] = -x_check[1]
    return Q


def lift_factor_to_Q(
    factor: Union[
        ToyExamplePrior,
        pyfg_da.RotationPrior2D,
        pyfg_da.PositionPrior2D,
        pyfg_da.PosePrior2D,
        pyfg_measurements.PoseMeasurement2D,
    ],
    hom_var: variables.HomogenizationVariable,
) -> List[QuadraticForm]:
    if any(
        [isinstance(factor, meas_type) for meas_type in pyfg_da.PRIOR_FACTOR_TYPES_LIST]
    ):
        # TODO: Figure out the weights and offsets.
        _, col_names = get_column_names_from_factor(factor)
        offsets = None
        if isinstance(factor, ToyExamplePrior):
            vals = [factor.center]
            offsets = [factor.offset]
            weights = [factor.Q[0, 0]]

        if isinstance(factor, pyfg_da.RotationPrior2D):
            dim = factor.rotation.shape[0]
            vals = [factor.rotation[:, lv1] for lv1 in range(dim)]
            weights = [factor.weight, factor.weight]

        if isinstance(factor, pyfg_da.PositionPrior2D):
            dim = factor.position.shape[0]
            vals = [factor.position]
            weights = [factor.weight]

        if isinstance(factor, pyfg_da.PosePrior2D):
            dim = factor.rotation.shape[0]
            vals = [factor.rotation[:, lv1] for lv1 in range(dim)] + [
                np.array(factor.position)
            ]
            weights = [factor.weight_rot, factor.weight_rot, factor.weight_pos]

        Q_list = []
        for col_name, val, weight in zip(col_names, vals, weights):
            Q = lift_quadratic_prior(col_name, val, hom_var)
            Q_list.append(weight * Q)
        if offsets is not None:
            for offset, Q in zip(offsets, Q_list):
                Q[hom_var.col_names[0], hom_var.col_names[0]] += offset

    if isinstance(factor, pyfg_measurements.PoseToLandmarkMeasurement2D):
        y = factor.r_b_lb
        rot_name, pos_name = variables.rotation_position_names_from_pose(
            factor.pose_name
        )
        weight = factor.weight
        landmark_name = factor.landmark_name
        _, var_col_names = get_column_names_from_factor(factor)

        lifted_variables = hom_var.col_names + var_col_names
        Q = initialize_poly_matrix(
            lifted_variables,
        )
        Q[hom_var.col_names[0], hom_var.col_names[0]] = y.T @ y
        Q[pos_name, pos_name] = 1
        Q[landmark_name, landmark_name] = 1
        Q[pos_name, landmark_name] = -1  # half of the actual valye - Q is symmetric.
        Q[var_to_colvar(rot_name, 0), pos_name] = y[0]
        Q[var_to_colvar(rot_name, 1), pos_name] = y[1]
        Q[var_to_colvar(rot_name, 0), landmark_name] = -y[0]
        Q[var_to_colvar(rot_name, 1), landmark_name] = -y[1]
        Q_list = [weight * Q]

    if isinstance(factor, pyfg_measurements.PoseMeasurement2D):
        # TODO: Unit test of frobenius norm...
        _, var_col_names = get_column_names_from_factor(factor)
        lifted_variables = hom_var.col_names + var_col_names

        rot_name1, pos_name1 = variables.rotation_position_names_from_pose(
            factor.base_pose
        )
        rot_name2, pos_name2 = variables.rotation_position_names_from_pose(
            factor.to_pose
        )

        dC = factor.rotation_matrix
        dr = factor.translation_vector

        Q_r = initialize_poly_matrix(
            lifted_variables,
        )  # translation cost.
        Q_r[hom_var.col_names[0], hom_var.col_names[0]] = np.linalg.norm(dr, 2) ** 2
        Q_r[pos_name1, pos_name1] = 1
        Q_r[pos_name2, pos_name2] = 1
        Q_r[pos_name1, pos_name2] = -1  # half of the actual valye - Q is symmetric.
        Q_r[var_to_colvar(rot_name1, 0), pos_name1] = dr[0]
        Q_r[var_to_colvar(rot_name1, 1), pos_name1] = dr[1]
        Q_r[var_to_colvar(rot_name1, 0), pos_name2] = -dr[0]
        Q_r[var_to_colvar(rot_name1, 1), pos_name2] = -dr[1]

        Q_c = initialize_poly_matrix(lifted_variables)
        for lv1 in range(2):
            Q_c[hom_var.col_names[lv1], hom_var.col_names[lv1]] = 2
        for lv1 in range(2):
            for lv2 in range(2):
                Q_c[var_to_colvar(rot_name1, lv1), var_to_colvar(rot_name2, lv2)] = -dC[
                    lv1, lv2
                ]
        Q_list = [factor.weight_rot * Q_c, factor.weight_pos * Q_r]
        for Q in Q_list:
            if np.any(np.isinf(Q.toarray())):
                print("Qc")
                print(Q_c)
                print("Qr")
                print(Q_r)
                break
        # Localization task
        for Q in Q_list:
            if np.any(np.isinf(Q.toarray())):
                print(Q)
                raise (BaseException("Infinite entries in the cost matrix..."))

    if isinstance(factor, pyfg_measurements.PoseToKnownLandmarkMeasurement2D):
        y = factor.r_b_lb
        ell = factor.r_a_la
        rot_name, pos_name = variables.rotation_position_names_from_pose(
            factor.pose_name
        )
        weight = factor.weight

        _, var_col_names = get_column_names_from_factor(factor)
        lifted_variables = hom_var.col_names + var_col_names
        Q = initialize_poly_matrix(lifted_variables)
        Q[hom_var.col_names[0], hom_var.col_names[0]] = (
            np.linalg.norm(ell) ** 2 + np.linalg.norm(y) ** 2
        )
        y_ell_prod = ell.reshape(-1, 1) @ y.reshape(1, -1)
        for i in range(2):
            for j in range(2):
                Q[hom_var.col_names[i], var_to_colvar(rot_name, j)] = -y_ell_prod[i, j]

        for lv1 in range(2):
            Q[hom_var.col_names[lv1], pos_name] = -ell[lv1]
        for lv1 in range(2):
            Q[var_to_colvar(rot_name, lv1), pos_name] = y[lv1]
        Q[pos_name, pos_name] = 1
        Q_list = [weight * Q]
    if isinstance(factor, pyfg_measurements.KnownPoseToLandmarkMeasurement2D):
        C = factor.C_ab
        r = factor.r_a_ba
        y = factor.r_b_lb
        landmark_name = factor.landmark_name
        lifted_variables = hom_var.col_names + [landmark_name]
        weight = factor.weight
        Q = initialize_poly_matrix(lifted_variables)

        Q[hom_var.col_names[0], hom_var.col_names[0]] = (
            np.linalg.norm(r) ** 2
            + np.linalg.norm(y) ** 2
            + 2 * r.reshape(1, -1) @ C @ y.reshape(-1, 1)
        )

        rCy_sum = r + C @ y
        for lv1 in range(2):
            Q[hom_var.col_names[lv1], landmark_name] = -rCy_sum[lv1]
        Q[landmark_name, landmark_name] = 1
        Q_list = [Q]

    for Q in Q_list:
        if np.any(np.isinf(Q.toarray())):
            raise (BaseException("Infinite entries in the cost matrix..."))

    return Q_list


# This will also require quite a bit of thought.
# Now have something more general.
def lift_component_quadratic_form_to_inner_product_standard(
    hom_var: variables.HomogenizationVariable,
    col_name: str,
    q_xi: float,  # This formulation only works if Q_xi is multiple of identity..
    b_xi: np.ndarray,
    c_xi: float,
):
    lifted_variables = hom_var.column_names() + [col_name]
    Q: PolyMatrix = initialize_poly_matrix(lifted_variables)
    Q[col_name, col_name] = q_xi
    for lv1, hom_var_lv1 in enumerate(hom_var.column_names()):
        Q[hom_var_lv1, col_name] = 0.5 * b_xi[lv1]

    Q[hom_var.column_names()[0], hom_var.column_names()[0]] = 0.5 * c_xi
    return Q


def lift_scalar_variable(var: BooleanVariable, nx: int) -> variables.LiftedQcQpVariable:
    lifted_var = variables.LiftedQcQpVariable(
        dims=(nx, nx), name=var.name, true_value=np.eye(nx) * var.true_value
    )
    return lifted_var


def lift_data_association_variables(
    boolean_vars: List[BooleanVariable],
    continuous_vars: List[variables.LiftedQcQpVariable],
    nx=None,
) -> Tuple[List[variables.LiftedQcQpVariable], List[variables.LiftedQcQpVariable]]:

    nx = continuous_vars[0].dims[0]
    bool_lifted_vars: List[variables.LiftedQcQpVariable] = [
        lift_scalar_variable(var, nx)
        # variables.LiftedQcQpVariable(
        #     dims=(nx, nx), name=var.name, true_value=np.eye(nx) * var.true_value
        # )
        for var in boolean_vars
    ]

    bool_cont_lifted_vars: List[variables.LiftedQcQpVariable] = []
    for bool_var in boolean_vars:
        for cont_var in continuous_vars:
            bool_cont_lifted_var = variables.LiftedQcQpVariable(
                cont_var.dims, bool_cont_var_name(bool_var.name, cont_var.name)
            )
            bool_cont_lifted_vars.append(bool_cont_lifted_var)

    return bool_lifted_vars, bool_cont_lifted_vars


def get_uda_meas_variables(uda_meas: UnknownDataAssociationMeasurement):
    col_names = []
    opt_variable_names = []
    for meas in uda_meas.measurement_list:
        opt_variable_names_lv1, col_names_lv1 = get_column_names_from_factor(meas)
        col_names += col_names_lv1
        opt_variable_names += opt_variable_names_lv1
    return col_names


# Try doing this through uda_meas_group.
def get_lifted_variables_all(
    hom_var: variables.HomogenizationVariable,
    vector_lifted_vars: List[variables.ColumnQcQcpVariable],
    uda_meas_list: List[UnknownDataAssociationMeasurement],
    sparse_bool_cont_variables=False,
) -> List[variables.LiftedQcQpVariable]:

    lifted_variables_names_all = []
    bool_lifted_all: List[variables.LiftedQcQpVariable] = []
    bool_cont_all: List[variables.LiftedQcQpVariable] = []

    lifted_variables_names_all += hom_var.column_names()

    for uda_meas in uda_meas_list:
        if not sparse_bool_cont_variables:
            cont_vars_lv1 = vector_lifted_vars
        else:
            cont_vars_names_lv1 = get_uda_meas_variables(uda_meas)
            cont_vars_lv1 = [
                v for v in vector_lifted_vars if v.name in cont_vars_names_lv1
            ]
        bool_lifted_vars, bool_cont_lifted_vars = lift_data_association_variables(
            uda_meas.boolean_variables, cont_vars_lv1
        )

        bool_lifted_all += bool_lifted_vars
        bool_cont_all += bool_cont_lifted_vars

    for bool_lifted in bool_lifted_all:
        lifted_variables_names_all += bool_lifted.column_names()

    for bool_cont in bool_cont_all:
        lifted_variables_names_all += bool_cont.column_names()

    for x in vector_lifted_vars:
        lifted_variables_names_all += x.column_names()

    lifted_variables: List[variables.LiftedQcQpVariable] = (
        [hom_var] + bool_lifted_all + bool_cont_all + vector_lifted_vars
    )

    return lifted_variables, lifted_variables_names_all
