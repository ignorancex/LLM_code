from certifiable_uda_loc.utils.matrix_utils import initialize_matrix
from py_factor_graph.data_associations import (
    ToyExamplePrior,
    VectorVariable,
    BooleanVariable,
    UnknownDataAssociationMeasurement,
)
import py_factor_graph.data_associations as pyfg_da
import certifiable_uda_loc.constraints.constraints_common as constraints_common

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
import certifiable_uda_loc.constraints.premultiplication as premultiplication
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
import py_factor_graph.variables as pyfg_variables
import certifiable_uda_loc.lifting as lifting
from certifiable_uda_loc.variables import is_col_var, rotation_position_names_from_pose

CONTINUOUS_VARIABLE_TYPES = Union[
    POSE_VARIABLE_TYPES, LANDMARK_VARIABLE_TYPES, VectorVariable
]


def lift_continuous_variable(
    var: CONTINUOUS_VARIABLE_TYPES,
) -> List[variables.LiftedQcQpVariable]:
    if isinstance(var, VectorVariable):
        name = var.name
        dims = var.dims
        true_value = var.true_value
        estimated_value = var.estimated_value
        return [
            variables.LiftedQcQpVariable(
                dims, name, true_value=true_value, estimated_value=estimated_value
            )
        ]
    if isinstance(var, pyfg_variables.LandmarkVariable2D) or isinstance(
        var, pyfg_variables.LandmarkVariable3D
    ):
        dims = len(var.true_position)
        true_val = None
        est_val = None
        if var.true_position is not None:
            true_val = np.array(var.true_position).reshape(-1, 1)
        if var.estimated_position is not None:
            est_val = np.array(var.estimated_position).reshape(-1, 1)
        return [
            variables.LiftedQcQpVariable(
                dims, var.name, true_value=true_val, estimated_value=est_val
            )
        ]
    if isinstance(var, PoseVariable2D):
        n = len(var.true_position)
        rot_name, pos_name = rotation_position_names_from_pose(var.name)
        rot_qcqp_var = variables.RotationVariable(
            (n, n),
            rot_name,
            true_value=var.true_rotation_matrix,
            estimated_value=var.estimated_rotation_matrix,
        )
        pos_qcqp_var = variables.LiftedQcQpVariable(
            (n, 1),
            pos_name,
            true_value=np.array([var.true_position[0], var.true_position[1]]).reshape(
                -1, 1
            ),
            estimated_value=np.array(
                [var.estimated_position[0], var.estimated_position[1]]
            ).reshape(-1, 1),
        )
        return [rot_qcqp_var, pos_qcqp_var]


def lift_feasible_point(
    boolean_vars: List[BooleanVariable],
    continuous_vars: List[variables.LiftedQcQpVariable],
    column_names_all: str = None,
    true_or_estimated_value: str = "true",
    formulation="with_x",
):

    nx = continuous_vars[0].dims[0]
    hom_var = variables.HomogenizationVariable((nx, nx), "one", true_value=np.eye(nx))
    hom_var.estimated_value = np.eye(nx)

    bool_lifted_vars: List[variables.LiftedQcQpVariable] = [
        variables.LiftedQcQpVariable(
            dims=(nx, nx),
            name=var.name,
            true_value=int(var.true_value) * np.eye(nx),
            estimated_value=int(var.estimated_value) * np.eye(nx),
        )
        for var in boolean_vars
    ]

    bool_cont_lifted_vars: List[variables.LiftedQcQpVariable] = []
    bool_cont_true_values = []
    for bool_var in boolean_vars:
        for cont_var in continuous_vars:
            bool_cont_lifted_var = variables.LiftedQcQpVariable(
                cont_var.dims,
                bool_cont_var_name(bool_var.name, cont_var.name),
                true_value=int(bool_var.true_value) * cont_var.true_value,
                estimated_value=int(bool_var.estimated_value)
                * cont_var.estimated_value,
            )
            bool_cont_lifted_vars.append(bool_cont_lifted_var)
            bool_cont_true_values.append(bool_var.true_value * cont_var.true_value)

    if formulation == "with_x":
        all_vars_lifted = (
            [hom_var] + bool_lifted_vars + bool_cont_lifted_vars + continuous_vars
        )
    if formulation == "without_x":
        all_vars_lifted = [hom_var] + bool_lifted_vars + bool_cont_lifted_vars

    if true_or_estimated_value == "true":
        all_vars_values: List[np.ndarray] = [var.true_value for var in all_vars_lifted]
    if true_or_estimated_value == "estimated":
        all_vars_values: List[np.ndarray] = [
            var.estimated_value for var in all_vars_lifted
        ]
    # reshape to (-1,1) for each variable if it has only one dimension
    for lv1 in range(len(all_vars_values)):
        if len(all_vars_values[lv1].shape) == 1:
            all_vars_values[lv1] = all_vars_values[lv1].reshape(-1, 1)

    all_vars_column_names = []
    for var in all_vars_lifted:
        all_vars_column_names += var.column_names()

    if column_names_all is None:
        column_names_all = all_vars_column_names

    X_np = np.hstack(all_vars_values)
    # Reorder X_np column order to match column_names_all
    idx = [all_vars_column_names.index(col) for col in column_names_all]
    X_np = X_np[:, idx]

    rows = [f"{lv1}" for lv1 in range(nx)]
    X = np_to_polymatrix(column_names_all, rows, X_np)
    return X


def moment_constraints_multirow_formulation(
    lifted_variables: List[str], hom_var_name: str, dim: int
) -> Tuple[List[str], List[PolyMatrix]]:
    # Have to be a bit careful here.
    # I seem to be passing in only the first col for each lifted
    # variable. This might be fine but check later..
    A_list = []
    var_unlifted_list = []
    unlifted_is_multicol_dict = {}
    for lv1 in range(len(lifted_variables)):
        if is_col_var(lifted_variables[lv1]):
            var_unlifted, idx = colvar_to_var(lifted_variables[lv1])
            if idx == 0:
                unlifted_is_multicol_dict[var_unlifted] = True
                var_unlifted_list.append(var_unlifted)
        else:
            var_unlifted = lifted_variables[lv1]
            unlifted_is_multicol_dict[var_unlifted] = False
            var_unlifted_list.append(var_unlifted)

    moment_constraint_dict = constraints_common.get_moment_constraint_dict(
        var_unlifted_list, hom_var_name
    )

    for key, equal_variable_list in moment_constraint_dict.items():
        for pair_unlifted in list(itertools.combinations(equal_variable_list, 2)):
            pair00_names = get_var_column_names(
                pair_unlifted[0][0], dim, unlifted_is_multicol_dict[pair_unlifted[0][0]]
            )
            pair01_names = get_var_column_names(
                pair_unlifted[0][1], dim, unlifted_is_multicol_dict[pair_unlifted[0][1]]
            )
            pair10_names = get_var_column_names(
                pair_unlifted[1][0], dim, unlifted_is_multicol_dict[pair_unlifted[1][0]]
            )
            pair11_names = get_var_column_names(
                pair_unlifted[1][1], dim, unlifted_is_multicol_dict[pair_unlifted[1][1]]
            )

            pair_lists = [pair00_names, pair01_names, pair10_names, pair11_names]
            is_multicol = [len(pair_list) > 1 for pair_list in pair_lists]

            for lv00, pair00 in enumerate(pair00_names):
                for lv01, pair01 in enumerate(pair01_names):
                    for lv10, pair10 in enumerate(pair10_names):
                        for lv11, pair11 in enumerate(pair11_names):
                            lv_arr = [lv00, lv01, lv10, lv11]
                            lv_arr_multicol = np.array(
                                [lv_arr[lv1] for lv1 in range(4) if is_multicol[lv1]]
                            )
                            if len(lv_arr_multicol) > 0:
                                lv_arr_multicol = np.atleast_1d(lv_arr_multicol)
                                if not (
                                    np.allclose(lv_arr_multicol - lv_arr_multicol[0], 0)
                                ):
                                    continue

                            A = initialize_poly_matrix(lifted_variables)
                            A[pair00, pair01] = 1
                            A[pair10, pair11] = -1

                            A_list.append(A)

    return A_list


# This should just take care of grouping the data association variables
# in the groups where we have sum c = 1,
# and creating those constraints...
# The rest can be applied at the top level.
# This structure doesnt quite make sense.
# TODO: Fix this. This function should NOT take care of all the cx stuff.
# Thats for the top level.
# The diag constraints should also be in a different place.


# TODO: Just return c_constraints then create c_constraints_for_lifted_problem separately
# for cleanliness.
def get_c_constraints_per_factor(
    boolean_vars: List[BooleanVariable],
    hom_var: variables.HomogenizationVariable,
    discrete_variable_constraints: List[str],
) -> List[PolyMatrix]:
    nx = hom_var.dims[0]

    bool_lifted_vars: List[variables.LiftedQcQpVariable] = [
        lifting.lift_scalar_variable(var, nx) for var in boolean_vars
    ]

    c_constraints, _ = constraints_boolean.create_c_constraint_list(
        hom_var.name,
        [var.name for var in bool_lifted_vars],
        discrete_variable_constraints,
    )
    return c_constraints


def data_association_factor_constraints(
    boolean_vars: List[BooleanVariable],
    continuous_vars: List[variables.LiftedQcQpVariable],
    hom_var: variables.HomogenizationVariable,
    lifted_variables_names_all: str,
) -> Tuple[
    List[variables.LiftedQcQpVariable],
    List[variables.LiftedQcQpVariable],
    Dict[str, List[PolyMatrix]],
]:
    # Constraints that don't involve lifted_variables_names_all are ignored.

    # Need more constraints involving x.
    # For a single data association factor. The boolean variables
    # sum to one.
    nx = continuous_vars[0].dims[0]
    # TODO: Will have to recheck if I have the correct c constraints.
    # Sum constraint nee            cxi_cxj_constraints.append(A)ds to be in there.
    bool_lifted_vars, bool_cont_lifted_vars = lifting.lift_data_association_variables(
        boolean_vars, continuous_vars
    )

    mix_var_names = []
    for bool_cont_lifted_var in bool_cont_lifted_vars:
        mix_var_names += bool_cont_lifted_var.column_names()

    lifted_vars_involving_bool = []
    for bool_lifted_var in bool_lifted_vars:
        lifted_vars_involving_bool += bool_lifted_var.column_names()
    for bool_cont_lifted_var in bool_cont_lifted_vars:
        lifted_vars_involving_bool += bool_cont_lifted_var.column_names()

    c_constraints_for_lifted_problem = []

    for lv_dim in range(nx):
        constraints, _ = constraints_boolean.create_c_constraint_list(
            hom_var.column_names()[lv_dim],
            [var.column_names()[lv_dim] for var in bool_lifted_vars],
            # [var_to_colvar(var.name, lv_dim) for var in bool_lifted_vars],
        )
        c_constraints_for_lifted_problem += constraints

    c_constraints, c_constraint_names = constraints_boolean.create_c_constraint_list(
        hom_var.name, [var.name for var in bool_lifted_vars]
    )

    x_variable_names = [var.name for var in continuous_vars]
    bool_variable_names = [var.name for var in boolean_vars]
    cx_constraints = []
    cxi_cxj_constraints = []

    # for A_th in c_constraints:
    for A_th, name in zip(c_constraints, c_constraint_names):
        # Create corresponding version for overall matrix...
        # if A_th["one", "one"] == 0:
        for x in x_variable_names:
            cx_constraints += premultiplication.x_premultiply_A_theta(
                x, lifted_variables_names_all, hom_var, bool_lifted_vars, A_th
            )

        if len(x_variable_names) == 1:
            combos = [(x_variable_names[0], x_variable_names[0])]
        else:
            combos = list(itertools.combinations(x_variable_names, 2))
            combos = combos + [(x, x) for x in x_variable_names]
        for x1, x2 in combos:
            A_list = premultiplication.xi_xj_premultiply_A_theta(
                x1,
                x2,
                lifted_variables_names_all,
                [var.name for var in bool_lifted_vars],
                A_th,
            )
            cxi_cxj_constraints += A_list

    # I dont think I truly need these, but this will be a test.
    bool_premultiplication_constraints_single_fac = []
    for bool_name in bool_variable_names:
        A = premultiplication.boolean_premultiply_constraint(
            hom_var,
            bool_names_to_premultiply=bool_variable_names,
            bool_var_premultiplying=bool_name,
            b_hom=-1,
            b=np.ones((len(bool_variable_names),)),
            lifted_variables_names_all=lifted_variables_names_all,
        )
        bool_premultiplication_constraints_single_fac.append(A)

    constraint_dict = {
        # "idiag_constraints": idiag_constraints,
        "c_constraints": c_constraints_for_lifted_problem,
        # THe cx and cxi_cxj here should happen at top level.
        "cx_constraints": cx_constraints,
        "cxi_cxj_constraints": cxi_cxj_constraints,
    }

    return constraint_dict
