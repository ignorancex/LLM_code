import certifiable_uda_loc.lifting as lifting
from certifiable_uda_loc.utils.matrix_utils import initialize_matrix
from typing import List, Tuple
from poly_matrix import PolyMatrix
import itertools
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
import certifiable_uda_loc.variables as variables
import py_factor_graph.data_associations as pyfg_da

# Little bit of house cleaning to be done here too.
# For happiness.


def constraint_association_variables_sum_one(
    A_constraint: PolyMatrix, c_var_names: List[str], hom_var_name: str
) -> PolyMatrix:
    for c_var in c_var_names:
        A_constraint[hom_var_name, c_var] = 1 / 2  # off-diagonal, duplicated

    A_constraint[hom_var_name, hom_var_name] = -1
    return A_constraint


def constraint_boolean(
    A_constraint: PolyMatrix, c_var_name: str, hom_var_name: str = "one"
) -> PolyMatrix:
    one = hom_var_name
    A_constraint[one, c_var_name] = -1 / 2  # off-diagonal, duplicated
    A_constraint[c_var_name, c_var_name] = 1
    return A_constraint


def constraint_sum_c_c_squared_redundant(
    A_constraint: PolyMatrix,
    c_var_name: str,
    all_c_vars: List[str],
    hom_var_name: str = "one",
) -> PolyMatrix:

    A_constraint[hom_var_name, hom_var_name] = -1
    A_constraint[c_var_name, c_var_name] = 1
    for c_var in all_c_vars:
        if c_var != c_var_name:
            A_constraint[hom_var_name, c_var] = 1 / 2  # off diagonal, duplicated

    return A_constraint


def constraints_prod_ci_cj_zero(
    lifted_variables: List[str], c_variable_names: List[str], type: str
):
    """
    c_i c_j = 0 for i not equal to j
    """
    A_list = []
    # Need to generate all pairwise combinations of c variables.
    for pair in itertools.combinations(c_variable_names, 2):

        A: PolyMatrix = initialize_matrix(lifted_variables, type)
        A[pair[0], pair[1]] = 1 / 2
        A_list.append(A)

    return A_list


def create_c_constraint_list(
    hom_var_name, c_var_names, discrete_variable_constraints: List[str]
):
    """
    discrete_variable_constraints: List[str]. Subset of: [
            "bool",
            "prod_ci_cj",
            "c_c_squared",
            "sum_one",
        ],
    """
    lifted_variables_names = [hom_var_name] + c_var_names
    prod_ci_cj_constraints = constraints_prod_ci_cj_zero(
        lifted_variables_names, c_var_names, "PolyMatrix"
    )

    sum_c_c_squared_constraints = []
    for c_var_name in c_var_names:
        A = initialize_poly_matrix(
            lifted_variables_names,
        )
        sum_c_c_squared_constraint = constraint_sum_c_c_squared_redundant(
            A, c_var_name, c_var_names, hom_var_name
        )
        sum_c_c_squared_constraints.append(sum_c_c_squared_constraint)

    bool_constraints = []
    for c_var_name in c_var_names:
        A = initialize_poly_matrix(
            lifted_variables_names,
        )
        bool_constraint = constraint_boolean(A, c_var_name, hom_var_name)
        bool_constraints.append(bool_constraint)
    # We maybe dont need boolean constraints? Need to add how we control this parameter.
    A_sum = initialize_poly_matrix(lifted_variables_names)
    A_sum = constraint_association_variables_sum_one(A_sum, c_var_names, hom_var_name)

    constraint_dict = {
        "bool": bool_constraints,
        "prod_ci_cj": prod_ci_cj_constraints,
        "c_c_squared": sum_c_c_squared_constraints,
        "sum_one": [A_sum],
    }
    c_constraints = []
    constraint_names = []
    for name, constraint_list in constraint_dict.items():
        if name in discrete_variable_constraints:
            c_constraints += constraint_list
            constraint_names += [f"{name}_{lv1}" for lv1 in range(len(c_constraints))]

    # c_constraints = (
    # bool_constraints
    # + prod_ci_cj_constraints
    # + sum_c_c_squared_constraints
    # + [A_sum]
    # )
    # constraint_names = (
    # [f"bool_{lv1}" for lv1 in range(len(bool_constraints))]
    # + [f"prod_{lv1}" for lv1 in range(len(prod_ci_cj_constraints))]
    # + [f"c_c_squared_{lv1}" for lv1 in range(len(sum_c_c_squared_constraints))]
    # + ["sum"]
    # )

    return c_constraints, constraint_names


def idiag_constraints(
    vars_cols: List[str], vars_rows: List[str] = None
) -> List[PolyMatrix]:
    n = len(vars_cols)

    if vars_rows is None:
        vars_rows = vars_cols
        symmetric = True
    else:
        symmetric = False

    def initialize_A():
        return initialize_poly_matrix(
            var_names=vars_cols, vars_rows=vars_rows if not symmetric else None
        )

    A_list_1 = []

    # Off diagonal entries are zero
    for lv1 in range(n):
        for lv2 in range(n):
            if lv1 != lv2:
                A = initialize_A()
                if vars_rows[lv1] == vars_cols[lv2]:
                    A[vars_rows[lv1], vars_cols[lv2]] = 1
                else:
                    A[vars_rows[lv1], vars_cols[lv2]] = 0.5
                A_list_1.append(A)

    A_list_2 = []
    # Diagonal entries equal to each other
    for lv1 in range(1, n):
        A = initialize_A()

        if vars_rows[0] == vars_cols[0]:
            A[vars_rows[0], vars_cols[0]] = 1
        else:
            A[vars_rows[0], vars_cols[0]] = 0.5

        if vars_rows[lv1] == vars_cols[lv1]:
            A[vars_rows[lv1], vars_cols[lv1]] = -1
        else:
            A[vars_rows[lv1], vars_cols[lv1]] = -0.5
        A_list_2.append(A)

    return A_list_1 + A_list_2


def turn_scalar_constraint_to_col_constraint(
    hom_var: variables.HomogenizationVariable,
    A_in: PolyMatrix,
    dim,
    lifted_boolean_variables: List[variables.LiftedQcQpVariable],
    lifted_variable_names_all: List[str],
):
    # For boolean variables, they live as elements in a diagonal matrix.
    # THis takes a constraint defined over the boolean variables to
    # one defined over the diagonal matrix..
    A: PolyMatrix = initialize_poly_matrix(lifted_variable_names_all)
    # A[hom_var.col_names[dim], hom_var.col_names[dim]] = A_in[hom_var.name, hom_var.name]
    # TODO: A is symmetric, can do half the looping here
    for bool_var1 in [hom_var] + lifted_boolean_variables:
        for bool_var2 in [hom_var] + lifted_boolean_variables:
            A[bool_var1.col_names[dim], bool_var2.col_names[dim]] = A_in[
                bool_var1.name, bool_var2.name
            ]

    # import certifiable_uda_loc.utils.matrix_utils as mat_utils
    # import certifiable_uda_loc.utils.print_utils as print_utils

    # mat_utils.print_non_zero_entries(A, symmetric=True)

    return A


import certifiable_uda_loc.constraints.premultiplication as premultiplication


def theta_premultiplication_of_sum_theta_constraint(
    hom_var_name,
    c_var_names_dict,
):
    # For each list of lists in c_var_names_dict, the interfactor constraints are implemented.
    # c_var_names_dict: {key, [c_var_list1, c_var_list2, c_var_list2]}

    # Each list in c_var_names contains
    # a group of c variables that correspond to a single factor.
    # In that they sum to one.
    A_list = []
    lifted_variables_names_all = [hom_var_name]
    for name, c_var_names_list_of_lists in c_var_names_dict.items():
        for c_var_names in c_var_names_list_of_lists:
            lifted_variables_names_all += c_var_names

        for meas_idx1, meas_idx2 in itertools.product(
            range(len(c_var_names_list_of_lists)), range(len(c_var_names_list_of_lists))
        ):
            if meas_idx1 == meas_idx2:
                continue
            c_var_names_1 = c_var_names_list_of_lists[meas_idx1]
            c_var_names_2 = c_var_names_list_of_lists[meas_idx2]

            var_names = [hom_var_name] + c_var_names_1 + c_var_names_2

            A_sum_1d = initialize_poly_matrix(var_names)
            A_sum_1d = constraint_association_variables_sum_one(
                A_sum_1d, c_var_names_1, hom_var_name
            )

            for bool_var2 in c_var_names_2:
                A_list += premultiplication.premultiply_sum_constraint(
                    hom_var_name, A_sum_1d, c_var_names_1, bool_var2, var_names
                )

    return A_list


def get_c_constraints_per_factor(
    boolean_vars: List[pyfg_da.BooleanVariable],
    hom_var: variables.HomogenizationVariable,
    discrete_variable_constraints: List[str],
) -> List[PolyMatrix]:
    nx = hom_var.dims[0]

    bool_lifted_vars: List[variables.LiftedQcQpVariable] = [
        lifting.lift_scalar_variable(var, nx) for var in boolean_vars
    ]

    c_constraints, _ = create_c_constraint_list(
        hom_var.name,
        [var.name for var in bool_lifted_vars],
        discrete_variable_constraints,
    )
    return c_constraints


def set_discrete_variable_constraints(
    hom_var: variables.HomogenizationVariable,
    discrete_variable_constraint_names: List[str],
    uda_meas_list: List[pyfg_da.UnknownDataAssociationMeasurement],
    cont_column_variables: List[str],
    c_var_list_of_lists: List[List[str]],
) -> Tuple[List[PolyMatrix], List[PolyMatrix]]:
    c_constraints_per_factor = []
    for uda_meas in uda_meas_list:
        c_constraints_per_factor += get_c_constraints_per_factor(
            uda_meas.boolean_variables,
            hom_var,
            discrete_variable_constraint_names,
        )
    key = tuple(cont_column_variables)
    c_var_names_dict = {key: c_var_list_of_lists}
    c_constraints_theta_interfactor = theta_premultiplication_of_sum_theta_constraint(
        hom_var.name, c_var_names_dict
    )

    return c_constraints_per_factor, c_constraints_theta_interfactor
