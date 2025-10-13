import certifiable_uda_loc.utils.string_utils as string_utils
from loguru import logger
from certifiable_uda_loc.settings import ProblemSolutionSettings
import certifiable_uda_loc.discrete_variables as discrete_variables
from certifiable_uda_loc.problem import get_c_constraints_per_factor
import certifiable_uda_loc.constraints.premultiplication as premultiplication

import certifiable_uda_loc.utils.string_utils as string_utils
import certifiable_uda_loc.constraints.constraints_boolean as constraints_boolean
import itertools
from typing import List
import numpy as np
from certifiable_uda_loc.variables import LiftedQcQpVariable
from py_factor_graph.data_associations import (
    VectorVariable,
    UnknownDataAssociationMeasurement,
    ToyExamplePrior,
)
import certifiable_uda_loc.utils.matrix_utils as mat_utils

import py_factor_graph.data_associations as pyfg_da

import numpy as np
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
from typing import List, Tuple
from poly_matrix import PolyMatrix
from certifiable_uda_loc.utils.string_utils import var_to_colvar
import certifiable_uda_loc.utils.print_utils as print_utils
import certifiable_uda_loc.utils.matrix_utils as mat_utils
from typing import Union, Dict
import certifiable_uda_loc.variables as variables
import certifiable_uda_loc.lifting as lifting
from py_factor_graph.factor_graph import FactorGraphData
import certifiable_uda_loc.lifted_factor_graph as lifted_factor_graph
from certifiable_uda_loc.redundant_constraints import (
    construct_redundant as constr_red,
)
from certifiable_uda_loc.lifted_factor_graph import UdaMeasurementGroup


def create_constraints_manually(
    constraint_dict_all, lifted_variables_names_all, sparse_matrices: bool = False
):

    A_list_all = []
    RHS_all = []
    lv_hom = 1
    for name, A_list in constraint_dict_all.items():
        if name == "homogenization_constraint":
            RHS = 1
            hom_constraint_idx = lv_hom
        else:
            RHS = 0
        for A in A_list:
            A_constr = A
            if not sparse_matrices:
                A_constr = initialize_poly_matrix(lifted_variables_names_all)
                if A.symmetric:
                    A_constr = A_constr + A
                else:
                    for var_i in A.variable_dict_i.keys():
                        for var_j in A.variable_dict_j.keys():
                            A_constr[var_i, var_j] = A[var_i, var_j]

                # cvxpyConstraintList.append(
                # cp.trace(A_constr.get_matrix_sparse() @ Z) == RHS
                # )
            lv_hom = lv_hom + 1
            A_list_all.append(A_constr)
            RHS_all.append(RHS)
    return A_list_all, RHS_all, lv_hom


def create_constraints_automatically(
    uda_meas_list, opt_variables, lifted_variables_names_all, formulation="with_x"
):
    feasible_point_mat_list = constr_red.get_feasible_points(
        uda_meas_list,
        opt_variables,
        lifted_variables_names_all,
    )
    return create_constraints_from_feasible_points(
        feasible_point_mat_list, lifted_variables_names_all
    )


def create_constraints_from_feasible_points(
    feasible_point_mat_list, lifted_variables_names_all
):
    m_vec_list = constr_red.find_sparse_constraints(
        feasible_point_mat_list,
        [],
        lifted_variables_names_all,
        formulation="qrp",
        # formulation="non-orthogonal",
        verbose=False,
    )
    RHS_list = []
    A_list = []
    for lv1, m in enumerate(m_vec_list):
        m_vec = mat_utils.round_vec(m)
        A_np = mat_utils.vech2mat(m_vec, N=len(lifted_variables_names_all))
        A = mat_utils.np_to_polymatrix(
            lifted_variables_names_all, lifted_variables_names_all, A_np
        )
        A_zero = initialize_poly_matrix(lifted_variables_names_all)
        A = A + A_zero

        assert list(A.variable_dict_i.keys()) == lifted_variables_names_all

        A_list.append(A)
        RHS_list.append(0)
    return A_list, RHS_list


def get_lifted_variable_names_from_uda_group_list(
    hom_var: variables.HomogenizationVariable,
    uda_group_list: List[UdaMeasurementGroup],
    lifted_variable_names_filter: List[str],
):
    # This is a bit of a fustercluck.
    # We lost the information on which columns belong to which optimization variable
    # which is annoying in making the bool-cont variables. # TODO first thing tomorrow.
    lifted_variable_names = []
    lifted_variable_names += hom_var.column_names()
    nx = hom_var.dims[0]
    for uda_group in uda_group_list:
        # Okay that thing actually takes in the full continuous opt_variable...
        # not just the columns.
        bool_lifted_vars: List[variables.LiftedQcQpVariable] = [
            lifting.lift_scalar_variable(var, nx) for var in uda_group.boolean_variables
        ]

        bool_cont_lifted_vars: List[variables.LiftedQcQpVariable] = []
        for bool_var in uda_group.boolean_variables:
            for cont_var in uda_group.opt_variables:
                cont_var: LiftedQcQpVariable = cont_var
                bool_cont_lifted_var = variables.LiftedQcQpVariable(
                    cont_var.dims,
                    string_utils.bool_cont_var_name(bool_var.name, cont_var.name),
                )
                bool_cont_lifted_vars.append(bool_cont_lifted_var)
        for var in bool_lifted_vars + bool_cont_lifted_vars:
            lifted_variable_names += var.col_names
        lifted_variable_names += uda_group.cont_column_variables

    lifted_variable_names_temp = []
    for var in lifted_variable_names:
        if (
            var not in lifted_variable_names_temp
            and var in lifted_variable_names_filter
        ):
            lifted_variable_names_temp.append(var)

    lifted_variable_names = lifted_variable_names_temp
    return lifted_variable_names


# Alright so we need to do a bit of a refactor here
# To work in terms of these cliques.
# Need to do it both for Q and for the constraint matrices.
# 1. Can just add A_list for each Q as property of the UdaMeasurementGroup?
#   1a. Need to handle constraints and costs that are NOT part of UDA group.
# 2. How do I handle the cost bookkeeping... has to link some cliques.


def problem_constraints(
    lifted_fg: lifted_factor_graph.LiftedFactorGraph,
    problem_settings: ProblemSolutionSettings,
) -> Tuple[List[str], Dict]:

    create_discrete_variable_constraints_from_nullspace = (
        problem_settings.create_discrete_variable_constraints_from_nullspace,
    )[0]

    bool_lifted_vars = lifted_fg.bool_lifted_vars
    constraint_dict_all = {}
    hom_var = lifted_fg.hom_var
    lifted_variables_names_all = lifted_fg.lifted_variables_names_all
    nx = lifted_fg.nx
    opt_variables = lifted_fg.opt_variables
    # vector_var_names = lifted_fg.cont_column_variable_nams
    vector_var_names = [var.name for var in lifted_fg.cont_column_variables]
    uda_meas_list = lifted_fg.uda_meas_list
    boolean_vars_all = lifted_fg.bool_lifted_vars
    bool_vars_unlifted = []
    for uda_meas in uda_meas_list:
        bool_vars_unlifted += uda_meas.boolean_variables
    x_variable_names = vector_var_names

    # Perhaps a cleaner way of doing all this UDA stuff would be to create one big uda_meas group
    # for the non-sparse case. Eh, whatever. TODO
    """
    %%%%%%%%%%%%% IDIAG CONSTRAINTS %%%%%%%%%%
    """
    logger.info(f"\n\nStarting problem constraint setup..")
    idiag_constraints = []
    for var in bool_lifted_vars:
        idiag_constraints += constraints_boolean.idiag_constraints(var.column_names())

    for var in bool_lifted_vars:
        idiag_constraints += constraints_boolean.idiag_constraints(
            var.column_names(), hom_var.column_names()
        )
    constraint_dict_all["idiag_constraints"] = idiag_constraints
    """
    %%%%%%%%%%%%% HOMOGENIZATION VARIABLE CONSTRAINTS %%%%%%%%%%
    """

    homogenization_constraint = hom_var.qcqpConstraintMatrices()[0]
    idiag_hom_constraints = constraints_boolean.idiag_constraints(
        hom_var.column_names()
    )
    constraint_dict_all["homogenization_constraint"] = homogenization_constraint
    constraint_dict_all["idiag_hom_constraints"] = idiag_hom_constraints

    logger.info(f"\nDone Hom and iDiag constraints")
    rot_vars: List[variables.RotationVariable] = []
    for var in opt_variables:
        if isinstance(var, variables.RotationVariable):
            rot_vars.append(var)
    """
    GIVEN [1,  xi^trans] mbf{A}_xi [1  xi], add into problem
    """
    A_dict_xi_by_variable, A_dict_xi_intervariable = (
        lifted_fg.get_continuous_variable_constraints()
    )
    # Locked first pose..
    if problem_settings.locked_first_pose:
        for var in lifted_fg.locked_pose_variables:
            key = tuple(var.col_names)
            if key not in A_dict_xi_by_variable:
                A_dict_xi_by_variable[key] = []
            for col_var in var.column_variables:
                x_true = col_var.true_value.reshape(-1, 1)
                # A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)
                A: PolyMatrix = PolyMatrix()
                A[hom_var.col_names[0], hom_var.col_names[0]] = (
                    x_true.T @ x_true
                ).squeeze()
                for lv1 in range(nx):
                    A[hom_var.col_names[lv1], col_var.name] = -x_true[lv1, 0]
                A[col_var.name, col_var.name] = 1
                A_dict_xi_by_variable[key] += [A]

    A_list_xi_all = []
    for key in A_dict_xi_by_variable:
        A_list_xi_all += A_dict_xi_by_variable[key]

    constraint_dict_all["cont_variable_constraints_xi_pure"] = A_list_xi_all
    logger.info(f"\nDone A_xi_all (continuous var) constraints")

    """
    %%%%%%%%%%%%%%%% BOOLEAN VARIABLE CONSTRAINTS PER FACTOR
    """
    for uda_group in lifted_fg.uda_group_list_sparse + lifted_fg.uda_group_list_dense:
        uda_group.reorder_column_variables_based_on_top_level(
            lifted_variables_names_all
        )
        constraints_factor, c_constraints_theta_interfactor = (
            constraints_boolean.set_discrete_variable_constraints(
                hom_var,
                problem_settings.discrete_variable_constraints,
                uda_group.uda_meas_list,
                uda_group.cont_column_variables,
                c_var_list_of_lists=uda_group.c_var_list_of_lists,
            )
        )
        uda_group.c_constraints_all = (
            constraints_factor + c_constraints_theta_interfactor
        )
    uda_group_list_sparse = lifted_fg.uda_group_list_sparse
    uda_group_list_dense = lifted_fg.uda_group_list_dense
    uda_group_dense = lifted_fg.uda_group_list_dense[0]

    def switch_between_group_lists(sparse_switch: bool):
        if sparse_switch:
            return uda_group_list_sparse
        else:
            return uda_group_list_dense

    if not create_discrete_variable_constraints_from_nullspace:
        """
        BOOLEAN VARIABLE CONSTRAINTS PER FACTOR
        """
        # c_constraints_all = uda_group_list_dense[0].c_constraints_all
        uda_group_list = uda_group_list_sparse

        c_constraints_all = uda_group_list_dense[0].c_constraints_all

        c_var_names_dict_flattened = {}
        c_var_names_dict_flattened["all"] = uda_group_dense.c_var_list_of_lists

        """
        BOOLEAN VARIABLE CONSTRAINTS INTERFACTOR - PREMULTIPLICATION OF SUM CONSTRAINT WITH ALL OTHER THETAS
        """
        uda_group_list = switch_between_group_lists(
            problem_settings.sparsify_interfactor_discrete_constraints
        )
        c_var_names_dict = {
            tuple(uda_group.cont_column_variables): uda_group.c_var_list_of_lists
            for uda_group in uda_group_list
        }
        c_constraints_theta_interfactor = (
            constraints_boolean.theta_premultiplication_of_sum_theta_constraint(
                hom_var.name, c_var_names_dict
            )
        )

        c_constraints_all += c_constraints_theta_interfactor
        logger.info(
            f"\nManually found {len(c_constraints_all)} for just the discrete variables"
        )

    if create_discrete_variable_constraints_from_nullspace:
        logger.info(f"\nAuto creating constraints for discrete variables")
        c_var_names_list_of_lists = []
        for uda_meas in uda_meas_list:
            bool_var_names = [var.name for var in uda_meas.boolean_variables]
            c_var_names_list_of_lists.append(bool_var_names)

        lifted_variables_discrete = [hom_var.name]
        for c_var_list in c_var_names_list_of_lists:
            lifted_variables_discrete += c_var_list

        feas_point_vec_list = discrete_variables.create_discrete_feasible_point_vectors(
            lifted_variables=lifted_variables_discrete,
            c_var_names_list_of_lists=c_var_names_list_of_lists,
        )
        feas_point_vec_list = [v.reshape(-1, 1) for v in feas_point_vec_list]

        feasible_point_mat_list = [
            mat_utils.np_to_polymatrix(
                lifted_variables_discrete, lifted_variables_discrete, v @ v.T
            )
            for v in feas_point_vec_list
        ]
        A_c_constraints, _ = create_constraints_from_feasible_points(
            feasible_point_mat_list, lifted_variables_discrete
        )
        c_constraints_all = A_c_constraints
        # Now have to lift these to the setting with the higher variables that I have..
        logger.info(
            f"\nAutomatically found {len(c_constraints_all)} for just the discrete variables"
        )

    A_list = []
    # Premultiplication of A_xi constraints..
    """
    GIVEN [1, xi trans] mbf{A}_xi [1 \ xi], additional constraint
    yielded by premultiplication with theta
    """
    logger.info(
        f"Creating theta premultiply A xi constraints. Sparse: {problem_settings.sparsify_A_th_premultiplication}"
    )

    uda_group_list = switch_between_group_lists(
        problem_settings.sparsify_A_th_premultiplication
    )

    A_list = []
    for var_set, A_xi_list in A_dict_xi_by_variable.items():
        # var_set are the set of variables that
        # the constraint depends on
        for uda_group in uda_group_list:
            include_flag = any(
                var in var_set for var in uda_group.cont_column_variables
            )
            if not include_flag:
                continue
            for A_xi in A_xi_list:
                for bool_var in uda_group.boolean_variables:
                    bool_lifted_var = lifting.lift_scalar_variable(bool_var, nx)
                    A_list += premultiplication.theta_premultiply_A_xi(
                        A_xi,
                        bool_lifted_var,
                        var_set,
                        hom_var,
                        lifted_variables_names_all,
                    )
    logger.info(f"Added {len(A_list)} theta premultiply A xi.")

    constraint_dict_all["premultiply_A_xi_with_bool"] = A_list

    A_list_v1 = []
    A_list_v2 = []
    uda_group_list = switch_between_group_lists(
        problem_settings.sparsify_A_th_premultiplication
    )
    boolean_iterables = []
    for var_set, A_xi_list in A_dict_xi_by_variable.items():
        boolean_iterable_list = []
        for uda_group in uda_group_list:
            include_flag = (
                any(var in var_set for var in uda_group.cont_column_variables)
                or not problem_settings.sparsify_A_th_premultiplication
            )
            if include_flag:
                boolean_iterable_list.append(
                    itertools.product(
                        uda_group.boolean_variables, uda_group.boolean_variables
                    )
                )
        boolean_iterables.append(list(itertools.chain(*boolean_iterable_list)))

    for lv1, (var_set, A_xi_list) in enumerate(A_dict_xi_by_variable.items()):
        for A_xi in A_xi_list:
            for bool_var1, bool_var2 in boolean_iterables[lv1]:
                if bool_var1.name == bool_var2.name:
                    continue
                bool_var1 = lifting.lift_scalar_variable(bool_var1, nx)
                bool_var2 = lifting.lift_scalar_variable(bool_var2, nx)

                A_list_v1 += (
                    premultiplication.theta_premultiply_A_xi_with_theta_i_theta_j(
                        A_xi,
                        bool_var1,
                        bool_var2,
                        var_set,
                        hom_var,
                        lifted_variables_names_all,
                        version2=False,
                    )
                )
                A_list_v2 += (
                    premultiplication.theta_premultiply_A_xi_with_theta_i_theta_j(
                        A_xi,
                        bool_var1,
                        bool_var2,
                        var_set,
                        hom_var,
                        lifted_variables_names_all,
                        version2=True,
                    )
                )

    constraint_dict_all["premultiply_A_xi_with_th1_th2_v1"] = A_list_v1
    constraint_dict_all["premultiply_A_xi_with_th1_th2_v2"] = A_list_v2

    logger.info(
        f"\nNumber of  premultiply_A_xi_with_th1_th2_v1 constraints: {len(A_list_v1)}"
    )
    logger.info(
        f"\nNumber of  premultiply_A_xi_with_th1_th2_v2 constraints: {len(A_list_v2)}"
    )

    """
    LIFT THE SCALAR CONSTRAINTS TO MULTIPLE DIMENSIONS
    """
    c_constraints_lifted = []
    for lv_dim in range(nx):
        for A_in in c_constraints_all:
            c_constraints_lifted.append(
                constraints_boolean.turn_scalar_constraint_to_col_constraint(
                    hom_var,
                    A_in,
                    lv_dim,
                    lifted_fg.bool_lifted_vars,
                    lifted_variables_names_all,
                )
            )
    constraint_dict_all["c_constraints_lifted"] = c_constraints_lifted

    """
    PREMULTIPLICATION OF A_theta WITH x_i as well as x_i^\trans x_j
    """

    cx_constraints = []
    cxi_cxj_constraints = []
    # Need a similar story to the boolean sparsificaiton...
    logger.info(
        f"Creating constraints for cx premultiplication. Sparsify: {problem_settings.sparsify_cx_premultiplication_of_A_th}"
    )
    uda_group_list = switch_between_group_lists(
        problem_settings.sparsify_cx_premultiplication_of_A_th
    )

    cx_constraints = []
    for uda_group in uda_group_list:
        for A_th in uda_group.c_constraints_all:
            for x in uda_group.cont_column_variables:
                bool_lifted_vars_lv1 = [
                    lifting.lift_scalar_variable(var, nx)
                    for var in uda_group.boolean_variables
                ]
                cx_constraints += premultiplication.x_premultiply_A_theta(
                    x,
                    lifted_variables_names_all,
                    hom_var,
                    bool_lifted_vars_lv1,
                    A_th,
                )

    logger.info(f"Done. Number of constraints: {len(cx_constraints)}")
    constraint_dict_all["cx_constraints"] = cx_constraints

    logger.info(
        f"Creating constraints for cx_i cx_j premultiplication. Sparsify: {problem_settings.sparsify_cxi_cxj_premultiplication_of_A_th}"
    )
    cxi_cxj_constraints = []
    uda_group_list = switch_between_group_lists(
        problem_settings.sparsify_cxi_cxj_premultiplication_of_A_th
    )

    for uda_group in uda_group_list:
        for A_th in uda_group.c_constraints_all:
            bool_lifted_vars_lv1 = [
                lifting.lift_scalar_variable(var, nx)
                for var in uda_group.boolean_variables
            ]

            if len(uda_group.cont_column_variables) == 1:
                combos = [
                    (
                        uda_group.cont_column_variables[0],
                        uda_group.cont_column_variables[0],
                    )
                ]
            else:
                combos = list(
                    itertools.product(
                        uda_group.cont_column_variables, uda_group.cont_column_variables
                    )
                )
            for x1, x2 in combos:
                A_list = premultiplication.xi_xj_premultiply_A_theta(
                    x1,
                    x2,
                    lifted_variables_names_all,
                    [var.name for var in bool_lifted_vars_lv1],
                    A_th,
                )
                cxi_cxj_constraints += A_list
    logger.info(f"Done. Number of constraints: {len(cxi_cxj_constraints)}")

    constraint_dict_all["cxi_cxj_constraints"] = cxi_cxj_constraints

    """
    %%%% ADD SOME MOMENT CONSTRAINTS THAT COME UP DUE TO THE COLUMN STRUCTURE.. 
    """
    uda_group_list = switch_between_group_lists(
        problem_settings.sparsify_analytic_moment_constraints
    )
    A_list = []
    for uda_group in uda_group_list:
        bool_lifted_vars_lv1 = [
            lifting.lift_scalar_variable(var, nx) for var in uda_group.boolean_variables
        ]
        for lv_dim in range(nx):
            for x_var in uda_group.cont_column_variables:
                for bool_var in bool_lifted_vars_lv1:
                    bool_cont_name = string_utils.bool_cont_var_name(
                        bool_var.name, x_var
                    )
                    A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)
                    A[hom_var.col_names[lv_dim], bool_cont_name] = 1
                    A[bool_var.col_names[lv_dim], x_var] = -1
                    A_list.append(A)
    constraint_dict_all["moment_constraints_columns_analytic"] = A_list
    logger.info(f"Added moment_constraints_columns_analytic. Amount: {len(A_list)}")

    uda_group_list = switch_between_group_lists(
        problem_settings.sparsify_analytic_moment_constraints_2
    )
    A_list = []
    for uda_group in uda_group_list:
        bool_lifted_vars_lv1 = [
            lifting.lift_scalar_variable(var, nx) for var in uda_group.boolean_variables
        ]
        # for lv_dim in range(nx):
        lv_dim = 0
        for x_var in uda_group.cont_column_variables:
            for bool_var1, bool_var2 in itertools.combinations_with_replacement(
                bool_lifted_vars_lv1, 2
            ):
                if bool_var1.name == bool_var2.name:
                    continue
                bool_cont_name1 = string_utils.bool_cont_var_name(bool_var1.name, x_var)
                bool_cont_name2 = string_utils.bool_cont_var_name(bool_var2.name, x_var)
                A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)
                A[bool_cont_name1, var_to_colvar(bool_var2.name, lv_dim)] = 1
                A[bool_cont_name2, var_to_colvar(bool_var1.name, lv_dim)] = -1
                A_list.append(A)
    constraint_dict_all["moment_constraints_columns_analytic_2"] = A_list
    logger.info(f"Added moment_constraints_columns_analytic_2. Amount: {len(A_list)}")

    uda_group_list = switch_between_group_lists(
        problem_settings.sparsify_analytic_moment_constraints_3
    )
    A_list = []
    for uda_group in uda_group_list:
        bool_lifted_vars_lv1 = [
            lifting.lift_scalar_variable(var, nx) for var in uda_group.boolean_variables
        ]
        for x_var1, x_var2 in itertools.combinations(
            uda_group.cont_column_variables, 2
        ):
            for bool_var1, bool_var2 in itertools.combinations_with_replacement(
                bool_lifted_vars_lv1, 2
            ):
                if bool_var1.name == bool_var2.name:
                    continue
                pair00 = string_utils.bool_cont_var_name(bool_var1.name, x_var1)
                pair01 = string_utils.bool_cont_var_name(bool_var2.name, x_var2)
                pair10 = string_utils.bool_cont_var_name(bool_var1.name, x_var2)
                pair11 = string_utils.bool_cont_var_name(bool_var2.name, x_var1)

                A: PolyMatrix = initialize_poly_matrix(lifted_variables_names_all)
                A[pair00, pair01] = 1
                A[pair10, pair11] = -1
                A_list.append(A)
        constraint_dict_all["moment_constraints_columns_analytic_3"] = A_list
    logger.info(f"Added moment_constraints_columns_analytic_3. Amount: {len(A_list)}")

    """
    %%%% SOME HAPPY EXTRAS NEEDED FOR THE BOOLEANS
    """
    uda_group_list = switch_between_group_lists(
        problem_settings.sparsify_off_diag_boolean
    )
    A_list = []
    for uda_group in uda_group_list:
        bool_lifted_vars_lv1 = [
            lifting.lift_scalar_variable(var, nx) for var in uda_group.boolean_variables
        ]
        for bool1, bool2 in itertools.product(
            bool_lifted_vars_lv1, bool_lifted_vars_lv1
        ):
            for dim1, dim2 in itertools.combinations(range(nx), 2):
                if dim1 == dim2:
                    continue
                else:
                    A_var_1 = var_to_colvar(bool1.name, dim1)
                    A_var_2 = var_to_colvar(bool2.name, dim2)
                    A: PolyMatrix = initialize_poly_matrix(
                        [A_var_1], vars_rows=[A_var_2]
                    )
                    A[A_var_1, A_var_2] = 1
                    A_list.append(A)
    constraint_dict_all["extra_off_diag_boolean"] = A_list
    logger.info(f"Added extra_off_diag_boolean. Amount: {len(A_list)}")

    uda_group_list = switch_between_group_lists(
        problem_settings.sparsify_off_diag_boolean
    )
    A_list = []
    for uda_group in uda_group_list:
        bool_lifted_vars_lv1 = [
            lifting.lift_scalar_variable(var, nx) for var in uda_group.boolean_variables
        ]
        for bool1, bool2 in itertools.combinations(bool_lifted_vars_lv1, 2):
            dim1 = 0
            dim2 = 1
            lifted_vars_lv1 = [
                var_to_colvar(bool1.name, dim1),
                var_to_colvar(bool2.name, dim1),
                var_to_colvar(bool1.name, dim2),
                var_to_colvar(bool2.name, dim2),
            ]
            if any([var not in lifted_variables_names_all for var in lifted_vars_lv1]):
                continue

            A: PolyMatrix = initialize_poly_matrix(
                lifted_variables_names_all, vars_rows=lifted_variables_names_all
            )

            A[var_to_colvar(bool1.name, dim1), var_to_colvar(bool2.name, dim1)] = 1
            A[var_to_colvar(bool1.name, dim2), var_to_colvar(bool2.name, dim2)] = -1

            A_list.append(A)
    constraint_dict_all["bool_product_moment"] = A_list

    for key in problem_settings.constraints_to_remove:
        del constraint_dict_all[key]

    if problem_settings.sparse_bool_cont_variables:
        # To be honest this is a little bit cursed.
        # We will see if this has an effect on performance lol
        logger.info(
            f"\nRemoving constraints that do not concern only sparse variables.."
        )
        for key, A_list in constraint_dict_all.items():
            idx_to_remove = []
            for lv1, A in enumerate(A_list):
                for var in A.variable_dict_i.keys():
                    if var not in lifted_variables_names_all:
                        idx_to_remove.append(lv1)
                        break
                if not A.symmetric:
                    for var in A.variable_dict_j.keys():
                        if var not in lifted_variables_names_all:
                            idx_to_remove.append(lv1)
                            break

                # for var1, var2 in itertools.combinations_with_replacement(
                #     A.variable_dict_i.keys(), 2
                # ):
                #     if (
                #         var1 not in lifted_variables_names_all
                #         or var2 not in lifted_variables_names_all
                #     ):
                #         idx_to_remove.append(lv1)
                #         break
            A_list = [A for lv1, A in enumerate(A_list) if lv1 not in idx_to_remove]
            constraint_dict_all[key] = A_list

    logger.info(f"\nDone with initial constraint setup")

    return lifted_variables_names_all, constraint_dict_all
