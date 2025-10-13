from certifiable_uda_loc.constraints.problem_constraint_creation import (
    problem_constraints,
)
import dill as pickle
import numpy as np
import certifiable_uda_loc.problem as rf
from certifiable_uda_loc.toy_example import set_ground_truth_values

import numpy as np
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
from py_factor_graph.factor_graph import FactorGraphData

import certifiable_uda_loc.problem_setup as prob_setup
import cvxpy as cp
import certifiable_uda_loc.test_cases as rf_cases
import certifiable_uda_loc.trials as trials
import certifiable_uda_loc.redundant_constraints.construct_redundant as constr_red
import certifiable_uda_loc.utils.matrix_utils as mat_utils
import certifiable_uda_loc.variables as variables
import certifiable_uda_loc.test_cases as test_cases
from py_factor_graph.variables import PoseVariable2D
from certifiable_uda_loc.variables import ColumnQcQcpVariable
import certifiable_uda_loc.generate_se2_cases as gen_se2
from poly_matrix import PolyMatrix
import os
from certifiable_uda_loc.lifted_factor_graph import LiftedFactorGraph
from typing import List
import certifiable_uda_loc.subtask_factors as subtask_factors
import certifiable_uda_loc.constraints.problem_constraint_creation as problem_constraint_creation
from certifiable_uda_loc.settings import (
    ProblemSolutionSettings,
)


def test_constraints():
    np.random.seed(1)
    problem_settings = (
        ProblemSolutionSettings(
            sparse_bool_cont_variables=True,
            create_constraints_from_nullspace=False,
            create_discrete_variable_constraints_from_nullspace=False,
            discrete_variable_constraints=[
                "bool",
                "prod_ci_cj",
                "sum_one",
            ],
            use_sparse_matrices=True,
            sparsify_interfactor_discrete_constraints=True,
            sparsify_A_th_premultiplication=True,
            sparsify_cx_premultiplication_of_A_th=True,
            sparsify_cxi_cxj_premultiplication_of_A_th=True,
            sparsify_analytic_moment_constraints=True,
            sparsify_analytic_moment_constraints_2=True,
            sparsify_analytic_moment_constraints_3=True,
            sparsify_off_diag_boolean=False,
            sparsify_bool_product_moment=True,
            locked_first_pose=True,
            constraints_to_remove=[
                "c_constraints_lifted",
                "premultiply_A_xi_with_th1_th2_v1",
                # "cx_constraints",
                # "cxi_cxj_constraints",
                # "bool_product_moment",
                # "extra_off_diag_boolean",
                # "premultiply_A_xi_with_th1_th2_v2",
            ],
            solver_primal=False,
            solver_cost_matrix_adjust=True,
        ),
    )[0]
    noise_parameters = gen_se2.Se2CaseNoiseParameters(
        prior_pos_cov=1,
        prior_landmark_noise_corrupt_cov=1,
        prior_rot_kappa_langevin_inverse=0.1,
        prior_landmark_cov=0.1,
        rel_rot_kappa_langevin_inverse=1,
        rel_pos_cov=1,
        rel_landmark_meas_cov=0.1,
    )
    fg = gen_se2.create_se2_factor_graph(
        n_poses=2,
        n_landmarks=2,
        landmark_spread=5,
        meas_per_landmark=1,
        fraction_removal=0,
        uda_fraction=1,
        noise_parameters=noise_parameters,
    )
    fg: FactorGraphData = fg
    subtask_factors.convert_to_localization_task(fg)
    for var in fg.pose_variables:
        var: PoseVariable2D = var
        var.true_theta = np.random.random()
        var.true_position = np.random.random((2, 1))
        var.estimated_theta = np.random.random()
        var.estimated_position = np.random.random((2, 1))

    lifted_fg = LiftedFactorGraph(fg)
    opt_variables = lifted_fg.opt_variables
    lifted_variables_names_all = lifted_fg.lifted_variables_names_all
    uda_meas_list = lifted_fg.uda_meas_list
    hom_var = lifted_fg.hom_var
    _, constraint_dict_all = problem_constraints(
        lifted_fg,
        problem_settings=problem_settings,
    )
    print("Creating constraints..")
    A_list_mine, RHS_mine_list, lv_hom = (
        problem_constraint_creation.create_constraints_manually(
            constraint_dict_all, lifted_variables_names_all
        )
    )
    print("Done")
    feasible_point_mat_list = constr_red.get_feasible_points(
        uda_meas_list,
        opt_variables,
        lifted_variables_names_all,
        n_random_x_points_per_bool_combo=1,
    )
    XTXT_list = [x.toarray().T @ x.toarray() for x in feasible_point_mat_list]

    print("Testing feasible points.. ")
    for A, b in zip(A_list_mine, RHS_mine_list):
        for XTX in XTXT_list:
            err = np.abs(np.trace(A.toarray() @ XTX) - b)
            if np.abs(err) > 1e-5:
                import certifiable_uda_loc.utils.matrix_utils as mat_utils
                import certifiable_uda_loc.utils.print_utils as print_utils

                print_utils.pretty_print_array(
                    A.toarray(),
                    lifted_variables_names_all,
                    lifted_variables_names_all,
                )
                mat_utils.print_non_zero_entries(A, symmetric=True)
            assert np.abs(err) < 1e-5
    print("Great success!")


if __name__ == "__main__":
    test_constraints()
