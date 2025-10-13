from loguru import logger
from certifiable_uda_loc.settings import (
    Se2CaseNoiseParameters,
    ProblemSolutionSettings,
)
from py_factor_graph.data_associations import VectorVariable
import certifiable_uda_loc.constraints.problem_constraint_creation as problem_constraint_creation
import certifiable_uda_loc.lifted_factor_graph as lifted_factor_graph
from certifiable_uda_loc import problem
from certifiable_uda_loc.redundant_constraints import (
    construct_redundant as constr_red,
)
import numpy as np
import certifiable_uda_loc.problem as rf

import numpy as np
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
from py_factor_graph.factor_graph import FactorGraphData

from certifiable_uda_loc.unlifting import (
    unlift,
    unlift_lifted_variables_to_factor_graph,
)
from certifiable_uda_loc.cost_matrix import get_cost_matrix

import certifiable_uda_loc.problem_setup as prob_setup
import cvxpy as cp
import certifiable_uda_loc.test_cases as rf_cases
from dataclasses import dataclass
from typing import List
from certifiable_uda_loc.utils.print_utils import pretty_print_array
import certifiable_uda_loc.utils.matrix_utils as mat_utils
import certifiable_uda_loc.utils.print_utils as print_utils
import argparse
import certifiable_uda_loc.variables as variables

from typing import Union
from certifiable_uda_loc.variables import LiftedQcQpVariable
from py_factor_graph.data_associations import UnknownDataAssociationMeasurement

from certifiable_uda_loc.variables import ColumnQcQcpVariable
from poly_matrix import PolyMatrix
from py_factor_graph.variables import (
    POSE_VARIABLE_TYPES,
    LANDMARK_VARIABLE_TYPES,
    PoseVariable2D,
    PoseVariable3D,
    LandmarkVariable2D,
)
from collections import namedtuple

# from loguru import logger
import io
from contextlib import redirect_stdout, redirect_stderr


class ProblemQCQP:
    def __init__(
        self,
        A_list: List[PolyMatrix],
        RHS_list: List[float],
        Q: PolyMatrix,
        constraint_dict_all=None,
    ):
        self.A_list = A_list
        self.RHS_list = RHS_list
        self.Q = Q
        self.lifted_variable_names_all = list(Q.variable_dict_i.keys())
        self.constraint_dict_all = constraint_dict_all
        self.hom_constraint_idx = None
        for lv, RHS in enumerate(RHS_list):
            if np.abs(RHS - 1) < 1e-5:
                self.hom_constraint_idx = lv + 1


@dataclass
class MonteCarloParameters:
    run_id: str
    nx_list: List[int]
    n_components_per_factor_list: List[int]
    n_trials_per_run: int
    create_constraints_from_nullspace: bool
    verbose: bool
    formulation: str  # either "with_x" or "without_x"
    num_parallel_jobs: int

    def __repr__(self):
        repr_str = f"Run id:{self.run_id}\
                    \nNx list: {self.nx_list}\
                    \nNumber of components per factor list: {self.n_components_per_factor_list}\
                    \nNumber of trials per run: {self.n_trials_per_run}\
                    \nCreate constraints from nullspace: {self.create_constraints_from_nullspace}\
                    \nVerbose: {self.verbose}\
                    \nFormulation: {self.formulation}\
                    \nNumber of parallel jobs: {self.num_parallel_jobs}"
        return repr_str

    @staticmethod
    def from_args(args) -> "MonteCarloParameters":
        return MonteCarloParameters(
            run_id=args.run_id,
            nx_list=args.nx_list,
            n_components_per_factor_list=args.n_components_per_factor_list,
            n_trials_per_run=args.n_trials_per_run,
            create_constraints_from_nullspace=args.create_constraints_from_nullspace,
            verbose=args.verbose,
            formulation=args.formulation,
            num_parallel_jobs=args.num_parallel_jobs,
        )


class MonteCarloSummary:

    def __init__(
        self,
        true_cost: float,
        est_cost: float,
        dual_cost: float,
        duality_gap: float,
        prob_value: float,
        eigvals: List[float],
        nx: int,
        state_error: float,
    ):
        self.true_cost = true_cost
        self.est_cost = est_cost
        self.dual_cost = dual_cost
        self.duality_gap = duality_gap
        self.prob_value = prob_value
        self.eigvals = eigvals
        self.nx = nx
        self.state_error = state_error
        if self.eigvals is not None:
            self.eigval_ratio = np.real_if_close(
                self.eigvals[self.nx - 1] / self.eigvals[self.nx]
            )
            self.log_eigval_ratio = np.log10(np.abs(self.eigval_ratio))
        else:
            self.log_eigval_ratio = None

    def __repr__(self):
        if self.true_cost is None:
            return "MC summary of failed optimization </3"

        return f"True cost: {self.true_cost},\
              \nEstimated cost: {self.est_cost},\
              \nDual cost: {self.dual_cost},\
              \nDuality gap: {self.duality_gap},\
              \nFirst nx+1 eigvals: {self.eigvals[:self.nx+1]},\
              \nEigenvalue Ratio: {self.eigval_ratio:.2e}\
              \nProblem value: {self.prob_value}\
              \nState Error: {self.state_error}\
              \nDims: {self.nx}"


class MonteCarloResultSdp:
    def __init__(
        self,
        fg: FactorGraphData,
        prob_qcqp: ProblemQCQP,
        Z: np.ndarray,
        sdp_cost: float,
        sdp_dual_cost: float,
        H: np.ndarray,
        yvals: np.ndarray,
        msg: str,
    ):
        self.fg = fg
        # self.prob_qcqp = prob_qcqp
        self.lifted_variable_names_all = prob_qcqp.lifted_variable_names_all
        self.Q_all = prob_qcqp.Q
        self.Z = Z
        self.sdp_cost = sdp_cost
        self.sdp_dual_cost = sdp_dual_cost
        self.H = H
        self.yvals = yvals
        self.msg = msg
        self.fg_lifted = lifted_factor_graph.LiftedFactorGraph(fg)

    def postprocess(self, verbose=True):

        opt_variables = self.fg_lifted.opt_variables
        pose_to_lifted_var_indices = self.fg_lifted.pose_to_lifted_var_indices
        landmark_to_lifted_var_indices = self.fg_lifted.landmark_to_lifted_var_indices

        nx = opt_variables[0].column_variables[0].dims[0]
        hom_var = variables.HomogenizationVariable(
            (nx, nx), "one", true_value=np.eye(nx)
        )

        # lifted_variables_names_all = self.prob_qcqp.lifted_variable_names_all
        lifted_variables_names_all = self.lifted_variable_names_all

        if self.Z is None or np.isnan(self.Z).any():
            mc_summary = MonteCarloSummary(
                true_cost=None,
                est_cost=None,
                dual_cost=None,
                duality_gap=None,
                prob_value=None,
                eigvals=None,
                nx=None,
                state_error=None,
            )
            return mc_summary
        # print_utils.pretty_print_array(self.Z[:2])
        eigvals = np.linalg.eigvals(self.Z)
        eigvals = np.sort(eigvals)[::-1]
        X_est_np = self.Z[:nx, :]  # Assumption that first rows are hom var
        rows = hom_var.column_names()
        X_est_poly = initialize_poly_matrix(
            var_names=lifted_variables_names_all, vars_rows=rows
        )
        for lv1, row_var in enumerate(rows):
            for lv2, col_var in enumerate(lifted_variables_names_all):
                X_est_poly[row_var, col_var] = X_est_np[lv1, lv2]

        uda_meas_list = self.fg.unknown_data_association_measurements

        unlift(opt_variables, uda_meas_list, X_est_poly, hom_var)

        unlift_lifted_variables_to_factor_graph(
            self.fg,
            opt_variables,
            pose_to_lifted_var_indices,
            landmark_to_lifted_var_indices,
        )
        rotation_error_dict = {}
        position_error_dict = {}
        landmark_error_dict = {}
        vector_variable_error_dict = {}
        for var in self.fg.pose_variables:
            var: variables.PoseVariable2D = var
            rotation_error_dict[var.name + "_rot"] = (
                np.linalg.norm(
                    var.true_rotation_matrix - var.estimated_rotation_matrix, "fro"
                )
                ** 2
            )

            position_error_dict[var.name + "_pos"] = (
                np.linalg.norm(var.true_position - var.estimated_position) ** 2
            )

        for var in self.fg.landmark_variables:
            var: variables.LandmarkVariable2D = var
            landmark_error_dict[var.name] = (
                np.linalg.norm(var.true_position - var.estimated_position) ** 2
            )

        for var in self.fg.vector_variables:
            var: VectorVariable
            vector_variable_error_dict[var.name] = (
                np.linalg.norm(var.true_value - var.estimated_value) ** 2
            )

        all_var_error_dict = {
            **rotation_error_dict,
            **position_error_dict,
            **landmark_error_dict,
            **vector_variable_error_dict,
        }
        state_error = 0.0
        for key, value in all_var_error_dict.items():
            state_error += value
        boolean_vars_all = []
        for uda_meas in uda_meas_list:
            boolean_vars_all += uda_meas.boolean_variables

        X_true = rf.lift_feasible_point(
            boolean_vars=boolean_vars_all,
            continuous_vars=opt_variables,
            column_names_all=lifted_variables_names_all,
            true_or_estimated_value="true",
        )

        X_est = rf.lift_feasible_point(
            boolean_vars=boolean_vars_all,
            continuous_vars=opt_variables,
            column_names_all=lifted_variables_names_all,
            true_or_estimated_value="estimated",
        )

        Q_all = self.Q_all
        true_cost = np.trace(Q_all.toarray() @ X_true.toarray().T @ X_true.toarray())
        est_cost = np.trace(Q_all.toarray() @ X_est.toarray().T @ X_est.toarray())
        sdp_recalc_cost = np.trace(Q_all.toarray() @ self.Z) / 2
        sdp_recalc_cost = np.trace(
            Q_all.toarray() @ self.Z[:2, :].T @ self.Z[:2, :] / 2
        )

        dual_cost = self.sdp_dual_cost
        dx_list = []
        for var in opt_variables:
            dx = var.true_value - var.estimated_value
            dx_list.append(dx)

        duality_gap = est_cost - dual_cost
        mc_summary = MonteCarloSummary(
            true_cost=true_cost,
            est_cost=est_cost,
            dual_cost=dual_cost,
            duality_gap=duality_gap,
            prob_value=self.sdp_cost,
            eigvals=eigvals,
            nx=nx,
            state_error=state_error,
        )

        if np.isnan(mc_summary.eigval_ratio):
            logger.info("Eigenvalue ratio is nan")
        return mc_summary


def setup_problem_qcqp(
    lifted_fg: lifted_factor_graph.LiftedFactorGraph,
    problem_settings: ProblemSolutionSettings,
    verbose: bool = True,
) -> ProblemQCQP:

    create_constraints_from_nullspace = (
        problem_settings.create_constraints_from_nullspace
    )
    sparse_bool_cont_variables = problem_settings.sparse_bool_cont_variables

    uda_meas_list = lifted_fg.uda_meas_list
    hom_var = lifted_fg.hom_var

    lifted_variables_names_all = lifted_fg.lifted_variables_names_all

    n_lifted = len(lifted_variables_names_all)
    Q_all = get_cost_matrix(
        lifted_fg.standard_meas_list
        + lifted_fg.fg.unknown_data_association_measurements,
        lifted_fg.cont_column_variables,
        verbose=verbose,
        safe_check=False,
        sparse_bool_cont_variables=sparse_bool_cont_variables,
    )

    Z = cp.Variable((n_lifted, n_lifted), symmetric=True)

    cvxpyConstraintList = [Z >> 0]
    constraint_dict_all = None
    if not create_constraints_from_nullspace:
        logger.info(f"\nSetting up constraint dict..")
        _, constraint_dict_all = problem_constraint_creation.problem_constraints(
            lifted_fg,
            problem_settings,
        )
        for key, A_list in constraint_dict_all.items():
            for A in A_list:
                if np.linalg.norm(A.get_matrix_sparse().todense(), "fro") < 1e-10:
                    bop = 1

        logger.info(f"\nSetting up A_list from the created constraint dict..")
        A_list, RHS_list, lv_hom = (
            problem_constraint_creation.create_constraints_manually(
                constraint_dict_all,
                lifted_variables_names_all,
                problem_settings.use_sparse_matrices,
            )
        )
        logger.info(f"\nSetting up cvxpyConstraintList..")
        for lv, (A_constr, RHS) in enumerate(zip(A_list, RHS_list)):
            cvxpyConstraintList.append(
                cp.trace(A_constr.get_matrix(variables=Q_all.variable_dict_i) @ Z)
                == RHS
            )

            # cp.trace(A_constr.get_matrix(variables=(lifted_variables_names_all, lifted_variables_names_all)) @ Z) == RHS

    else:
        A_list, RHS_list = problem_constraint_creation.create_constraints_automatically(
            uda_meas_list, lifted_fg.opt_variables, lifted_variables_names_all
        )
        A_constr = initialize_poly_matrix(lifted_variables_names_all)
        A_constr[hom_var.column_names()[0], hom_var.column_names()[0]] = 1
        A_list.append(A_constr)
        RHS_list.append(1)

        for A_constr, RHS in zip(A_list, RHS_list):
            cvxpyConstraintList.append(
                cp.trace(A_constr.get_matrix_sparse() @ Z) == RHS
            )

    logger.info(f"\nDone. Returning ProblemQCQP..")

    return ProblemQCQP(
        A_list=A_list,
        RHS_list=RHS_list,
        Q=Q_all,
        constraint_dict_all=constraint_dict_all,
    )


def run_qcqp_optimization(
    A_list: List[PolyMatrix], RHS_list: List[float], Q: PolyMatrix, verbose=False
):
    col_variable_names = list(Q.variable_dict_i.keys())
    n_lifted = len(col_variable_names)
    Z = cp.Variable((n_lifted, n_lifted), symmetric=True)

    cvxpyConstraintList = [Z >> 0]

    for lv, (A_constr, RHS) in enumerate(zip(A_list, RHS_list)):
        cvxpyConstraintList.append(
            cp.trace(A_constr.get_matrix(variables=Q.variable_dict_i) @ Z) == RHS
        )

    logger.info("Setting up CVXPY problem..")
    prob = cp.Problem(
        cp.Minimize(cp.trace(Q.get_matrix_sparse() @ Z)),
        constraints=cvxpyConstraintList,
    )
    logger.info("Solving CVXPY problem..")

    prob.solve(solver=cp.MOSEK, verbose=True)

    return Z, prob, cvxpyConstraintList
