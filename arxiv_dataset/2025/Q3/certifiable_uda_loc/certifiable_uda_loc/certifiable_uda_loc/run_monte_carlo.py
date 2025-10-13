from navlie.batch.problem import Problem
import navlie
from typing import Dict
from matplotlib import pyplot as plt
import navlie.utils.plot as nv_plot

import navlie.lib.states as nv_states
from typing import List
import py_factor_graph.variables as pyfg_variables
import pymlg
import py_factor_graph.measurements as pyfg_measurements

from certifiable_uda_loc.utils.tqdm_joblib import tqdm_joblib
import certifiable_uda_loc.local_solvers as local_solvers
import navlie.batch.gaussian_mixtures as gaussian_mixtures

from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from py_factor_graph.factor_graph import FactorGraphData
from certifiable_uda_loc.trials import (
    MonteCarloResultSdp,
    MonteCarloSummary,
)
from certifiable_uda_loc.utias_framework import UnknownAssociationLocalizationProblem

import certifiable_uda_loc.generate_se2_cases as gen_se2
import time
import certifiable_uda_loc.lifted_factor_graph as lifted_factor_graph
import certifiable_uda_loc.subtask_factors as subtask_factors
import dill as pickle
from certifiable_uda_loc.settings import (
    ProblemSolutionSettings,
    Se2CaseNoiseParameters,
)
from certifiable_uda_loc.parsing import get_parser
import certifiable_uda_loc.trials as trials
import certifiable_uda_loc.test_cases as test_cases
import os
from pathlib import Path
from loguru import logger
from typing import List
import certifiable_uda_loc.utils.conversion_utils as conversion_utils


# def fro_error_se2(est_pose: nv_states.SE2State, true_pose: nv_states.SE2State):
#     rot_error = np.linalg.norm(est_pose.attitude - true_pose.attitude, "fro") ** 2
#     pos_error = np.linalg.norm(est_pose.position - est_pose.position, 2) ** 2
#     return rot_error + pos_error


class MonteCarloResultComparison:
    def __init__(
        self,
        mc_result_sdp: MonteCarloResultSdp,
        gt_states: List[nv_states.SE2State],
        dr_states: List[nv_states.SE2State],
        max_mix_gt_init_states: List[nv_states.SE2State],
        max_mix_dr_init_states: List[nv_states.SE2State],
        problem_gt_init: navlie.batch.problem.Problem,
        problem_dr_init: navlie.batch.problem.Problem,
        problem_mm_sdp_init: navlie.batch.problem.Problem,
        result_max_mix_gt_init,
        result_max_mix_dr_init,
        result_mm_sdp_init,
        problem_pure_dr: navlie.batch.problem.Problem,
        result_pure_dr,
        problem_gt_association: navlie.batch.problem.Problem,
        result_gt_association,
    ):
        self.mc_result_sdp = mc_result_sdp
        self.gt_states: List[nv_states.SE2State] = gt_states
        self.dr_states: List[nv_states.SE2State] = dr_states
        self.max_mix_gt_init_states: List[nv_states.SE2State] = max_mix_gt_init_states
        self.max_mix_dr_init_states: List[nv_states.SE2State] = max_mix_dr_init_states
        self.problem_gt_init: navlie.batch.problem.Problem = problem_gt_init
        self.problem_dr_init: navlie.batch.problem.Problem = problem_dr_init
        self.problem_mm_sdp_init: navlie.batch.problem.Problem = problem_mm_sdp_init
        self.result_max_mix_gt_init: Dict = result_max_mix_gt_init
        self.result_max_mix_dr_init: Dict = result_max_mix_dr_init
        self.result_mm_sdp_init: Dict = result_mm_sdp_init
        self.problem_pure_dr = problem_pure_dr
        self.result_pure_dr = result_pure_dr

        self.problem_gt_association: navlie.batch.problem.Problem = (
            problem_gt_association
        )
        self.result_gt_association = result_gt_association

        self.solutions_dict: Dict[str, List[nv_states.SE2State]] = {}
        self.data_associations_dict: Dict[str, np.ndarray] = {}

    def get_costs_dict(self):
        cost_dict = {}
        cost_mm_gt = self.result_max_mix_gt_init["summary"].cost[-1]
        cost_mm_dr = self.result_max_mix_dr_init["summary"].cost[-1]
        cost_dr_opt = self.result_pure_dr["summary"].cost[-1]
        mc_summary = self.mc_result_sdp.postprocess()
        cost_sdp = mc_summary.est_cost / 2
        cost_dict["SDP"] = cost_sdp
        cost_dict["Max-Mix GT Init"] = cost_mm_gt
        cost_dict["Max-Mix DR Init"] = cost_mm_dr
        cost_dict["DR Cost"] = self.result_max_mix_dr_init["summary"].cost[0]
        cost_dict["GT Cost"] = self.result_max_mix_gt_init["summary"].cost[0]
        cost_dict["DR Optimization"] = cost_dr_opt
        return cost_dict

    def set_solutions_dict(self):
        pose_keys = [var.name for var in self.mc_result_sdp.fg.pose_variables]

        solutions_dict = {}
        solutions_dict["SDP"] = conversion_utils.fg_poses_to_se2(
            self.mc_result_sdp.fg.pose_variables, "estimated"
        )
        solutions_dict["Max-Mix GT Init"] = [
            self.problem_gt_init.variables[key] for key in pose_keys
        ]
        solutions_dict["Max-Mix DR Init"] = [
            self.problem_dr_init.variables[key] for key in pose_keys
        ]
        solutions_dict["DR Optimization"] = [
            self.problem_pure_dr.variables[key] for key in pose_keys
        ]
        solutions_dict["GT Associations"] = [
            self.problem_gt_association.variables[key] for key in pose_keys
        ]
        # solutions_dict["Max-Mix SDP Init"] = [
        #     self.problem_mm_sdp_init.variables[key] for key in pose_keys
        # ]

        solutions_dict["GT"] = self.gt_states
        solutions_dict["DR"] = self.dr_states

        self.solutions_dict = solutions_dict

    def compute_errors(self):
        # Wouldnt this be the same as DeltaC??
        true_soln = self.solutions_dict["Max-Mix GT Init"]
        error_dict_rot = {}
        error_dict_pos = {}
        for key, states in self.solutions_dict.items():
            error_dict_rot[key] = 0.0
            error_dict_pos[key] = 0.0
            states: List[nv_states.SE2State] = states
            for lv1 in range(len(states)):
                rot_error = (
                    np.linalg.norm(
                        states[lv1].attitude - true_soln[lv1].attitude, "fro"
                    )
                    ** 2
                )

                pos_error = (
                    np.linalg.norm(states[lv1].position - true_soln[lv1].position, 2)
                    ** 2
                )

                error_dict_rot[key] += rot_error
                error_dict_pos[key] += pos_error
            error_dict_rot[key] /= len(states)
            error_dict_rot[key] = np.sqrt(error_dict_rot[key])

            error_dict_pos[key] /= len(states)
            error_dict_pos[key] = np.sqrt(error_dict_pos[key])
        return error_dict_rot, error_dict_pos

    def set_data_associations_dict(
        self,
    ):
        data_associations_dict = {}
        mm_resids = [
            resid
            for resid in self.problem_gt_init.residual_list
            if isinstance(resid, gaussian_mixtures.MaxMixtureResidual)
        ]
        solutions = self.solutions_dict
        for lv1, (label, se2_states) in enumerate(solutions.items()):
            data_associations_dict[label] = (
                local_solvers.extract_data_associations_max_mixture(
                    mm_resids, se2_states
                )
            )
        self.data_associations_dict = data_associations_dict

    def summarize(self) -> str:
        summary = ""
        summary += f"----Errors Rotation----\n"
        error_dict_rot, error_dict_pos = self.compute_errors()
        for key, e in error_dict_rot.items():
            summary += f"{key}: {e:.2e}\n"
        summary += f"----Errors Position----\n"
        for key, e in error_dict_pos.items():
            summary += f"{key}: {e:.2e}\n"

        da_error_dict = self.compute_data_association_errors()
        summary += f"----Data Associations----\n"
        for key, e in da_error_dict.items():
            summary += f"{key}: {e}\n"

        cost_dict = self.get_costs_dict()
        summary += f"----Costs----\n"
        for key, e in cost_dict.items():
            summary += f"{key}: {e}\n"
        summary += f"----Number of measurements-----\n"
        summary += f"{len(self.mc_result_sdp.fg.unknown_data_association_measurements)}"
        return summary

    def compute_data_association_errors(self):
        da_error_dict = {}
        true_da = self.data_associations_dict["GT"]
        # Each matrix has shape n_meas by n_landmarks.
        for label, da in self.data_associations_dict.items():
            count_wrong_da = 0

            diff = da - true_da
            for lv1 in range(diff.shape[0]):
                if not np.allclose(diff[lv1, :], np.zeros(diff[lv1, :].shape)):
                    count_wrong_da += 1
            da_error_dict[label] = count_wrong_da
        return da_error_dict

    def plot_solutions(
        self,
        ax: plt.Axes,
        labels_to_exclude: List[str] = [],
        plot_measurements: bool = False,
    ):
        from certifiable_uda_loc.utils.plotting import get_plot_colormap
        import seaborn as sns

        landmark_list = conversion_utils.extract_unique_landmarks(self.mc_result_sdp.fg)
        # colors = get_plot_colormap(10)
        colors = sns.color_palette("colorblind")
        linestyles = ["-", "--", "-.", ":", "-"] * 3

        for lv1, (label, se2_states) in enumerate(self.solutions_dict.items()):
            if label in labels_to_exclude:
                continue
            fig, ax = nv_plot.plot_poses(
                se2_states,
                ax,
                triad_color=colors[lv1],
                arrow_length=0.03,
                step=1,
                label=label,
                kwargs_line={
                    "linestyle": linestyles[lv1],
                    "color": colors[lv1],
                },
            )
            if plot_measurements:
                for robot_state in se2_states:
                    for (
                        uda_meas
                    ) in self.mc_result_sdp.fg.unknown_data_association_measurements:
                        if np.abs(uda_meas.timestamp - robot_state.stamp) < 1e-2:
                            for bool_var, meas in zip(
                                uda_meas.boolean_variables, uda_meas.measurement_list
                            ):
                                if not bool_var.estimated_value:
                                    continue
                                meas: (
                                    pyfg_measurements.PoseToKnownLandmarkMeasurement2D
                                ) = meas
                                C_ab = robot_state.attitude
                                r_a_ba = robot_state.position
                                r_a_la = meas.r_a_la
                                r_b_lb = meas.r_b_lb
                                r_a_lb = C_ab @ r_b_lb.reshape(-1, 1)
                                point1 = r_a_ba
                                point2 = r_a_ba.reshape(-1, 1) + r_a_lb.reshape(-1, 1)
                                point2 = point2.reshape(-1)
                                ax.plot([point1[0], point2[0]], [point1[1], point2[1]])
        for l in landmark_list:
            ax.scatter(l[0], l[1], color="blue")
        ax.legend()
        return fig, ax


def get_initializations(fg: FactorGraphData):
    pose_keys = [var.name for var in fg.pose_variables]
    """ --------------------- Ground Truth -----------------------------"""

    se2_states: List[nv_states.SE2State] = []
    for lv_var, var in enumerate(fg.pose_variables):
        var: pyfg_variables.PoseVariable2D = var
        pose = pymlg.SE2.from_components(var.true_rotation_matrix, var.true_position)
        se2_states.append(nv_states.SE2State(pose, stamp=lv_var, state_id=var.name))
    se2_states_gt = se2_states

    """ --------------------- Dead reckoning -----------------------------"""
    se2_states: List[nv_states.SE2State] = []
    var: pyfg_variables.PoseVariable2D = fg.pose_variables[0]
    pose = pymlg.SE2.from_components(var.true_rotation_matrix, var.true_position)
    se2_states.append(
        nv_states.SE2State(pose, stamp=0, state_id=pose_keys[0])
    )  # first one
    for lv1, odom_meas in enumerate(fg.odom_measurements):
        odom_meas: pyfg_measurements.PoseMeasurement2D = odom_meas
        stamp = se2_states_gt[lv1 + 1].stamp
        dT = pymlg.SE2.from_components(
            odom_meas.rotation_matrix, odom_meas.translation_vector
        )
        pose_kp = se2_states[-1].pose @ dT
        se2_states.append(
            nv_states.SE2State(
                pose_kp,
                stamp=stamp,
                state_id=pose_keys[lv1 + 1],
            )
        )
    se2_states_dr = se2_states
    return {"GT": se2_states_gt, "DR": se2_states_dr}, None


# TODO: Specify constraints to remove from args
def main_monte_carlo(args, fg_fnames: List[str], continue_trials=False):

    save_dir = os.path.join(args.results_dir, args.run_id)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    problem_settings = ProblemSolutionSettings(
        sparse_bool_cont_variables=True,
        create_constraints_from_nullspace=args.create_constraints_from_nullspace,
        create_discrete_variable_constraints_from_nullspace=args.create_discrete_variable_constraints_from_nullspace,
        discrete_variable_constraints=args.discrete_variable_constraints,
        use_sparse_matrices=args.use_sparse_matrices_properly,
        sparsify_interfactor_discrete_constraints=not args.no_sparsify_interfactor_discrete_constraints,
        sparsify_A_th_premultiplication=not args.no_sparsify_A_th_premultiplication,
        sparsify_cx_premultiplication_of_A_th=not args.no_sparsify_cx_premultiplication_of_A_th,
        sparsify_cxi_cxj_premultiplication_of_A_th=not args.no_sparsify_cxi_cxj_premultiplication_of_A_th,
        sparsify_analytic_moment_constraints=not args.no_sparsify_analytic_moment_constraints,
        sparsify_analytic_moment_constraints_2=not args.no_sparsify_analytic_moment_constraints_2,
        sparsify_analytic_moment_constraints_3=not args.no_sparsify_analytic_moment_constraints_3,
        sparsify_off_diag_boolean=not args.no_sparsify_off_diag_boolean,
        # sparsify_off_diag_boolean=False,
        sparsify_bool_product_moment=not args.no_sparsify_bool_product_moment,
        locked_first_pose=args.locked_first_pose,
        solver_primal=args.solver_primal,
        solver_cost_matrix_adjust=args.solver_cost_matrix_adjust,
        prior_landmark_location=args.prior_landmark_location,
        solver=args.solver,
    )

    # problem_settings = ProblemSolutionSettings(
    #     sparse_bool_cont_variables=args.sparse_bool_cont_variables,
    #     create_constraints_from_nullspace=args.create_constraints_from_nullspace,
    #     create_discrete_variable_constraints_from_nullspace=args.create_discrete_variable_constraints_from_nullspace,
    #     discrete_variable_constraints=args.discrete_variable_constraints,
    #     use_sparse_matrices=args.use_sparse_matrices_properly,
    #     sparsify_interfactor_discrete_constraints=not args.no_sparsify_interfactor_discrete_constraints,
    #     sparsify_A_th_premultiplication=not args.no_sparsify_A_th_premultiplication,
    #     sparsify_cx_premultiplication_of_A_th=not args.no_sparsify_cx_premultiplication_of_A_th,
    #     sparsify_cxi_cxj_premultiplication_of_A_th=not args.no_sparsify_cxi_cxj_premultiplication_of_A_th,
    #     sparsify_analytic_moment_constraints=not args.no_sparsify_analytic_moment_constraints,
    #     sparsify_analytic_moment_constraints_2=not args.no_sparsify_analytic_moment_constraints_2,
    #     sparsify_analytic_moment_constraints_3=not args.no_sparsify_analytic_moment_constraints_3,
    #     sparsify_off_diag_boolean=not args.no_sparsify_off_diag_boolean,
    #     sparsify_bool_product_moment=not args.no_sparsify_bool_product_moment,
    # )
    logfile_top = os.path.join(save_dir, "execution.log")
    logger.add(logfile_top, enqueue=True, mode="w")

    args_str = f"\n"
    for arg in vars(args):
        args_str += f"{arg}: {str(getattr(args, arg))}\n"
    logger.info(args_str)
    logger.remove()

    def trial(lv_seed, logfile_lv_seed):
        mc_result_fpath = os.path.join(
            save_dir,
            f"mc_result_{lv_seed}.pkl",
        )
        if continue_trials:
            # Check if path exists. If it exists, skip this trial.
            if os.path.exists(mc_result_fpath):
                return
        # np.random.seed(lv_seed)
        logger.info(os.path.abspath(logfile_lv_seed))
        logger.add(logfile_lv_seed, enqueue=True, mode="w")
        logger.info(f"Logging initialized for run {lv_seed}")
        with open(fg_fnames[lv_seed], "rb") as f:
            fg = pickle.load(f)

        logger.info(f"\nFactor Graph Details\n {fg.summarize()}")
        # Need to set fg gt values to something.
        logger.info("Entering optimize_se_factor_graph..")
        mc_result = optimize_se2_factor_graph_sdp(
            fg,
            problem_settings,
            verbose=True,
        )
        # TODO
        # Set up Max-Mixture and GN fixed association solve in
        # Monte Carlo runs
        # summary = mc_result.postprocess()
        mc_result.postprocess()
        """------------SECTION FOR MAX-MIXTURES---------------"""
        initializations_dict, info = get_initializations(fg)
        landmark_list = conversion_utils.extract_unique_landmarks(fg)

        logger.info("Solving Max-Mix, GT Init")
        print("Solving Max-Mix, GT Init")
        problem_gt_init, result_max_mix_gt_init = local_solvers.solve_max_mixture(
            fg, initializations_dict["GT"]
        )

        logger.info("Solving GT associations")
        mm_resids = [
            res
            for res in problem_gt_init.residual_list
            if isinstance(res, gaussian_mixtures.MaxMixtureResidual)
        ]

        data_associations_gt = local_solvers.extract_data_associations_max_mixture(
            mm_resids, initializations_dict["GT"]
        )
        problem_gt_association, result_gt_association = local_solvers.solve_max_mixture(
            fg,
            initializations_dict["GT"],
            data_association_matrix_if_fixed=data_associations_gt,
        )

        logger.info("Solving DR optimization")
        problem_pure_dr, result_pure_dr = local_solvers.solve_max_mixture(
            fg, initializations_dict["DR"], include_max_mixture_residuals=False
        )

        # sdp_solution = conversion_utils.fg_poses_to_se2(
        # mc_result.fg.pose_variables, "estimated"
        # )
        # NEXT TODO Finish the problem max mix sdp init.
        # problem_mm_sdp_init, result_mm_sdp_init = local_solvers.solve_max_mixture(
        # fg, sdp_solution, landmark_list
        # )
        problem_mm_sdp_init = None
        result_mm_sdp_init = None
        # mm_resids = [
        #     resid
        #     for resid in problem_gt_init.residual_list
        #     if isinstance(resid, gaussian_mixtures.MaxMixtureResidual)
        # ]
        pose_keys = [var.name for var in fg.pose_variables]

        [problem_gt_init.variables[key] for key in pose_keys]

        print("Solving Max-Mix, DR Init")
        logger.info("Solving Max-Mix, DR Init")
        problem_dr_init, result_max_mix_dr_init = local_solvers.solve_max_mixture(
            fg, initializations_dict["DR"]
        )

        mc_result_comparison = MonteCarloResultComparison(
            mc_result_sdp=mc_result,
            gt_states=initializations_dict["GT"],
            dr_states=initializations_dict["DR"],
            max_mix_gt_init_states=[
                problem_gt_init.variables[key] for key in pose_keys
            ],
            max_mix_dr_init_states=[
                problem_dr_init.variables[key] for key in pose_keys
            ],
            problem_gt_init=problem_gt_init,
            problem_dr_init=problem_dr_init,
            problem_mm_sdp_init=problem_mm_sdp_init,
            result_max_mix_dr_init=result_max_mix_dr_init,
            result_max_mix_gt_init=result_max_mix_gt_init,
            result_mm_sdp_init=result_mm_sdp_init,
            problem_pure_dr=problem_pure_dr,
            result_pure_dr=result_pure_dr,
            problem_gt_association=problem_gt_association,
            result_gt_association=result_gt_association,
        )

        with open(
            os.path.join(
                save_dir,
                f"mc_result_comparison_{lv_seed}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(mc_result_comparison, f)
        logger.remove()

    with open(
        os.path.join(
            save_dir,
            f"run_args.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(args, f)

    with open(
        os.path.join(
            save_dir,
            f"problem_settings.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(problem_settings, f)

    def log_path_from_args(args, lv):
        return os.path.join(args.results_dir, args.run_id, f"log_{lv}.txt")

    if args.n_jobs > 1 or args.n_jobs == -1:
        with tqdm_joblib(
            tqdm(desc=f"Monte Carlo", total=args.n_trials_per_run)
        ) as progress_bar:
            Parallel(n_jobs=int(args.n_jobs))(
                delayed(trial)(lv2, log_path_from_args(args, lv2))
                for lv2 in range(args.n_trials_per_run)
            )
    else:
        for lv2 in range(args.n_trials_per_run):
            trial(lv2, log_path_from_args(args, lv2))


def hardcoded_se2_case():

    noise_parameters = Se2CaseNoiseParameters(
        prior_rot_kappa_langevin_inverse=0.01,
        prior_pos_cov=0.01,
        prior_landmark_cov=0.01,
        prior_landmark_noise_corrupt_cov=0.01,
        rel_rot_kappa_langevin_inverse=0.01,
        rel_pos_cov=0.01,
        rel_landmark_meas_cov=0.01,
    )
    fg = gen_se2.create_se2_factor_graph(
        n_poses=3,
        n_landmarks=2,
        landmark_spread=5,
        meas_per_landmark=1,
        fraction_removal=0,
        uda_fraction=1,
        noise_parameters=noise_parameters,
        add_noise=True,
    )
    return fg, noise_parameters


def hardcoded_problem_settings():
    problem_settings = ProblemSolutionSettings(
        sparse_bool_cont_variables=False,
        use_moment_constraints=False,
        create_constraints_from_nullspace=False,
        create_discrete_variable_constraints_from_nullspace=False,
        discrete_variable_constraints=[
            # "bool",
            "prod_ci_cj",
            "c_c_squared",
            "sum_one",
        ],
    )
    return problem_settings


def hardcoded_toy_example():
    fg = test_cases.generate_random_toy_case(
        nx=3,
        n_components_per_factor=[3, 3, 2],
        scale_center=5,
        scale_offset=10,
        scale_Q=3,
    )
    return fg


def optimize_se2_factor_graph_sdp(
    fg: FactorGraphData,
    problem_settings: ProblemSolutionSettings,
    verbose=True,
    utias_framework=True,
):

    fg_lifted = lifted_factor_graph.LiftedFactorGraph(
        fg,
        problem_settings.sparse_bool_cont_variables,
        problem_settings.locked_first_pose,
    )
    if verbose:
        print(fg_lifted)
    t0 = time.time()
    if verbose:
        print("Setting up problem.. ")
    prob_qcqp = trials.setup_problem_qcqp(fg_lifted, problem_settings=problem_settings)
    t1 = time.time()
    logger.info(f"Problem setup finished, took {t1-t0} seconds.")

    # if utias_framework:
    problem = UnknownAssociationLocalizationProblem(fg_lifted, problem_settings)
    # Should set these at top level too.
    TOL = 1e-11
    options_cvxpy = {}
    options_cvxpy["mosek_params"] = {
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": TOL,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": TOL,
        "MSK_DPAR_INTPNT_CO_TOL_MU_RED": TOL,
        # "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": TOL,
        # "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-6,
        "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1e7,
    }
    Z, info = problem.solve_sdp(
        solver=problem_settings.solver,
        primal=problem_settings.solver_primal,
        adjust=problem_settings.solver_cost_matrix_adjust,
        options_cvxpy=options_cvxpy,
    )
    # else:
    #     Z, prob_cvxpy, cvxpyConstraintList = trials.run_qcqp_optimization(
    #         A_list=prob_qcqp.A_list,
    #         RHS_list=prob_qcqp.RHS_list,
    #         Q=prob_qcqp.Q,
    #         verbose=verbose,
    #     )

    mc_result = MonteCarloResultSdp(
        fg,
        prob_qcqp,
        Z,
        info["cost"],
        info["dual_cost"],
        info["H"],
        info["yvals"],
        info["msg"],
    )

    return mc_result


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main_monte_carlo(args)
