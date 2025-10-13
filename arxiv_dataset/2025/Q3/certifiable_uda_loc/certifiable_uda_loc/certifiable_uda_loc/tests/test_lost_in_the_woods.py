import pymlg
from scipy.stats import vonmises
import os
from certifiable_uda_loc.lost_in_the_woods.dataset import LandmarkMeasurement
import py_factor_graph.measurements as pfg_measurements
import certifiable_uda_loc.lost_in_the_woods.dataset as dataset
from navlie.filters import ExtendedKalmanFilter
from navlie.lib.models import BodyFrameVelocity
from pymlg import SE2, SO2

from certifiable_uda_loc.lost_in_the_woods.utils import (
    egovehicle_key_at_stamp,
    index_list_by_numpy_index,
    landmark_key_at_idx,
)
import navlie.lib.states as nv_states
import pymlg
from py_factor_graph.variables import PoseVariable2D, LandmarkVariable2D
from certifiable_uda_loc.generate_se2_cases import (
    pose_name_from_timestep,
    timestep_from_pose_name,
)
import matplotlib.pyplot as plt

import certifiable_uda_loc.lost_in_the_woods.convert_to_fg as liw_convert_to_fg
from certifiable_uda_loc.run_monte_carlo import main_monte_carlo
from certifiable_uda_loc.parsing import ParserArguments
from certifiable_uda_loc.settings import (
    ProblemSolutionSettings,
    Se2CaseNoiseParameters,
)
from certifiable_uda_loc.run_monte_carlo import optimize_se2_factor_graph_sdp
import certifiable_uda_loc.generate_se2_cases as gen_se2

from typing import List
from py_factor_graph.factor_graph import FactorGraphData
import numpy as np
from certifiable_uda_loc.analysis_machinery import default_arg_dict
import dill as pickle
from certifiable_uda_loc.trials import (
    MonteCarloResultSdp,
    MonteCarloSummary,
)
from certifiable_uda_loc.run_monte_carlo import MonteCarloResultComparison
from collections import namedtuple
from certifiable_uda_loc.lost_in_the_woods import noise_properties
import certifiable_uda_loc.path_configuration as path_config

# def test_range_bearing_to_rel_pos():
#     test_case = namedtuple("test_case", "C_ab d r_a_la r_a_ba")
#     test_cases = [
#         test_case(C_ab=np.eye(2), d=0, r_a_la=np.array([1, 0]), r_a_ba=np.array([0, 0]))
#     ]

#     for tc in test_cases:
#         tc.C_ab
#         pass


def test_noise_properties():
    run_id = "noise_paramters"
    parameter_dict = {
        "solver": "mosek",
        "rel_landmark_meas_cov": 1,
        # "rel_rot_kappa_langevin_inverse": rot_noise,
        # "rel_pos_cov": rel_pos_noise,
        "rel_noise_scale": 1,
        "n_poses": 3,
        "n_landmarks": 2,
        "overall_time_bounds": [300, 310],
        "results_dir": path_config.top_level_result_dir,
        "n_jobs": 1,
        "run_id": run_id,
    }
    # Total dataset is 20 min.
    tmin = 500
    tmax = tmin + 60 * 5
    # Let's say we use a 5 second spacing.
    pose_spacing = 40
    n_poses = int((tmax - tmin) / pose_spacing)
    liw_ds: dataset.LostInTheWoodsDataset = dataset.LostInTheWoodsDataset.from_mat_file(
        path_config.lost_in_the_woods_dataset_path,
        time_bounds=[tmin, tmax],
    )

    # For range bearing can actually use all of them... but whatever.
    cov_lndmrk_meas = noise_properties.translation_from_range_bearing_noise(
        liw_ds, [tmin, tmax], n_poses
    )
    kappa, loc, cov_dr = noise_properties.relative_pose_noise(
        liw_ds, [tmin, tmax], n_poses, von_mises_fig_path="./dtheta_hist.pdf"
    )
    print("Cov landmark meas noise")
    print(cov_lndmrk_meas)
    print("Kappa rotation noise")
    print(kappa)
    print("Loc rotation noise")
    print(loc)
    print("Covariance translation")
    print(cov_dr)


def test_optimization():
    run_id = "test"
    parameter_dict = {
        "solver": "mosek",
        "rel_landmark_meas_cov": 1,
        # "rel_rot_kappa_langevin_inverse": rot_noise,
        # "rel_pos_cov": rel_pos_noise,
        "rel_noise_scale": 1,
        "n_poses": 3,
        "n_landmarks": 2,
        "overall_time_bounds": [300, 310],
        "results_dir": path_config.top_level_result_dir,
        "n_jobs": 1,
        "run_id": run_id,
    }
    fig_dir = os.path.join(path_config.top_level_result_dir, run_id)
    arg_dict = default_arg_dict()
    arg_dict.update(parameter_dict)
    args = ParserArguments(**arg_dict)
    noise_parameters = gen_se2.Se2CaseNoiseParameters.from_args(args)
    noise_parameters = gen_se2.Se2CaseNoiseParameters(
        prior_rot_kappa_langevin_inverse=0.01,
        prior_pos_cov=0.01,
        prior_landmark_cov=None,
        prior_landmark_noise_corrupt_cov=None,
        rel_rot_kappa_langevin_inverse=1 / 9.497851540070904,
        rel_pos_cov=0.7430593929955722,
        rel_landmark_meas_cov=0.0014746213431101004,
    )

    n_poses = 4
    tmin = 800
    spacing = 40
    tmax = tmin + spacing * n_poses
    liw_ds: dataset.LostInTheWoodsDataset = dataset.LostInTheWoodsDataset.from_mat_file(
        path_config.lost_in_the_woods_dataset_path,
        time_bounds=[tmin, tmax],
    )

    fg = liw_convert_to_fg.get_factor_graph_from_liw(
        liw_ds=liw_ds,
        time_bounds=[tmin, tmax],
        n_poses=n_poses,
        noise_parameters=noise_parameters,
        landmark_indices_to_use=[1],
        # landmark_indices_to_use=[2, 3, 4, 5],
        unknown_data_association=True,
        # num_landmarks_to_use=2,
    )

    fg_path = "./liw_fg"
    with open(fg_path, "wb") as f:
        pickle.dump(fg, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(fg.summarize())

    main_monte_carlo(args, [fg_path])

    with open(
        os.path.join(args.results_dir, run_id, f"mc_result_comparison_{0}.pkl"),
        "rb",
    ) as f:
        mc_result_comparison: MonteCarloResultComparison = pickle.load(f)

    mc_result_comparison.set_solutions_dict()
    mc_result_comparison.set_data_associations_dict()

    mc_summary: MonteCarloSummary = mc_result_comparison.mc_result_sdp.postprocess()
    print(mc_summary)
    lv_res = 0

    plt.figure()
    ax = plt.gca()
    fig, ax = mc_result_comparison.plot_solutions(ax)
    fig_path = os.path.join(fig_dir, f"mc_result_comparison_{lv_res}.pdf")
    plt.savefig(fig_path)
    plt.close()
    summary = mc_result_comparison.summarize()
    summary_file = os.path.join(fig_dir, f"mc_result_comparison_{lv_res}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(summary)
    print(summary)


def problem_settings_from_args(args):
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
    return problem_settings


if __name__ == "__main__":
    test_optimization()
    test_noise_properties()
