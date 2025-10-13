import numpy as np
from py_factor_graph.factor_graph import FactorGraphData
from certifiable_uda_loc.trials import (
    MonteCarloResultSdp,
    MonteCarloSummary,
)
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
from loguru import logger
from certifiable_uda_loc.utias_framework import UnknownAssociationLocalizationProblem
import cert_tools.sdp_solvers as sdp_solvers
from certifiable_uda_loc.run_monte_carlo import (
    main_monte_carlo,
    MonteCarloResultComparison,
)
import os
from loguru import logger
from matplotlib import pyplot as plt
import certifiable_uda_loc.path_configuration as path_config


def main():
    POSTPROCESS_ONLY = False
    np.random.seed(0)

    param_string = "solver-mosek_rel_landmark_meas_cov-0.5_rel_noise_scale-0.1_noiseless-False_n_trials_per_run-10_n_poses-3_n_landmarks-2_rel_rot_kappa_langevin_inverse_base_val-0.010000000000000002_rel_pos_base_val-0.749956"
    analysis_id = "analysis_fine"
    run_num = 3
    fg_path = os.path.join(
        path_config.project_root_dir,
        "factor_graphs",
        analysis_id,
        param_string,
        f"fg_{run_num}.pkl",
    )
    args_path = os.path.join(
        path_config.project_root_dir,
        "mc_result",
        analysis_id,
        param_string,
        f"run_args.pkl",
    )

    with open(args_path, "rb") as f:
        args = pickle.load(f)
    args.n_trials_per_run = 1
    args.results_dir = path_config.top_level_result_dir
    args.run_id = "reproducing"
    args.n_jobs = 1
    if not POSTPROCESS_ONLY:
        main_monte_carlo(args, [fg_path])

    with open(
        os.path.join(args.results_dir, args.run_id, f"mc_result_comparison_0.pkl"),
        "rb",
    ) as f:
        mc_result_comparison: MonteCarloResultComparison = pickle.load(f)
        mc_result_comparison.set_solutions_dict()
        mc_result_comparison.set_data_associations_dict()
        mc_summary: MonteCarloSummary = mc_result_comparison.mc_result_sdp.postprocess()

    print(f"Duality gap: {mc_summary.duality_gap}")
    print(f"Log eigval ratio: {mc_summary.log_eigval_ratio}")
    print(f"Num poses: {args.n_poses}")
    print(f"Num landmarks: {args.n_landmarks}")

    print(f"State error: {mc_summary.state_error}")
    cost_mm_gt = mc_result_comparison.result_max_mix_gt_init["summary"].cost[-1]
    cost_mm_dr = mc_result_comparison.result_max_mix_dr_init["summary"].cost[-1]
    cost_gt_associations = mc_result_comparison.result_gt_association["summary"].cost[
        -1
    ]
    print(f"cost_mm_gt: {cost_mm_gt}")
    print(f"cost_mm_dr: {cost_mm_dr}")
    print(f"cost_gt_associations: {cost_gt_associations}")
    print(f"cost sdp: {mc_summary.est_cost/2}")

    error_dict_rot, error_dict_pos = mc_result_comparison.compute_errors()
    print("Rotation errors")
    print(error_dict_rot)
    print("Position errors")
    print(error_dict_pos)

    plt.figure()
    ax = plt.gca()
    fig, ax = mc_result_comparison.plot_solutions(
        ax,
        # labels_to_exclude=["Max-Mix DR Init"],
        # labels_to_exclude=["SDP", "Max-Mix DR Init"],
    )
    plt.savefig(f"./poses_problem_of_interest.pdf")
    summary = mc_result_comparison.summarize()
    summary_file = os.path.join(f"./mc_result_comparison_summary.txt")
    with open(summary_file, "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
