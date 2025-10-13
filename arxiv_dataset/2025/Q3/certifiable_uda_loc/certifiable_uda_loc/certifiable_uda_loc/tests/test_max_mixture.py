import os
from certifiable_uda_loc.run_monte_carlo import main_monte_carlo
from typing import List
import dill as pickle
from certifiable_uda_loc.run_monte_carlo import MonteCarloResultComparison
from certifiable_uda_loc.parsing import ParserArguments
from py_factor_graph.factor_graph import FactorGraphData
import pymlg
import py_factor_graph.variables as pyfg_variables
import navlie.types as nv_types
import navlie.lib.states as nv_states
import py_factor_graph.measurements as pyfg_measurements
import navlie
from navlie.batch import gaussian_mixtures
import navlie.batch.residuals as nv_residuals
import navlie.lib.models as nv_models
import py_factor_graph.data_associations as pyfg_data_associations
import numpy as np
from certifiable_uda_loc.gauss_newton_baseline import (
    PriorResidualFro,
    RelativePoseResidualFro,
)
import certifiable_uda_loc.path_configuration as path_config
from certifiable_uda_loc.analysis_machinery import Analysis


def test_max_mixture_comparison():

    arg_dict = default_arg_dict()
    args = ParserArguments(**arg_dict)
    args.rel_rot_kappa_langevin_inverse = 2
    args.rel_pos_cov = 2
    script_path = os.path.join(path_config.test_dir)
    fg_fname = os.path.join(script_path, "temp_fg.pkl")
    analysis = Analysis(
        analysis_id="test",
        parameter_dict_list=None,
        param_settings_list_breakdown=None,
        root_dir=path_config.project_root_dir,
        default_arg_dict=None,
    )
    analysis.top_level_factor_graph_dir = script_path
    analysis.factor_graph_setup_from_args(args, fnames=[fg_fname])

    with open(fg_fname, "rb") as f:
        fg: FactorGraphData = pickle.load(f)

    """ --------------------- Ground Truth -----------------------------"""

    se2_states: List[nv_states.SE2State] = []
    for lv_var, var in enumerate(fg.pose_variables):
        var: pyfg_variables.PoseVariable2D = var
        pose = pymlg.SE2.from_components(var.true_rotation_matrix, var.true_position)
        se2_states.append(nv_states.SE2State(pose, stamp=lv_var, state_id=var.name))
    se2_states_gt = se2_states

    # Next: Dead-reckoning.
    """ --------------------- Dead reckoning -----------------------------"""
    se2_states: List[nv_states.SE2State] = []
    var: pyfg_variables.PoseVariable2D = fg.pose_variables[0]
    pose = pymlg.SE2.from_components(var.true_rotation_matrix, var.true_position)
    se2_states.append(nv_states.SE2State(pose))  # first one
    for odom_meas in fg.odom_measurements:
        odom_meas: pyfg_measurements.PoseMeasurement2D = odom_meas
        dT = pymlg.SE2.from_components(
            odom_meas.rotation_matrix, odom_meas.translation_vector
        )
        pose_kp = se2_states[-1].pose @ dT
        se2_states.append(nv_states.SE2State(pose_kp))

    se2_states_dr = se2_states

    """ --------------------- SDP SOLUTION -----------------------------"""
    results_dir = args.results_dir
    save_dir = os.path.join(results_dir, args.run_id)
    main_monte_carlo(args, [fg_fname])
    with open(
        os.path.join(results_dir, "test", f"mc_result_comparison_{0}.pkl"), "rb"
    ) as f:
        mc_result_comparison: MonteCarloResultComparison = pickle.load(f)
        mc_result_sdp = mc_result_comparison.mc_result_sdp
    mc_summary = mc_result_sdp.postprocess()
    print(mc_summary)
    fg_sdp_solved = mc_result_sdp.fg
    se2_states: List[nv_states.SE2State] = []
    for var in fg_sdp_solved.pose_variables:
        var: pyfg_variables.PoseVariable2D = var
        pose = pymlg.SE2.from_components(var.true_rotation_matrix, var.true_position)
        se2_states.append(nv_states.SE2State(pose))

    """ --------------------- Max Mixture -----------------------------"""
    se2_states: List[nv_states.SE2State] = []
    for lv_var, var in enumerate(fg.pose_variables):
        var: pyfg_variables.PoseVariable2D = var
        pose = pymlg.SE2.from_components(var.true_rotation_matrix, var.true_position)
        se2_states.append(nv_states.SE2State(pose, stamp=lv_var, state_id=var.name))
    pose_keys = [var.name for var in fg.pose_variables]
    prior_residuals = []
    for prior_meas in fg.prior_pose_measurements:
        prior_meas: pyfg_data_associations.PosePrior2D = prior_meas
        T_check = pymlg.SE2.from_components(prior_meas.rotation, prior_meas.position)
        prior_state = nv_states.SE2State(value=T_check, stamp=0, state_id=pose_keys[0])
        prior_residuals.append(
            PriorResidualFro(
                pose_keys[0],
                prior_state,
                weight_rot=prior_meas.weight_rot,
                weight_pos=prior_meas.weight_pos,
            )
        )
    process_residuals = []
    for odom_meas in fg.odom_measurements:
        odom_meas: pyfg_measurements.PoseMeasurement2D = odom_meas
        # TODO: Figure out noises.
        dT = pymlg.SE2.from_components(
            odom_meas.rotation_matrix, odom_meas.translation_vector
        )
        resid = RelativePoseResidualFro(
            keys=[odom_meas.base_pose, odom_meas.to_pose],
            dT=dT,
            weight_rot=odom_meas.weight_rot,
            weight_pos=odom_meas.weight_pos,
        )
        process_residuals.append(resid)
    # TODO: For SLAM will have PoseToLandmark measurement (not known)
    # will need to add
    mixture_residuals = []
    for uda_meas in fg.unknown_data_association_measurements:
        uda_meas: pyfg_data_associations.UnknownDataAssociationMeasurement = uda_meas
        component_residuals = []
        for pyfg_meas in uda_meas.measurement_list:
            pyfg_meas: pyfg_measurements.PoseToKnownLandmarkMeasurement2D = pyfg_meas
            meas_model = nv_models.PointRelativePosition(
                landmark_position=pyfg_meas.r_a_la,
                R=np.eye(2) * 1 / pyfg_meas.weight,
            )
            nv_meas = nv_types.Measurement(pyfg_meas.r_b_lb, model=meas_model)
            resid = nv_residuals.MeasurementResidual([pyfg_meas.pose_name], nv_meas)
            component_residuals.append(resid)
        num_comp = len(component_residuals)

        mix_resid = gaussian_mixtures.MaxMixtureResidual(
            component_residuals, weights=[1 / num_comp] * num_comp
        )
        mixture_residuals.append(mix_resid)
    problem = navlie.batch.problem.Problem()
    for var in se2_states:  # add names to gt se2states
        var: nv_states.SE2State = var
        problem.add_variable(var.state_id, var)
    for resid in process_residuals + mixture_residuals + prior_residuals:
        problem.add_residual(resid)

    result_max_mix = problem.solve()
    se2_states_max_mixture = [problem.variables[key] for key in pose_keys]

    cost_mm = result_max_mix["summary"].cost[-1]
    print(f"Max-Mixture Cost: {cost_mm}")
    print(f"SDP Cost: {mc_summary.est_cost}")
    print(f"Half of SDP Cost: {mc_summary.est_cost/2}")

    assert np.abs(mc_summary.est_cost / 2 - cost_mm) < 1e-4


def default_arg_dict():
    arg_dict = {
        "run_id": "test",
        "verbose": False,
        "results_dir": path_config.top_level_result_dir,
        "n_trials_per_run": 1,
        "create_constraints_from_nullspace": False,
        "create_discrete_variable_constraints_from_nullspace": False,
        "discrete_variable_constraints": ["bool", "prod_ci_cj", "sum_one"],
        "n_jobs": 1,
        "use_sparse_matrices_properly": True,
        "sparse_bool_cont_variables": True,
        "no_sparsify_interfactor_discrete_constraints": False,
        "no_sparsify_A_th_premultiplication": False,
        "no_sparsify_cx_premultiplication_of_A_th": False,
        "no_sparsify_cxi_cxj_premultiplication_of_A_th": False,
        "no_sparsify_analytic_moment_constraints": False,
        "no_sparsify_analytic_moment_constraints_2": False,
        "no_sparsify_analytic_moment_constraints_3": False,
        "no_sparsify_off_diag_boolean": True,
        "no_sparsify_bool_product_moment": False,
        "problem_type": "se2",
        "subtask": "localization",
        "prior_rot_kappa_langevin_inverse": 0.01,
        "prior_pos_cov": 0.01,
        "prior_landmark_cov": 200,
        "prior_landmark_noise_corrupt_cov": None,
        "rel_rot_kappa_langevin_inverse_base_val": 0.01,
        "rel_pos_base_val": 0.1,
        "rel_noise_scale": 1,
        "rel_landmark_meas_cov": 0.01,
        "n_poses": 3,
        "n_landmarks": 2,
        "landmark_spread": 5,
        "meas_per_landmark": 1,
        "fraction_removal": 0,
        "uda_fraction": 1,
        "noiseless": False,
        "locked_first_pose": False,
        "solver": "mosek",
        "solver_primal": False,
        "solver_cost_matrix_adjust": True,
        "prior_landmark_location": None,
    }
    return arg_dict


if __name__ == "__main__":
    test_max_mixture_comparison()
