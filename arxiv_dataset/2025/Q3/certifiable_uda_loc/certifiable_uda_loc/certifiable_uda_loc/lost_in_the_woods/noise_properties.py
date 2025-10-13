# Need to have a test of the conversion between range/bearing and the dr...
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
import certifiable_uda_loc.path_configuration as path_config


def translation_from_range_bearing_noise(
    liw_ds: dataset.LostInTheWoodsDataset, time_bounds: List[float], n_poses: int
) -> np.ndarray:
    run_id = "noise_paramters"
    parameter_dict = {
        "solver": "mosek",
        "rel_landmark_meas_cov": 1,
        # "rel_rot_kappa_langevin_inverse": rot_noise,
        # "rel_pos_cov": rel_pos_noise,
        "rel_noise_scale": 1,
        "n_poses": None,
        "n_landmarks": None,
        "overall_time_bounds": [None, None],
        "results_dir": path_config.top_level_result_dir,
        "n_jobs": 1,
        "run_id": run_id,
    }
    arg_dict = default_arg_dict()
    arg_dict.update(parameter_dict)
    args = ParserArguments(**arg_dict)
    noise_parameters = gen_se2.Se2CaseNoiseParameters.from_args(args)

    tmin = time_bounds[0]
    tmax = time_bounds[1]
    # tmin = 400
    # tmax = 450
    # Main quesiton: how are we hitting situation where no pose with given measurement is found..
    liw_ds: dataset.LostInTheWoodsDataset = dataset.LostInTheWoodsDataset.from_mat_file(
        path_config.lost_in_the_woods_dataset_path,
        time_bounds=[tmin, tmax],
    )

    fg: FactorGraphData = liw_convert_to_fg.get_factor_graph_from_liw(
        liw_ds=liw_ds,
        time_bounds=[tmin, tmax],
        n_poses=n_poses,
        noise_parameters=noise_parameters,
        landmark_indices_to_use=[lv1 for lv1 in range(17)],
        unknown_data_association=False,
    )

    fg: FactorGraphData = fg

    # Timestamp verification.
    # stamps_rel_pos = [meas.timestamp for meas in fg.pose_known_landmark_measurements]
    # stamps_states = [state.timestamp for state in fg.pose_variables]

    # plt.figure()
    # ax: plt.Axes = plt.gca()
    # ax.vlines([stamps_rel_pos], label="Msmt stamps", ymin=0, ymax=1, colors="red")
    # ax.vlines([stamps_states], label="State stamps", ymin=-1, ymax=0, colors="blue")
    # plt.legend()
    # plt.savefig("./stamps.pdf", bbox_inches="tight")
    # for meas in fg.pose_known_landmark_measurements:
    #     pass

    def get_robot_state_from_stamp(stamp: float):
        tol = 1e-1
        fg_stamps = np.array([pose.timestamp for pose in fg.pose_variables])
        dt_arr = np.abs(fg_stamps - stamp)
        min_idx = np.argmin(dt_arr)
        if dt_arr[min_idx] > tol:
            raise (BaseException("dt is too big.."))
        return fg.pose_variables[min_idx]

    dr_list = []
    for meas in fg.pose_known_landmark_measurements:
        meas: pfg_measurements.PoseToKnownLandmarkMeasurement2D = meas
        pose: PoseVariable2D = get_robot_state_from_stamp(meas.timestamp)

        y_meas = meas.r_b_lb
        C_ab = pose.true_rotation_matrix
        r_a_ba = pose.true_position
        r_a_la = meas.r_a_la
        y_check = C_ab.T @ (r_a_la - r_a_ba)
        dr = y_meas - y_check
        dr_list.append(dr)
    dr_list = [dr.reshape(-1, 1) for dr in dr_list]
    dr_mat = np.hstack(dr_list)
    cov_dr = np.cov(dr_mat)  # dr - delta translation, d r, dr. not deadreckoning.
    return cov_dr


def relative_pose_noise(
    liw_ds: dataset.LostInTheWoodsDataset,
    time_bounds: List[float],
    n_poses: int,
    von_mises_fig_path=None,
):
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
    arg_dict = default_arg_dict()
    arg_dict.update(parameter_dict)
    args = ParserArguments(**arg_dict)
    noise_parameters = gen_se2.Se2CaseNoiseParameters.from_args(args)

    tmin = time_bounds[0]
    tmax = time_bounds[1]
    liw_ds: dataset.LostInTheWoodsDataset = dataset.LostInTheWoodsDataset.from_mat_file(
        path_config.lost_in_the_woods_dataset_path,
        time_bounds=[tmin, tmax],
    )

    fg: FactorGraphData = liw_convert_to_fg.get_factor_graph_from_liw(
        liw_ds=liw_ds,
        time_bounds=[tmin, tmax],
        n_poses=n_poses,
        noise_parameters=noise_parameters,
        landmark_indices_to_use=[lv1 for lv1 in range(17)],
        unknown_data_association=False,
    )

    fg: FactorGraphData = fg

    # Timestamp verification.
    stamps_odom = [meas.timestamp for meas in fg.odom_measurements]
    stamps_states = [state.timestamp for state in fg.pose_variables]

    bop = 1

    def get_robot_state_from_stamp(stamp: float):
        tol = 2e-1
        fg_stamps = np.array([pose.timestamp for pose in fg.pose_variables])
        dt_arr = np.abs(fg_stamps - stamp)
        min_idx = np.argmin(dt_arr)
        if dt_arr[min_idx] > tol:
            raise (BaseException("dt is too big.."))
        return fg.pose_variables[min_idx]

    dt = fg.pose_variables[1].timestamp - fg.pose_variables[0].timestamp

    dC_err_list = []  # rotation
    dr_err_list = []  # translation
    for meas in fg.odom_measurements:
        meas: pfg_measurements.PoseMeasurement2D = meas
        pose_k: PoseVariable2D = get_robot_state_from_stamp(meas.timestamp)
        pose_kp: PoseVariable2D = get_robot_state_from_stamp(meas.timestamp + dt)
        T_k = pose_k.true_transformation_matrix
        T_kp = pose_kp.true_transformation_matrix

        dT_true = SE2.inverse(T_k) @ T_kp
        dT_est = SE2.from_components(meas.rotation_matrix, meas.translation_vector)

        dC_true, dr_true = SE2.to_components(dT_true)
        dC_est, dr_est = SE2.to_components(dT_est)

        dC_err = np.linalg.inv(dC_true) @ dC_est
        dr_err = dr_true - dr_est
        dC_err_list.append(dC_err)
        dr_err_list.append(dr_err)

    dr_err_list = [dr.reshape(-1, 1) for dr in dr_err_list]
    dr_err_mat = np.hstack(dr_err_list)
    cov_dr = np.cov(dr_err_mat)  # dr - delta translation, d r, dr. not deadreckoning.
    # print("Covariance on position part of odom measurements given by")
    # print(cov_dr)
    # print("Use only diagonal and average, ")
    # print(np.mean(np.diag(cov_dr)))

    # Now for rotation part.
    # dC = dC_err_list[0]
    # print(SO2.vee(SO2.Log(dC)))
    dtheta_arr = [SO2.Log(dC) for dC in dC_err_list]
    kappa, loc, scale = vonmises.fit(np.array(dtheta_arr), fscale=1)

    x = np.linspace(min(dtheta_arr), max(dtheta_arr), 100)
    von_mises_arr = []
    for lv1 in range(x.shape[0]):
        von_mises_arr.append(vonmises.pdf(x[lv1], loc=loc, kappa=kappa))

    if von_mises_fig_path is not None:
        plt.figure()
        ax: plt.Axes = plt.gca()
        ax.hist(dtheta_arr, bins=5, label="Data hist", density=True)
        ax.plot(x, von_mises_arr, label="Fit")
        ax.legend()
        plt.savefig(von_mises_fig_path, bbox_inches="tight")

    return kappa, loc, cov_dr
