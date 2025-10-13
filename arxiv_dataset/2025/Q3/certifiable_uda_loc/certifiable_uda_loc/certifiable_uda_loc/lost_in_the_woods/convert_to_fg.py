import certifiable_uda_loc.subtask_factors as subtask_factors
from certifiable_uda_loc.lost_in_the_woods.dataset import LandmarkMeasurement
import py_factor_graph.measurements as pfg_measurements
import py_factor_graph.data_associations as pfg_data_associations
import certifiable_uda_loc.lost_in_the_woods.dataset as dataset
from navlie.filters import ExtendedKalmanFilter
from navlie.lib.models import BodyFrameVelocity

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
import certifiable_uda_loc.generate_se2_cases as gen_se2

# Need
# a) Specification of times.
# b) Specification of # poses required between those times.
# c) Managing landmark visibility.

# Dont really know where to start here.
# Just start with process residuals. If we are doing covariances empirically, just need ground truth values.

from typing import List
from py_factor_graph.factor_graph import FactorGraphData
import numpy as np


# This seems to work somehow. Have to write out diagrams and stuff properly after.
def rangefinder_to_relative_translation(
    range_meas: float, bearing: float, d_to_rangefinder: float
):
    translation_vector = np.array(
        [range_meas * np.cos(bearing), range_meas * np.sin(bearing)]
    )
    translation_vector[0] = translation_vector[0] + d_to_rangefinder
    return translation_vector


def get_factor_graph_from_liw(
    liw_ds: dataset.LostInTheWoodsDataset,
    time_bounds: List[float],
    n_poses: int,
    noise_parameters: gen_se2.Se2CaseNoiseParameters,
    landmark_indices_to_use: List[int] = None,
    num_landmarks_to_use: int = None,
    unknown_data_association=True,
    use_ground_truth_for_relative_pose: bool = False,
    use_ground_truth_for_relative_landmark_meas: bool = False,
):
    # One of landmark_indices_to_use and num_landmarks must be specified.
    # If num_landmarks is specified, will look for landmarks
    # that are visible at each timestep.

    fg: FactorGraphData = FactorGraphData(dimension=2)

    # Q1: Get index from timestamp.
    # There will be questions of rounding...
    state_times = np.linspace(time_bounds[0], time_bounds[1], n_poses)
    state_times_rounded = []
    for stamp in state_times:
        diffs = np.abs(liw_ds.t - stamp)
        min_idx = np.argmin(diffs)
        state_times_rounded.append(liw_ds.t[min_idx])
    state_indices = [liw_ds.timestamp_to_index[t] for t in state_times_rounded]

    gt_states = [liw_ds.ground_truth_states[lv] for lv in state_indices]

    """---------Pose Variables-------"""
    fg.pose_variables = []

    for pose_idx, state in enumerate(gt_states):
        pose = state.value
        C, r = pymlg.SE2.to_components(pose)
        fg.pose_variables.append(
            PoseVariable2D(
                name=pose_name_from_timestep(pose_idx),
                true_position=r.reshape(-1),
                true_theta=pymlg.SO2.Log(C),
                estimated_position=-1000
                * np.ones(
                    2,
                ),
                estimated_theta=-1000,
                timestamp=state.stamp,
            )
        )

    input_groups: List[List[nv_states.VectorInput]] = []
    for lv_state in range(len(gt_states) - 1):
        idxs = np.logical_and(
            np.array(liw_ds.t) >= state_times_rounded[lv_state],
            np.array(liw_ds.t) < state_times_rounded[lv_state + 1],
        )
        input_group = []
        for u, idx in zip(liw_ds.inputs, idxs):
            if idx:
                input_group.append(u)
        input_groups.append(input_group)

    dT_list = []
    process_model = BodyFrameVelocity(np.eye(3))  # covariance here doesnt matter

    kf = ExtendedKalmanFilter(process_model)
    for lv_state, (x_k, input_group) in enumerate(zip(gt_states, input_groups)):
        x_k_hat = nv_states.StateWithCovariance(x_k.copy(), np.eye(x_k.dof))
        # TODO: Q: How to test this part?
        for lv1, u in enumerate(input_group):
            if lv1 == len(input_group) - 1:
                dt = gt_states[lv_state + 1].stamp - u.stamp
            else:
                dt = input_group[lv1 + 1].stamp - u.stamp
            x_k_hat, details_dict = kf.predict(
                x_k_hat.copy(), u, dt, output_details=True
            )
        x_k_hat: nv_states.StateWithCovariance = x_k_hat
        dT = np.linalg.inv(x_k.value) @ x_k_hat.state.value
        dT_list.append(dT)

    fg.odom_measurements = []
    poses_gt = [state.value for state in gt_states]
    for lv_k, (pose_k, pose_kp, dT_odom) in enumerate(
        zip(poses_gt[:-1], poses_gt[1:], dT_list)
    ):
        dT_gt = pymlg.SE2.inverse(pose_k) @ pose_kp
        if noise_parameters.rel_pos_cov < 1e-14:
            noise_parameters.rel_pos_cov = 1e-8
        if noise_parameters.rel_rot_kappa_langevin_inverse < 1e-14:
            noise_parameters.rel_rot_kappa_langevin_inverse = 1e-8
        if use_ground_truth_for_relative_pose:
            dT = dT_gt
        else:
            dT = dT_odom

        dC, dr = pymlg.SE2.to_components(dT)
        # dC, dr = pymlg.SE2.to_components(dT_gt)
        assert (np.abs(np.linalg.det(dC) - 1)) < 1e-2
        fg.odom_measurements.append(
            pfg_measurements.PoseMeasurement2D(
                base_pose=fg.pose_variables[lv_k].name,
                to_pose=fg.pose_variables[lv_k + 1].name,
                translation_vector=dr,
                rotation_matrix=dC,
                weight_pos=1 / noise_parameters.rel_pos_cov,
                weight_rot=1 / noise_parameters.rel_rot_kappa_langevin_inverse,
                timestamp=fg.pose_variables[lv_k].timestamp,
            )
        )
    fg.pose_known_landmark_measurements = []

    landmark_measurements: List[List[LandmarkMeasurement]] = (
        liw_ds.landmark_measurements(r_max=1000)
    )  # Outer list timestep, inner lists landmarks

    n_landmarks = len(landmark_measurements[0])

    landmark_measurements = [landmark_measurements[lv1] for lv1 in state_indices]

    if num_landmarks_to_use is not None:
        if landmark_indices_to_use is not None:
            raise BaseException(
                "Both num_landmarks_to_use and landmark_indices_to_use is not None.."
            )
        meas_mat = np.zeros((n_poses, n_landmarks))
        for lv_stamp in range(n_poses):
            for lv_lndmrk in range(n_landmarks):
                if landmark_measurements[lv_stamp][lv_lndmrk] is not None:
                    meas_mat[lv_stamp, lv_lndmrk] = 1
        landmark_indices_to_use = []

        # Pick the most visible landmarks.
        visibility_vec = np.sum(meas_mat, axis=0)
        idx_sort = np.argsort(visibility_vec)[::-1]
        # print(visibility_vec)
        # print(visibility_vec[idx_sort])
        landmark_indices_to_use = idx_sort[:num_landmarks_to_use].tolist()
        # for lv_lndmrk in range(n_landmarks):
        #     # condition is a convoluted way of saying - lndmrk visible all timesteps
        #     if np.abs(np.sum(meas_mat[:, lv_lndmrk]) - n_poses) < 1e-5:
        #         landmark_indices_to_use.append(lv_lndmrk)

        # landmark_indices_to_use = landmark_indices_to_use[:num_landmarks_to_use]

    for lv_stamp in range(n_poses):
        landmark_measurements[lv_stamp] = [
            landmark_measurements[lv_stamp][lv_lndmrk]
            for lv_lndmrk in landmark_indices_to_use
        ]

    n_landmarks = len(landmark_measurements[0])

    for landmark_idx, landmark_gt in enumerate(liw_ds.landmarks):
        if landmark_idx in landmark_indices_to_use:
            fg.landmark_variables.append(
                LandmarkVariable2D(
                    name=f"L{landmark_idx}",
                    true_position=landmark_gt.reshape(-1),
                    estimated_position=-1000 * np.ones(2),
                ),
            )

    for lv_stamp in range(n_poses):
        for lv_landmark in range(n_landmarks):
            meas: LandmarkMeasurement = landmark_measurements[lv_stamp][lv_landmark]
            if meas is None:
                continue
            # Have to do some transformation shenanigans with the extrinsics.
            if not use_ground_truth_for_relative_landmark_meas:
                translation_vector = rangefinder_to_relative_translation(
                    meas.range, meas.bearing, liw_ds.d_to_rangefinder
                )
            else:
                x = None
                for x_lv1 in gt_states:
                    if np.abs(meas.stamp - x_lv1.stamp) < 1e-2:
                        x = x_lv1
                        break
                C_ab = x.attitude
                r_a_ba = x.position
                r_a_la = meas.landmark_position
                r_b_lb = C_ab.T @ (r_a_la - r_a_ba).reshape(-1, 1)
                translation_vector = r_b_lb

            fg.pose_landmark_measurements.append(
                pfg_measurements.PoseToLandmarkMeasurement2D(
                    pose_name=fg.pose_variables[lv_stamp].name,
                    landmark_name=meas.landmark_id,
                    r_b_lb=translation_vector,
                    weight=1 / noise_parameters.rel_landmark_meas_cov,
                    timestamp=state_times_rounded[lv_stamp],
                )
            )

    fg.prior_pose_measurements.append(
        pfg_data_associations.PosePrior2D(
            fg.pose_variables[0].name,
            fg.pose_variables[0].true_rotation_matrix,
            fg.pose_variables[0].true_position,
            1 / noise_parameters.prior_rot_kappa_langevin_inverse,
            1 / noise_parameters.prior_pos_cov,
        )
    )
    if unknown_data_association:
        print("UNKNWON DATA ASSOCIATION CASE - # POSE LANDMARK MEAS")
        print(fg.pose_landmark_measurements)
        print(len(fg.pose_landmark_measurements))
    if unknown_data_association:
        gen_se2.set_some_landmark_measurements_to_uda_se2(fg, fraction_uda=1)
    if unknown_data_association:
        print("UNKNWON DATA ASSOCIATION CASE - # UDA MEAS")
        print(fg.unknown_data_association_measurements)
        print(len(fg.unknown_data_association_measurements))
        bop = 1
    subtask_factors.convert_to_localization_task(fg)
    print(len(fg.pose_known_landmark_measurements))
    bop = 1
    # stamps_rel_pos = [meas.timestamp for meas in fg.pose_known_landmark_measurements]
    # stamps_states = [state.timestamp for state in fg.pose_variables]
    # state_times
    return fg


import certifiable_uda_loc.path_configuration as path_config


def main():
    tmin = 200
    tmax = 210
    liw_ds: dataset.LostInTheWoodsDataset = dataset.LostInTheWoodsDataset.from_mat_file(
        path_config.lost_in_the_woods_dataset_path,
        time_bounds=[tmin, tmax],
    )
    # get_factor_graph_from_liw(liw_ds, [tmin, tmax], 3, list(range(17)))
    get_factor_graph_from_liw(liw_ds, [tmin, tmax], 3, [1, 2, 3])


if __name__ == "__main__":
    main()
