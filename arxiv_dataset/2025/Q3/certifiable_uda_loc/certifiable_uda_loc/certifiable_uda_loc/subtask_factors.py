import py_factor_graph.measurements as pfg_measurements
from py_factor_graph.factor_graph import FactorGraphData
import numpy as np


def fix_landmark_position(
    meas: pfg_measurements.PoseToLandmarkMeasurement2D,
    true_landmark_position: np.ndarray,
) -> pfg_measurements.PoseToKnownLandmarkMeasurement2D:
    return pfg_measurements.PoseToKnownLandmarkMeasurement2D(
        pose_name=meas.pose_name,
        r_a_la=true_landmark_position,
        r_b_lb=np.array(meas.translation_vector),
        weight=meas.weight,
        timestamp=meas.timestamp,
    )


def fix_robot_pose(
    meas: pfg_measurements.PoseToLandmarkMeasurement2D,
    C_ab: np.ndarray,
    r_a_ba: np.ndarray,
):
    return pfg_measurements.KnownPoseToLandmarkMeasurement2D(
        C_ab=C_ab,
        r_a_ba=r_a_ba,
        r_b_lb=np.array(meas.translation_vector),
        weight=meas.weight,
        timestamp=meas.timestamp,
        landmark_name=meas.landmark_name,
    )


def convert_to_localization_task(fg: FactorGraphData):
    landmark_to_state_dict = {}
    for landmark_state in fg.landmark_variables:
        landmark_to_state_dict[landmark_state.name] = landmark_state

    for meas in fg.pose_landmark_measurements:
        fg.pose_known_landmark_measurements.append(
            fix_landmark_position(
                meas, landmark_to_state_dict[meas.landmark_name].true_position
            )
        )
    fg.pose_landmark_measurements = []

    # Assumption: All of fg.unknown_data_association_measurements are created
    # # from pose to landmark measurements
    for lv1, uda_meas in enumerate(fg.unknown_data_association_measurements):
        temp_meas_list = []
        for meas in uda_meas.measurement_list:
            meas: pfg_measurements.PoseToLandmarkMeasurement2D = meas
            temp_meas_list.append(
                fix_landmark_position(
                    meas,
                    landmark_to_state_dict[meas.landmark_name].true_position,
                )
            )
        fg.unknown_data_association_measurements[lv1].measurement_list = temp_meas_list
    fg.landmark_variables = []
    fg.landmark_priors = []


def convert_to_mapping_task(fg: FactorGraphData):
    pose_name_to_state_dict = {}
    for pose_state in fg.pose_variables:
        pose_name_to_state_dict[pose_state.name] = pose_state

    for meas in fg.pose_landmark_measurements:
        fg.pose_known_landmark_measurements.append(
            fix_robot_pose(
                meas,
                pose_name_to_state_dict[meas.pose_name].true_rotation_matrix,
                np.array(pose_name_to_state_dict[meas.pose_name].true_position),
            )
        )
    fg.pose_landmark_measurements = []

    # Assumption: All of fg.unknown_data_association_measurements are created
    # # from pose to landmark measurements
    for lv1, uda_meas in enumerate(fg.unknown_data_association_measurements):
        temp_meas_list = []
        for meas in uda_meas.measurement_list:
            meas: pfg_measurements.PoseToLandmarkMeasurement2D = meas
            temp_meas_list.append(
                fix_robot_pose(
                    meas,
                    pose_name_to_state_dict[meas.pose_name].true_rotation_matrix,
                    np.array(pose_name_to_state_dict[meas.pose_name].true_position),
                )
            )
        fg.unknown_data_association_measurements[lv1].measurement_list = temp_meas_list
    fg.pose_variables = []
    fg.odom_measurements = []
    fg.prior_pose_measurements = []
