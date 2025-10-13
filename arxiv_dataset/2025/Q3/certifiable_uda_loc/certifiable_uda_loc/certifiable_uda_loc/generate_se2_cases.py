from certifiable_uda_loc.settings import Se2CaseNoiseParameters
from dataclasses import dataclass
from collections import namedtuple
import certifiable_uda_loc.noise as noise
import certifiable_uda_loc.utils.matrix_utils as matrix_utils
from typing import Union
import certifiable_uda_loc.problem as rf
import py_factor_graph as pfg
from py_factor_graph.variables import PoseVariable2D, LandmarkVariable2D

from py_factor_graph.data_associations import (
    VectorVariable,
    BooleanVariable,
    UnknownDataAssociationMeasurement,
    ToyExamplePrior,
)
import py_factor_graph.data_associations as pfg_da
import numpy as np
from certifiable_uda_loc.utils.matrix_utils import initialize_poly_matrix
from typing import List, Dict
from collections import namedtuple
from py_factor_graph.factor_graph import FactorGraphData
from certifiable_uda_loc.toy_example import (
    set_ground_truth_values,
)
import py_factor_graph.variables as pfg_variables
import py_factor_graph.measurements as pfg_measurements
import pymlg
import certifiable_uda_loc.test_cases as test_cases
import certifiable_uda_loc.problem_setup as prob_setup
from certifiable_uda_loc.trials import (
    MonteCarloSummary,
)
from certifiable_uda_loc.variables import LiftedQcQpVariable, ColumnQcQcpVariable
import certifiable_uda_loc.variables as variables
from certifiable_uda_loc.unlifting import (
    unlift,
    unlift_lifted_variables_to_factor_graph,
)
import certifiable_uda_loc.utils.string_utils as string_utils
from typing import Tuple
import pymlg
from py_factor_graph.data_associations import PosePrior2D
import py_factor_graph.data_associations as pyfg_da
from loguru import logger


def pose_name_from_timestep(timestep: float):
    return f"T{timestep}"


def timestep_from_pose_name(pose_name: str):
    assert pose_name[0] == "T"
    return float(pose_name[1:])


def landmark_name_from_idx(idx: int):
    return f"L{idx}"


def idx_from_landmark_name(name: str):
    assert name[0] == "L"
    return float(name[1:])


def generate_se2_gt_states(num_poses: int, num_landmarks: int, landmark_scale: float):
    poses = [pymlg.SE2.random() for lv1 in range(num_poses)]
    landmarks = [np.random.random(2) * landmark_scale for lv1 in range(num_landmarks)]
    return poses, landmarks


def generate_random_integers(start, stop, amount):
    num_to_remove = amount
    landmark_indices = np.arange(start, stop, 1)
    int_list = np.random.choice(landmark_indices, size=num_to_remove, replace=False)
    return int_list


def set_some_landmark_measurements_to_uda_se2(
    fg: FactorGraphData, fraction_uda=0.5
) -> None:
    # modifies fg by reference

    landmark_names = [l.name for l in fg.landmark_variables]
    pose_landmark_measurements = fg.pose_landmark_measurements
    n_meas = len(fg.pose_landmark_measurements)
    # remove_list = (np.random.random((n_meas)) < fraction_to_remove).tolist()
    num_to_uda = int(np.floor(fraction_uda * n_meas))
    set_to_uda_list = generate_random_integers(0, n_meas, num_to_uda)

    # set_to_uda_list = np.random.random((len(pose_landmark_measurements))) < fraction_uda

    uda_measurement_list = []
    bool_var_names = []
    for lv1, true_meas in enumerate(pose_landmark_measurements):
        if lv1 not in set_to_uda_list:
            continue
        meas_list_for_uda = []
        bool_var_list = []
        true_landmark_name = true_meas.landmark_name
        for lv_landmark in range(len(landmark_names)):
            meas = pfg_measurements.PoseToLandmarkMeasurement2D(
                pose_name=true_meas.pose_name,
                landmark_name=landmark_names[lv_landmark],
                r_b_lb=true_meas.r_b_lb,
                weight=true_meas.weight,
                timestamp=true_meas.timestamp,
            )
            # bool_var_name = f"(th-{true_meas.pose_name}-{landmark_names[lv_landmark]})"
            bool_var_name = f"(th-{lv_landmark}-group-{lv1}-pose-{true_meas.pose_name})"
            # if meas_per_landmark > 1:
            #     bool_var_name = f"{bool_var_name[:-1]}-meas-idx-{lv1})"
            # if bool_var_name in bool_var_names:
            #     bop = 1
            bool_var_names.append(bool_var_name)
            # if meas_per_landmark > 1:
            #     bool_var_name += "-meas-idx-{lv1})"
            boolean_var = pfg_da.BooleanVariable(
                name=bool_var_name,
                true_value=(landmark_names[lv_landmark] == true_landmark_name),
            )
            meas_list_for_uda.append(meas)
            bool_var_list.append(boolean_var)
        uda_meas = pfg_da.UnknownDataAssociationMeasurement(
            meas_list_for_uda, bool_var_list, timestamp=meas_list_for_uda[0].timestamp
        )
        uda_measurement_list.append(uda_meas)
    fg.unknown_data_association_measurements = uda_measurement_list
    fg.pose_landmark_measurements = [
        meas
        for lv1, meas in enumerate(fg.pose_landmark_measurements)
        if lv1 not in set_to_uda_list
    ]
    bop = 1


def remove_random_set_of_pose_landmark_measurements(
    fg, fraction_to_remove: float = 0.5
) -> None:
    # modifies fg by reference
    n_meas = len(fg.pose_landmark_measurements)
    num_to_remove = int(np.floor(fraction_to_remove * n_meas))
    remove_list = generate_random_integers(0, n_meas, num_to_remove)
    fg.pose_landmark_measurements = [
        meas
        for lv1, meas in enumerate(fg.pose_landmark_measurements)
        if lv1 not in remove_list
    ]


# Naming is mildly horrendous
def create_se2_factor_graph(
    n_poses: int,
    n_landmarks: int,
    landmark_spread: float,
    meas_per_landmark: int,
    noise_parameters: Se2CaseNoiseParameters,
    fraction_removal: float = 0,
    uda_fraction: float = 0,
    add_noise: bool = False,
    prior_landmark_location="random",
):
    logger.info(f"Creating se2 factor graph..")
    poses_gt, landmarks_gt = generate_se2_gt_states(
        n_poses, n_landmarks, landmark_spread
    )
    fg = generate_se2_factor_graph(
        poses_gt,
        landmarks_gt,
        meas_per_landmark,
        noise_parameters,
        add_noise=add_noise,
        prior_landmark_location=prior_landmark_location,
    )
    remove_random_set_of_pose_landmark_measurements(
        fg, fraction_to_remove=fraction_removal
    )
    set_some_landmark_measurements_to_uda_se2(fg, uda_fraction)
    return fg


def generate_se2_factor_graph(
    poses_gt: List[np.ndarray],
    landmarks_gt: List[np.ndarray],
    meas_per_landmark: int,
    noise_parameters: Se2CaseNoiseParameters,
    add_noise: bool = True,
    locked_first_pose=False,
    prior_landmark_location="random",  # TODO: Add to problem_settings.
) -> FactorGraphData:
    fg = FactorGraphData(dimension=2)
    fg.pose_variables = []
    for pose_idx, pose in enumerate(poses_gt):
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
                timestamp=pose_idx,
            )
        )
    for landmark_idx, landmark_gt in enumerate(landmarks_gt):
        fg.landmark_variables.append(
            LandmarkVariable2D(
                name=landmark_name_from_idx(landmark_idx),
                true_position=landmark_gt.reshape(-1),
                estimated_position=-1000 * np.ones(2),
            ),
        )
    if noise_parameters.prior_landmark_cov is not None:
        logger.info(
            f"Creating landmark priors. Prior positions initialized using {prior_landmark_location}"
        )
        # I think there is some weird stuff with seeding, when the np.random.random
        # method is called. Even for localization example, where the priors are removed, different results
        # are obtaine dif this np random random method is called first
        for landmark_state in fg.landmark_variables:
            pos = None
            if prior_landmark_location == "gt":
                pos = landmark_state.true_position
            if prior_landmark_location == "random":
                pos = np.random.random(size=landmark_state.true_position.shape[0])
            if pos is None:
                continue
            fg.landmark_priors.append(
                pyfg_da.PositionPrior2D(
                    name=landmark_state.name,
                    position=pos,
                    weight=1 / noise_parameters.prior_landmark_cov,
                )
            )

    fg.prior_pose_measurements = []
    if not locked_first_pose:
        fg.prior_pose_measurements.append(
            PosePrior2D(
                fg.pose_variables[0].name,
                fg.pose_variables[0].true_rotation_matrix,
                fg.pose_variables[0].true_position,
                1 / noise_parameters.prior_rot_kappa_langevin_inverse,
                1 / noise_parameters.prior_pos_cov,
            )
        )

    fg.odom_measurements = []
    for lv_k, (pose_k, pose_kp) in enumerate(zip(poses_gt[:-1], poses_gt[1:])):
        dT = np.linalg.inv(pose_k) @ pose_kp
        dC, dr = pymlg.SE2.to_components(dT)
        # How about them weights...
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
        assert np.allclose(fg.odom_measurements[-1].rotation_matrix, dC)
        r_kp = np.array(fg.pose_variables[lv_k + 1].true_position)
        r_k = np.array(fg.pose_variables[lv_k].true_position)
        C_k = fg.pose_variables[lv_k].true_rotation_matrix
        dr = fg.odom_measurements[lv_k].translation_vector

        dr_a = (C_k @ dr.reshape(-1, 1)).reshape(
            -1,
        )
        assert np.allclose(
            r_kp,
            r_k + dr_a,
        )
    fg.pose_landmark_measurements = []
    for lv_k, pose_k in enumerate(poses_gt):
        if lv_k == 0 and locked_first_pose:
            continue
        for lv_landmark, r_a_la in enumerate(landmarks_gt):
            for lv_meas_per_landmark in range(meas_per_landmark):
                C_ab, r_a_ba = pymlg.SE2.to_components(pose_k)
                dr_b = C_ab.T @ (r_a_la - r_a_ba)
                fg.pose_landmark_measurements.append(
                    pfg_measurements.PoseToLandmarkMeasurement2D(
                        pose_name=fg.pose_variables[lv_k].name,
                        landmark_name=fg.landmark_variables[lv_landmark].name,
                        r_b_lb=dr_b,
                        weight=1 / noise_parameters.rel_landmark_meas_cov,
                        timestamp=fg.pose_variables[lv_k].timestamp,
                    )
                )

    # TODO: We will have to be a bit careful with ensuring how
    # these noise characteristics match the dataset...
    # relationship with covariances??
    if add_noise:
        for lv_prior in range(len(fg.prior_pose_measurements)):
            # pass
            fg.prior_pose_measurements[lv_prior].rotation = fg.prior_pose_measurements[
                lv_prior
            ].rotation @ noise.sample_so2_langevin(
                np.eye(2), 1 / noise_parameters.prior_rot_kappa_langevin_inverse
            )

        for lv_odom in range(len(fg.odom_measurements)):
            fg.odom_measurements[lv_odom].rotation_matrix = fg.odom_measurements[
                lv_odom
            ].rotation_matrix @ noise.sample_so2_langevin(
                np.eye(2), 1 / noise_parameters.rel_rot_kappa_langevin_inverse
            )
            fg.odom_measurements[lv_odom].translation_vector += np.random.normal(
                loc=0, scale=np.sqrt(noise_parameters.rel_pos_cov), size=(2,)
            )

        for lv_meas in range(len(fg.pose_landmark_measurements)):
            dmeas = np.random.normal(
                loc=0, scale=np.sqrt(noise_parameters.rel_landmark_meas_cov), size=(2,)
            )
            fg.pose_landmark_measurements[lv_meas].r_b_lb += dmeas

    return fg
