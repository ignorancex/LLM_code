import certifiable_uda_loc.utils.matrix_utils as matrix_utils
from typing import Union
import certifiable_uda_loc.problem as rf
from py_factor_graph.data_associations import (
    VectorVariable,
    PoseVariable2D,
    BooleanVariable,
    UnknownDataAssociationMeasurement,
    ToyExamplePrior,
)
import py_factor_graph.data_associations as pfg_mod_da
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
import certifiable_uda_loc.generate_se2_cases as gen_se2
import certifiable_uda_loc.utils.print_utils as print_utils
import certifiable_uda_loc.lifting as lifting
import certifiable_uda_loc.subtask_factors as subtask_factors


def test_known_pose_cost():
    np.random.seed(3)
    landmark_pos = np.random.random(
        2,
    )
    r_a_la = landmark_pos
    hom_var = variables.HomogenizationVariable((2, 2), "one", true_value=np.eye(2))
    pose = pymlg.SE2.random()
    C_ab, r_a_ba = pymlg.SE2.to_components(pose)
    true_meas_value = C_ab.T @ (r_a_la - r_a_ba)
    meas_unknown_pose = pfg_measurements.PoseToLandmarkMeasurement2D(
        pose_name="T1",
        landmark_name="L1",
        r_b_lb=true_meas_value,
        weight=1,
    )

    meas: pfg_measurements.KnownPoseToLandmarkMeasurement2D = (
        subtask_factors.fix_robot_pose(meas_unknown_pose, C_ab, r_a_ba)
    )
    Q_list = lifting.lift_factor_to_Q(meas, hom_var)
    X_lifted = np.hstack([np.eye(2), r_a_la.reshape(-1, 1)])
    lifted_error = np.trace(Q_list[0].toarray() @ X_lifted.T @ X_lifted)
    assert lifted_error < 1e-10


def test_known_landmark_cost():
    np.random.seed(3)
    landmark_pos = np.random.random(
        2,
    )
    hom_var = variables.HomogenizationVariable((2, 2), "one", true_value=np.eye(2))
    pose = pymlg.SE2.random()
    # pose[:2, :2] = np.eye(2)
    # pose[:2, 2] = np.zeros(2)
    pose[:2, 2] = 500 * np.ones(2)
    r_a_la = landmark_pos
    C_ab, r_a_ba = pymlg.SE2.to_components(pose)
    true_meas_value = C_ab.T @ (r_a_la - r_a_ba)
    meas_unknown_landmark = pfg_measurements.PoseToLandmarkMeasurement2D(
        pose_name="T1",
        landmark_name="L1",
        r_b_lb=true_meas_value,
        weight=1,
    )

    meas = subtask_factors.fix_landmark_position(meas_unknown_landmark, landmark_pos)
    meas = pfg_measurements.PoseToKnownLandmarkMeasurement2D(
        "T1", r_a_la=landmark_pos, r_b_lb=true_meas_value, weight=1
    )
    # Print error from dot products
    ell = r_a_la
    y = true_meas_value

    # Test trace shuffling trick
    prod_y_ell = ell.reshape(-1, 1) @ y.reshape(1, -1)
    print(np.sum(prod_y_ell * C_ab))
    print(ell.reshape(1, -1) @ C_ab @ y.reshape(-1, 1))
    diff = np.sum(prod_y_ell * C_ab) - ell.reshape(1, -1) @ C_ab @ y.reshape(-1, 1)
    assert np.abs(diff) < 1e-8
    err_dot_summands = [
        np.linalg.norm(ell) ** 2 + np.linalg.norm(y) ** 2,
        np.linalg.norm(r_a_ba) ** 2,
        -2 * r_a_ba.T @ ell,
        (-2 * ell.reshape(1, -1) @ C_ab @ y.reshape(-1, 1))[0, 0],
        (2 * r_a_ba.reshape(1, -1) @ C_ab @ y.reshape(-1, 1))[0, 0],
    ]
    err_mid = np.sum(err_dot_summands)
    Q_list = lifting.lift_factor_to_Q(meas, hom_var)
    X_lifted = np.hstack([np.eye(2), C_ab, r_a_ba.reshape(-1, 1)])

    Q = Q_list[0]

    slices_list = [
        (slice(0, 1), slice(0, 1)),
        (slice(4, 5), slice(4, 5)),
        (slice(0, 2), slice(4, 5)),
        (slice(0, 2), slice(2, 4)),
        (slice(2, 4), slice(4, 5)),
    ]
    diagonal_indices = [0, 1]

    print(Q)
    for lv_sum in range(len(err_dot_summands)):
        slice_i, slice_j = slices_list[lv_sum]
        Q_subset = Q.toarray()[slice_i, slice_j]
        XTX_subset = (X_lifted.T @ X_lifted)[slice_i, slice_j]
        if lv_sum not in diagonal_indices:
            mult = 2
        else:
            mult = 1
        err = np.abs(err_dot_summands[lv_sum] - mult * np.sum(Q_subset * XTX_subset))
        assert err < 0.1
        # print(f"Successful at {lv_sum}")

    print("Manually computing dot product", err_mid)
    # Should be zero...
    lifted_error = np.trace(Q_list[0].toarray() @ X_lifted.T @ X_lifted)
    # go compare them..
    print("Lifted error..")
    print(lifted_error)
    assert np.abs(lifted_error) < 1e-10


def test_rotation_frobenius():
    noise_parameters = gen_se2.Se2CaseNoiseParameters(
        prior_rot_kappa_langevin_inverse=0.1,
        prior_pos_cov=0.1,
        rel_rot_kappa_langevin_inverse=0.1,
        rel_pos_cov=0.1,
        rel_landmark_meas_cov=0.1,
        prior_landmark_cov=0.1,
        prior_landmark_noise_corrupt_cov=0.1,
    )
    poses_gt, landmarks_gt = gen_se2.generate_se2_gt_states(2, 0, 4)
    fg: FactorGraphData = gen_se2.generate_se2_factor_graph(
        poses_gt, landmarks_gt, 1, noise_parameters, add_noise=False
    )

    opt_variables = []
    pose_to_lifted_var_indices: List[Tuple] = []
    landmark_to_lifted_var_indices: List[int] = []
    for var in fg.pose_variables:
        cur_vars = rf.lift_continuous_variable(var)
        rot_qcqp_var, pos_qcqp_var = cur_vars[0], cur_vars[1]
        opt_variables.append(rot_qcqp_var)
        opt_variables.append(pos_qcqp_var)

    column_vars = []
    for var in opt_variables:
        var: variables.LiftedQcQpVariable
        for lv1, column_var in enumerate(var.column_variables):
            column_var: ColumnQcQcpVariable = column_var
            column_vars.append(column_var)
            assert np.allclose(column_var.true_value, var.true_value[:, lv1])

    nx = opt_variables[0].dims[0]
    hom_var = variables.HomogenizationVariable((nx, nx), "one", true_value=np.eye(nx))
    lifted_variable_names_all = hom_var.column_names() + [
        var.name for var in column_vars
    ]

    X_true_np = np.hstack(
        [
            np.eye(2),
            fg.pose_variables[0].true_rotation_matrix,
            np.array(fg.pose_variables[0].true_position).reshape(-1, 1),
            fg.pose_variables[1].true_rotation_matrix,
            np.array(fg.pose_variables[1].true_position).reshape(-1, 1),
        ]
    )

    # Checking rotation specifically
    # assert np.trace(np.eye(2) - dC @ C1.T @ C0) < 1e-15
    # C_true_np = np.hstack([np.eye(2), C0, C1])
    # Q_c = np.zeros((6, 6))
    # Q_c[:2, :2] = 2 * np.eye(2)
    # Q_c[2:4, 4:6] = -2 * dC
    # Q_c_cost = np.sum(Q_c * (C_true_np.T @ C_true_np))

    C1 = fg.pose_variables[1].true_rotation_matrix
    C0 = fg.pose_variables[0].true_rotation_matrix
    dC = fg.odom_measurements[0].rotation_matrix
    r0 = np.array(fg.pose_variables[0].true_position)
    r1 = np.array(fg.pose_variables[1].true_position)
    dr = fg.odom_measurements[0].translation_vector

    # Checking translation specifically
    Jr_true = (
        np.linalg.norm(
            r1.reshape(-1, 1) - r0.reshape(-1, 1) - C0 @ dr.reshape(-1, 1), 2
        )
        ** 2
    )
    lifted_var_names = ["hom1", "hom2", "c01", "c11", "r0", "r1"]
    X_true = np.hstack([np.eye(2), C0, r0.reshape(-1, 1), r1.reshape(-1, 1)])
    Q_r = np.zeros((6, 6))
    Q_r[0, 0] = np.linalg.norm(dr) ** 2
    Q_r[2:4, 4] = 2 * dr
    Q_r[2:4, 5] = -2 * dr
    Q_r[4, 5] = -2
    Q_r[4, 4] = 1
    Q_r[5, 5] = 1
    Jr_lifted = np.sum(Q_r * (X_true.T @ X_true))
    print_utils.pretty_print_array(
        Q_r,
        rows=lifted_var_names,
        columns=lifted_var_names,
    )

    assert np.abs(Jr_lifted - Jr_true < 1e-14)
    # print(Q_c_cost)
    print_utils.pretty_print_array(
        X_true_np,
        rows=hom_var.column_names(),
        columns=lifted_variable_names_all,
    )

    dT = (
        fg.pose_variables[1].true_transformation_matrix
        @ np.linalg.inv(fg.pose_variables[0].true_transformation_matrix)
        - fg.odom_measurements[0].transformation_matrix
    )

    Q_list = lifting.lift_factor_to_Q(fg.odom_measurements[0], hom_var)
    for Q in Q_list:
        Q_all = rf.initialize_poly_matrix(lifted_variable_names_all)
        Q_all = Q_all + Q

        cost_lifted = np.trace(Q_all.toarray() @ X_true_np.T @ X_true_np)
        assert cost_lifted < 1e-12


if __name__ == "__main__":

    np.random.seed(11)
    test_rotation_frobenius()
    test_known_landmark_cost()
    test_known_pose_cost()
