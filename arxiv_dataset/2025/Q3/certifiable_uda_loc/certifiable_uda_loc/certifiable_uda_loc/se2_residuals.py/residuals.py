from typing import Hashable, List, Tuple

import numpy as np
from mixtures.lie_util import minus_jacobian

# from pymlg.numpy import SO2, SE2
from pymlg import SE2, SO2

from navlie.batch.residuals import Residual
from navlie.filters import ExtendedKalmanFilter
from navlie.lib.models import VectorInput
from navlie.lib.states import SE2State, VectorState
from navlie.types import ProcessModel, State, StateWithCovariance


def enforce_angle_domain(th):
    if th > np.pi:
        th = th - 2 * np.pi
    if th < -np.pi:
        th = th + 2 * np.pi
    return th


class DirectIntegrationResidual(Residual):
    u_list: List[VectorInput]
    process_model: ProcessModel
    kf: ExtendedKalmanFilter
    x_dr: List[State] = None

    def __init__(
        self,
        keys: List[Hashable],
        process_model: ProcessModel,
        u_list: List[VectorInput],
        x_dr: List[State] = None,
    ):
        super().__init__(keys)
        self.u_list = u_list
        self.process_model = process_model
        self.kf = ExtendedKalmanFilter(process_model)
        self.x_dr = x_dr

    def evaluate_unweighted(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        x consists of x_km and x_k.
        The inputs u have stamps from
        x_km -----   --------  ----------- --------- x_k
        u_i ------u_i+1 ------ u_i+2 ----------- u_{k-1}
        where i = km.
        """
        x_km1 = states[0].copy()
        x_k = states[1].copy()
        dt_x = x_k.stamp - x_km1.stamp

        x_k_hat = StateWithCovariance(x_km1, np.zeros((x_km1.dof, x_km1.dof)))
        jac_list_interoceptive: List[np.ndarray] = (
            []
        )  # Process Model Jacobians for each intermediate input
        #        x_k_hat = self._process_model.evaluate(x_km1.copy(), self._u, dt)
        x_k_hat_list = [x_k_hat.copy()]
        nt = len(self.u_list)
        for lv1 in range(1, nt):
            u_input = self.u_list[lv1]
            if lv1 == len(self.u_list) - 1:
                dt = x_k.stamp - self.u_list[lv1 - 1].stamp
            else:
                dt = self.u_list[lv1 + 1].stamp - u_input.stamp
            x_k_hat, details_dict = self.kf.predict(
                x_k_hat, u_input, dt, output_details=True
            )
            A_i = details_dict["A"]
            jac_list_interoceptive.append(A_i)
            x_k_hat_list.append(x_k_hat.copy())
        e = x_k.minus(x_k_hat.state)
        cov = x_k_hat.covariance
        if compute_jacobians:
            jac_list = [None] * len(states)
            if compute_jacobians[0]:
                # Chain jacobians for each input
                jac_pm_agglomerated = np.eye(jac_list_interoceptive[0].shape[0])
                for lv1 in range(len(jac_list)):
                    jac_pm_agglomerated = (
                        jac_list_interoceptive[lv1] @ jac_pm_agglomerated
                    )
                # Minus Jacobian. Needed w.r.t. second argument, X.
                # Y = x_k, X = x_k_hat = f(x_km1)
                jac_list[0] = (
                    minus_jacobian(x_k, x_k_hat.state, "X") @ jac_pm_agglomerated
                )
            if compute_jacobians[1]:
                jac_list[1] = minus_jacobian(x_k, x_k_hat.state, "Y")

            return e, jac_list, cov, x_k_hat.state

        return e, cov, x_k_hat.state

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        x consists of x_km and x_k.
        The inputs u have stamps from
        x_km -----   --------  ----------- --------- x_k
        u_i ------u_i+1 ------ u_i+2 ----------- u_{k-1}
        where i = km.
        """
        jac_list_unweighted = None
        if compute_jacobians:
            e_unweighted, jac_list_unweighted, cov, x_k_hat = self.evaluate_unweighted(
                states, compute_jacobians
            )
        else:
            e_unweighted, cov, x_k_hat = self.evaluate_unweighted(
                states, compute_jacobians
            )

        x_km1 = states[0]
        x_k = states[1]
        x_k_hat: State = x_k_hat
        minus_jac = minus_jacobian(x_k, x_k_hat, "X")
        # Q_batch = minus_jac @ cov @ minus_jac.T
        Q_batch = cov
        # Q_batch = cov
        L = np.linalg.cholesky(np.linalg.inv(Q_batch))
        e = L.T @ e_unweighted

        if compute_jacobians:
            jac_list = [None] * len(states)
            if compute_jacobians[0]:
                jac_list[0] = L.T @ jac_list_unweighted[0]
            if compute_jacobians[1]:
                jac_list[1] = L.T @ jac_list_unweighted[1]

            return e, jac_list

        return e


class RangeBearingResidual(Residual):
    """
    Keys must contain
    "robot_{timestamp_str}" corresponding to an SE2State
    as well as
    "landmark_{idx}" where idx is the landmark index.
    """

    stamp: float
    range_value: float
    bearing_value: float
    cov: np.ndarray
    L: np.ndarray
    r_b_db: np.ndarray

    def __init__(
        self,
        keys: List[Hashable],
        range_value: float,
        bearing_value: float,
        cov_range: float,
        cov_bearing: float,
        r_b_db: np.ndarray,
    ):
        super().__init__(keys)
        self.range_value = range_value
        self.bearing_value = bearing_value
        self.cov = np.diag([cov_range, cov_bearing])
        # Precompute square-root of info matrix
        self.L = np.linalg.cholesky(np.linalg.inv(self.cov))
        self.r_b_db = r_b_db

    def sqrt_info_matrix(self, states: List[State]):
        return self.L

    def copy(self):
        return RangeBearingResidual(
            keys=self.keys.copy(),
            range_value=self.range_value,
            bearing_value=self.bearing_value,
            cov_range=self.cov[0, 0],
            cov_bearing=self.cov[1, 1],
            r_b_db=self.r_b_db,
        )

    def invert_measurement_model(self, robot_state: SE2State):
        C_ab, r_a_ba = SE2.to_components(robot_state.value)
        r_b_ld = np.array(
            [
                self.range_value * np.cos(self.bearing_value),
                self.range_value * np.sin(self.bearing_value),
            ]
        )
        return (
            C_ab @ (r_b_ld + self.r_b_db).reshape(-1, 1)
        ).squeeze() + r_a_ba.squeeze()

    def evaluate(self, states: List[State], compute_jacobians: List[bool] = None):
        robot_state: SE2State = states[0]
        T_ab = robot_state.value
        landmark_state: VectorState = states[1]

        r_a_ld = RelLandmarkPos(
            T_ab=robot_state.value,
            r_b_db=self.r_b_db,
            landmark_pos=landmark_state.value,
        )
        # range_check = np.linalg.norm(r_a_ld, 2)
        range_check = np.sqrt(r_a_ld[0] ** 2 + r_a_ld[1] ** 2).squeeze()
        e_range = self.range_value - range_check

        vehicle_angle = enforce_angle_domain(
            SO2.vee(SO2.log(SE2.to_components(T_ab)[0]))
        )
        bearing_check = enforce_angle_domain(
            np.arctan2(r_a_ld[1], r_a_ld[0])
        ) - enforce_angle_domain(vehicle_angle)
        e_bearing = (self.bearing_value - bearing_check).squeeze()
        e_bearing = enforce_angle_domain(e_bearing)
        if np.abs(e_bearing) > 0.5:
            meow = 1
        error = self.L.T @ np.array([e_range, e_bearing]).reshape(-1, 1)
        error = error.squeeze()
        if compute_jacobians:
            jacobians = [None] * 2
            dRangeFirstPart = sqrt_norm_gradient(r_a_ld, norm_v=range_check)
            dBearingAtanFirstPart = (
                1 / range_check**2 * np.array([-r_a_ld[1], r_a_ld[0]]).reshape(1, -1)
            )
            if compute_jacobians[0]:
                Dr_a_ljd = D_RelLandmarkPosD_X(T_ab, self.r_b_db)
                dRange = dRangeFirstPart @ Dr_a_ljd
                # First term is atan2. Second term is derivative of vehicle angle, which is
                # derivative of the logMap.
                dBearingAtan = (dBearingAtanFirstPart @ Dr_a_ljd).reshape(1, -1)
                dRange = dRange.reshape(1, -1)

                # dBearingVehicleAngle = (
                # np.linalg.inv(SE2.left_jacobian(SE2.vee(SE2.log(T_ab))))[0, :]
                # ).reshape(1, -1)
                dBearingVehicleAngle = (
                    SE2.inverse(SE2.left_jacobian(SE2.vee(SE2.log(T_ab))))[0, :]
                ).reshape(1, -1)

                dBearing = dBearingAtan - dBearingVehicleAngle
                dBearing = dBearing.reshape(1, -1)
                jacobians[0] = -self.L.T @ np.vstack([dRange, dBearing])
            if compute_jacobians[1]:
                Dr_a_ljd = np.eye(2)
                dRange = dRangeFirstPart @ Dr_a_ljd
                dBearing = (dBearingAtanFirstPart @ Dr_a_ljd).reshape(1, -1)
                dRange = dRange.reshape(1, -1)
                dBearing = dBearing.reshape(1, -1)

                jacobians[1] = -self.L.T @ np.vstack([dRange, dBearing])
            return error, jacobians
        return error


def sqrt_norm_gradient(v: np.ndarray, norm_v: float = None):
    if norm_v is None:
        return 1 / np.linalg.norm(v, 2) * v.reshape(1, -1)
    else:
        if norm_v == 0:
            bop = 1
        return 1 / norm_v * v.reshape(1, -1)


def RelLandmarkPos(T_ab: np.ndarray, r_b_db: np.ndarray, landmark_pos: np.ndarray):
    D = np.hstack([np.eye(2), np.zeros((2, 1))])
    r_a_la = landmark_pos.reshape(-1, 1)  # landmark position in inertial frame
    p = np.append(r_b_db, np.array(1)).reshape(-1, 1)
    return r_a_la - D @ T_ab @ p


def D_RelLandmarkPosD_X(T_ab: np.ndarray, r_b_db: np.ndarray) -> np.ndarray:
    """Compute Jacobian of vector from front of vehicle to landmark, in the vehicle frame,
    with respect to the SE2 vehicle state.

    Parameters
    ----------
    T_ab : np.ndarray
        3x3 robot pose
    r_b_db : np.ndarray
        2d vector describing vehicle to rangefinder extrinsics.

    Returns
    -------
    np.ndarray
        2x3 numpy array containing the Jacobian
    """
    r_b_db = r_b_db.reshape(-1, 1)
    C_ab, r_a_ba = SE2.to_components(T_ab)

    r_a_ba = r_a_ba.reshape(-1, 1)

    r_a_ba = r_a_ba.reshape(-1, 1)
    DrDphi = -np.array([[0, -1], [1, 0]]) @ ((C_ab @ r_b_db).reshape(-1, 1) + r_a_ba)
    DrDphi = DrDphi.reshape(-1, 1)
    DrDpos = -np.eye(2)
    return np.hstack([DrDphi, DrDpos])
