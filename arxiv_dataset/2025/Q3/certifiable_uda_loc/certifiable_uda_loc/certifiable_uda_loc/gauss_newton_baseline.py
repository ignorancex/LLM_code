import navlie.batch.residuals as nv_residuals
from typing import List, Hashable, Tuple
from navlie.types import State
import numpy as np
import navlie.lib.states as nv_states
import pymlg


def group_generators(group: str):
    generators = None
    if group == "SO2":
        # First one is xi_phi; last two are the xi_r ones.
        Xi_phi = pymlg.SO2.wedge(1)
        generators = Xi_phi
    return generators


# TODO: Maybe this should be two separate residuals...
class PriorResidualFro(nv_residuals.Residual):
    def __init__(
        self,
        keys: List[Hashable],
        prior_state: State,
        weight_rot: float,
        weight_pos: float,
    ):
        super().__init__(keys)
        self._weight_rot = weight_rot
        self._weight_pos = weight_pos
        self._x0 = prior_state

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        x: nv_states.SE2State = states[0]
        x_check: nv_states.SE2State = self._x0
        C = x.attitude
        # Note that this is for right perturbation only with the Jacobian derivations.
        # For now also only for SE2.
        assert x.direction == "right"

        error = np.vstack(
            [
                np.sqrt(self._weight_rot)
                * np.matrix.flatten((C - x_check.attitude), "F").reshape(-1, 1),
                np.sqrt(self._weight_pos)
                * (x.position - x_check.position).reshape(-1, 1),
            ]
        )
        if compute_jacobians:
            jacobians = [None]
            nx = C.shape[0]
            generators_so2 = group_generators("SO2")
            flatGenMat = np.hstack([np.matrix.flatten(g, "F") for g in generators_so2])

            if compute_jacobians[0]:
                # TODO: The minus sign. I think i have an issue with the wedge sign convention? Check later.
                rot_jac = (
                    -np.sqrt(self._weight_rot) * np.kron(np.eye(nx), C) @ flatGenMat
                )
                pos_jac = np.sqrt(self._weight_pos) * x.attitude
                jacobians[0] = np.block(
                    [
                        [rot_jac.reshape(-1, 1), np.zeros((rot_jac.shape[0], 2))],
                        [np.zeros((pos_jac.shape[0], 1)), pos_jac],
                    ]
                )
            return error, jacobians

        return error

    def sqrt_info_matrix(self, states: List[State]):
        """
        Returns the square root of the information matrix
        """
        n_e_rot = 4
        n_e_pos = 2
        return np.diag(
            np.array(
                [np.sqrt(self._weight_rot)] * n_e_rot
                + [np.sqrt(self._weight_pos)] * n_e_pos
            )
        )


class RelativePoseResidualFro(nv_residuals.Residual):

    def __init__(
        self, keys: List[Hashable], dT: np.ndarray, weight_rot: float, weight_pos: float
    ):
        super().__init__(keys)
        self.dT: np.ndarray = dT
        self._weight_rot = weight_rot
        self._weight_pos = weight_pos

    def evaluate(
        self,
        states: List[State],
        compute_jacobians: List[bool] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:

        x_k: nv_states.SE2State = states[0]
        x_kp: nv_states.SE2State = states[1]
        assert x_k.direction == "right"
        assert x_kp.direction == "right"
        dC, dr = pymlg.SE2.to_components(self.dT)
        C_k, r_k = pymlg.SE2.to_components(x_k.pose)
        C_kp, r_kp = pymlg.SE2.to_components(x_kp.pose)

        error = np.vstack(
            [
                np.sqrt(self._weight_rot)
                * np.matrix.flatten((C_kp - C_k @ dC), "F").reshape(-1, 1),
                np.sqrt(self._weight_pos)
                * (r_kp - r_k - (C_k @ dr.reshape(-1, 1)).reshape(-1)).reshape(-1, 1),
            ]
        )

        # Compute the Jacobians of the residual w.r.t x_km1 and x_k
        if compute_jacobians:
            jac_list = [None] * len(states)
            generators_so2 = group_generators("SO2")
            flatGenMat = np.hstack([np.matrix.flatten(g, "F") for g in generators_so2])
            if compute_jacobians[0]:

                D_e_rot_dC_k = np.kron(dC.T, C_k) @ flatGenMat.reshape(-1, 1)
                P = np.array([[0, -1], [1, 0]])
                D_e_pos_dC_k = (
                    -C_k @ P @ dr
                )  # TODO: Check minus sign.... dont have in derivaiton. Again maybe generator shenanigans?
                D_e_pos_dr_k = -C_k

                D_e_rot_dC_k *= np.sqrt(self._weight_rot)
                D_e_pos_dC_k *= np.sqrt(self._weight_pos)
                D_e_pos_dr_k *= np.sqrt(self._weight_pos)

                jac_list[0] = np.block(
                    [
                        [D_e_rot_dC_k, np.zeros((D_e_rot_dC_k.shape[0], 2))],
                        [D_e_pos_dC_k.reshape(-1, 1), D_e_pos_dr_k],
                    ]
                )
            # TODO: Keep going on C_kp Jacs.
            if compute_jacobians[1]:
                D_e_rot_dC_kp = -np.kron(np.eye(2), C_kp) @ flatGenMat.reshape(-1, 1)
                D_e_rot_dC_kp *= np.sqrt(self._weight_rot)
                D_e_pos_dr_kp = np.sqrt(self._weight_pos) * C_kp
                jac_list[1] = np.block(
                    [
                        [D_e_rot_dC_kp, np.zeros((D_e_rot_dC_kp.shape[0], 2))],
                        [np.zeros((2, 1)), D_e_pos_dr_kp],
                    ]
                )
            assert error.shape[0] == jac_list[0].shape[0]
            assert error.shape[0] == jac_list[1].shape[0]
            return error, jac_list

        return error

    def sqrt_info_matrix(self, states: List[State]):
        n_e_rot = 4
        n_e_pos = 2
        return np.diag(
            np.array(
                [np.sqrt(self._weight_rot)] * n_e_rot
                + [np.sqrt(self._weight_pos)] * n_e_pos
            )
        )
