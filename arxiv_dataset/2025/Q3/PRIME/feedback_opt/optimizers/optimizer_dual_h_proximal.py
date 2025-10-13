import numpy as np

from feedback_opt.optimizers.optimizer_base import OptimizerProximal
from feedback_opt.systems.system_base import SystemBase
from feedback_opt.utils import Argmin


class OptimizerDualHProximal(OptimizerProximal):
    """
    this algorithm dualizes the ss-map y = h(u)
    """

    def __init__(
        self,
        params,
        system: SystemBase,
    ):
        super().__init__(params=params, system=system)

        # proximal z deviation cost
        if hasattr(params, "gamma_z"):
            assert isinstance(params.gamma_z, (int, float))
            self.gamma_z = float(params.gamma_z)
        else:
            self.gamma_z = 0.0

        # centralized?
        if hasattr(params, "centralized"):
            assert isinstance(params.centralized, bool)
            self.centralized = params.centralized
        else:
            self.centralized = True

        # cache argmin problems
        self.prob_prim_u = Argmin(system.U)
        self.prob_prim_z = Argmin(system.Y)

    def data_initial(self, u_0: np.ndarray = None) -> dict:
        data_k0 = super().data_initial(u_0)
        data_k0["z"] = self._system.Y.proj_2(data_k0["y"])
        data_k0["nu_h"] = np.zeros((self._system.p, 1))

        data_k0["p"] = np.zeros((self._system.m, 1))
        return data_k0

    def next_u(self, data_in: dict) -> np.ndarray:
        assert np.shape(data_in["u"]) == (self._system.m, 1)
        assert np.shape(data_in["y"]) == (self._system.p, 1)
        assert np.shape(data_in["z"]) == (self._system.p, 1)
        assert np.shape(data_in["nu_h"]) == (self._system.p, 1)

        # build QP
        H = self._system.du_h(data_in["u"])

        quad = self.quad_u + self.gamma_u / 2 * np.eye(self._system.m)
        if self.centralized:
            quad += self.rho / 2 * H.T @ H

        lin = self.lin_u + H.T @ data_in["nu_h"] - self.gamma_u * data_in["u"]
        if self.centralized:
            lin += self.rho * H.T @ (data_in["y"] - H @ data_in["u"] - data_in["z"])

        # solve QP
        u = self.prob_prim_u.solve(quad, lin, verify_psd=False)
        assert u is not None

        return u

    def next_z(self, data_in: dict) -> np.ndarray:
        assert np.shape(data_in["y"]) == (self._system.p, 1)
        assert np.shape(data_in["z"]) == (self._system.p, 1)
        assert np.shape(data_in["nu_h"]) == (self._system.p, 1)

        # build QP
        quad = self.quad_y + (self.rho + self.gamma_z) / 2 * np.eye(self._system.p)
        lin = self.lin_y - data_in["nu_h"] - self.rho * data_in["y"] - self.gamma_z * data_in["z"]

        # solve QP
        z = self.prob_prim_z.solve(quad, lin, verify_psd=False)
        assert z is not None

        return z

    def next_nu_h(self, data_in: dict) -> np.ndarray:
        assert np.shape(data_in["y"]) == (self._system.p, 1)
        assert np.shape(data_in["z"]) == (self._system.p, 1)
        assert np.shape(data_in["nu_h"]) == (self._system.p, 1)

        # gradient ascent nu_h
        return data_in["nu_h"] + self.rho * (data_in["y"] - data_in["z"])

    def data_step(self, data_in: dict) -> dict:
        # assert minimum input requirements
        assert "u" in data_in
        assert "y" in data_in
        assert "z" in data_in
        assert "nu_h" in data_in

        data_out = data_in.copy()

        ## output actor
        data_out["nu_h"] = self.next_nu_h(data_out)
        data_out["z"] = self.next_z(data_out)

        ## input actor
        # update
        data_out["u"] = self.next_u(data_out)

        # cost calculation
        H = self._system.du_h(data_in["u"])
        p = np.multiply(data_out["u"], H.T @ data_out["nu_h"])
        p += self.gamma_u / 2 * np.power(data_out["u"] - data_in["u"], 2)

        if self.centralized:
            p += (
                self.rho
                / 2
                * np.multiply(
                    data_out["u"] - data_in["u"], H.T @ H @ (data_out["u"] - data_in["u"])
                )
            )
            p += self.rho * np.multiply(data_out["u"], H.T @ (data_in["y"] - data_out["z"]))

        data_out["p"] = p

        return data_out
