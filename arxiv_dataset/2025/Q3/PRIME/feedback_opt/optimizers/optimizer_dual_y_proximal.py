import numpy as np

from feedback_opt.optimizers.optimizer_base import OptimizerProximal
from feedback_opt.systems.system_base import SystemBase
from feedback_opt.utils import Argmin


class OptimizerDualYProximal(OptimizerProximal):
    """
    this algorithm dualizes the ss-map y = h(u)
    """

    def __init__(
        self,
        params,
        system: SystemBase,
    ):
        super().__init__(params=params, system=system)

        # cache argmin problem
        self.prob_prim_u = Argmin(system.U)

    def data_initial(self, u_0: np.ndarray = None) -> dict:
        data_k0 = super().data_initial(u_0)
        data_k0["lamb_y"] = np.zeros((self._system.Y.num_constr, 1))

        data_k0["p"] = np.zeros((self._system.m, 1))
        return data_k0

    def next_u(self, data_in: dict) -> np.ndarray:
        assert np.shape(data_in["u"]) == (self._system.m, 1)
        assert np.shape(data_in["y"]) == (self._system.p, 1)
        assert np.shape(data_in["lamb_y"]) == (self._system.Y.num_constr, 1)

        # build QP
        H = self._system.du_h(data_in["u"])

        quad = self.quad_u + H.T @ self.quad_y @ H + self.gamma_u / 2 * np.eye(self._system.m)
        lin = (
            self.lin_u
            + H.T @ self.lin_y
            - self.gamma_u * data_in["u"]
            + 2 * H.T @ self.quad_y @ (data_in["y"] - H @ data_in["u"])
            + H.T @ self._system.Y.A.T @ data_in["lamb_y"]
        )

        # solve QP
        u = self.prob_prim_u.solve(quad, lin, verify_psd=False)
        assert u is not None

        return u

    def next_lamb_y(self, data_in: dict) -> np.ndarray:
        assert np.shape(data_in["y"]) == (self._system.p, 1)
        assert np.shape(data_in["lamb_y"]) == (self._system.Y.num_constr, 1)

        # projected gradient ascent lamb_y
        lamb_y_hat = data_in["lamb_y"] + self.rho * self._system.Y.c_x(data_in["y"])
        return np.clip(lamb_y_hat, a_min=0, a_max=None)

    def data_step(self, data_in: dict) -> dict:
        # assert minimum input requirements
        assert "u" in data_in
        assert "y" in data_in
        assert "lamb_y" in data_in

        data_out = data_in.copy()

        ## output actor
        data_out["lamb_y"] = self.next_lamb_y(data_out)

        ## input actor
        # update
        data_out["u"] = self.next_u(data_out)

        # cost calculation
        H = self._system.du_h(data_in["u"])
        p = np.multiply(
            data_out["u"],
            H.T @ self._system.Y.A.T @ data_out["lamb_y"]
            + 2 * H.T @ self.quad_y @ data_in["y"]
            + H.T @ self.lin_y,
        )
        p += self.gamma_u / 2 * np.power(data_out["u"] - data_in["u"], 2)
        p += np.multiply(
            data_in["u"] - data_in["u"], H.T @ self.quad_y @ H @ (data_in["u"] - data_in["u"])
        )
        data_out["p"] = p

        return data_out
