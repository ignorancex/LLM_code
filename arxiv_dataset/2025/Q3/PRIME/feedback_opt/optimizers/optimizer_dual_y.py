import numpy as np

from feedback_opt.optimizers.optimizer_base import OptimizerGradientStep


class OptimizerDualY(OptimizerGradientStep):
    """
    this algorithm dualizes the ss-map y = h(u)
    """

    def data_initial(self, u_0: np.ndarray = None) -> dict:
        data_k0 = super().data_initial(u_0)
        data_k0["lamb_y"] = np.zeros((self._system.Y.num_constr, 1))

        data_k0["p"] = np.zeros((self._system.m, 1))
        return data_k0

    def next_u(self, data_in: dict) -> np.ndarray:
        assert np.shape(data_in["u"]) == (self._system.m, 1)
        assert np.shape(data_in["y"]) == (self._system.p, 1)
        assert np.shape(data_in["lamb_y"]) == (self._system.Y.num_constr, 1)

        du_h = self._system.du_h(data_in["u"])
        step_u = (
            self.du_phi_u(data_in["u"])
            + du_h.T @ self.dy_phi_y(data_in["y"])
            + du_h.T @ self._system.Y.A.T @ data_in["lamb_y"]
        )
        u_hat = data_in["u"] - self.alpha * step_u
        return self._system.U.proj_2(u_hat)

    def next_lamb_y(self, data_in: dict) -> np.ndarray:
        assert np.shape(data_in["y"]) == (self._system.p, 1)
        assert np.shape(data_in["lamb_y"]) == (self._system.Y.num_constr, 1)

        lamb_y_hat = data_in["lamb_y"] + self.beta * self._system.Y.c_x(data_in["y"])
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
        data_out["p"] = np.multiply(
            data_out["u"],
            self._system.du_h(data_in["u"]).T @ self._system.Y.A.T @ data_out["lamb_y"],
        )

        return data_out
