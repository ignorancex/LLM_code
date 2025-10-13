import numpy as np

from feedback_opt.optimizers.optimizer_base import OptimizerGradientStep


class OptimizerDualH(OptimizerGradientStep):
    """
    this algorithm dualizes the ss-map y = h(u)
    """

    def data_initial(self, u_0: np.ndarray = None) -> dict:
        data_k0 = super().data_initial(u_0)
        data_k0["z"] = self._system.Y.proj_2(data_k0["y"])
        data_k0["nu_h"] = np.zeros((self._system.p, 1))

        data_k0["p"] = np.zeros((self._system.m, 1))
        return data_k0

    def next_u(self, data_in: dict) -> np.ndarray:
        assert np.shape(data_in["u"]) == (self._system.m, 1)
        assert np.shape(data_in["nu_h"]) == (self._system.p, 1)

        step_u = self.du_phi_u(data_in["u"]) + self._system.du_h(data_in["u"]).T @ data_in["nu_h"]
        u_hat = data_in["u"] - self.alpha * step_u
        return self._system.U.proj_2(u_hat)

    def next_z(self, data_in: dict) -> np.ndarray:
        assert np.shape(data_in["z"]) == (self._system.p, 1)
        assert np.shape(data_in["nu_h"]) == (self._system.p, 1)

        step_z = self.dy_phi_y(data_in["z"]) - data_in["nu_h"]
        z_hat = data_in["z"] - self.alpha * step_z
        return self._system.Y.proj_2(z_hat)

    def next_nu_h(self, data_in: dict) -> np.ndarray:
        assert np.shape(data_in["y"]) == (self._system.p, 1)
        assert np.shape(data_in["z"]) == (self._system.p, 1)
        assert np.shape(data_in["nu_h"]) == (self._system.p, 1)

        # gradient ascent nu_h
        return data_in["nu_h"] + self.beta * (data_in["y"] - data_in["z"])

    def data_step(self, data_in: dict) -> dict:
        # assert minimum input requirements
        assert "u" in data_in
        assert "y" in data_in
        assert "z" in data_in

        data_out = data_in.copy()

        ## output actor
        data_out["nu_h"] = self.next_nu_h(data_out)
        data_out["z"] = self.next_z(data_out)

        ## input actor
        # update
        data_out["u"] = self.next_u(data_out)

        # cost calculation
        p = self._system.du_h(data_in["u"]).T @ data_out["nu_h"]
        data_out["p"] = np.multiply(data_out["u"], p)

        return data_out
