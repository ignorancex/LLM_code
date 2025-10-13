import warnings

import numpy as np

from feedback_opt.optimizers.optimizer_base import OptimizerGradientStep
from feedback_opt.utils import Polytope


class OptimizerPrimal(OptimizerGradientStep):
    """
    this algorithm replaces all occurences of y with the ss-map y = h(u)\
    iterates are projected onto the linearized constraints

    min     phi_u(u) + phi(h(u))
    s.t.    c_u(u)    <= 0
            c_y(h(u)) <= 0
    """

    def primal_step(self, data_in: dict) -> dict:
        assert np.shape(data_in["u"]) == (self._system.m, 1)
        assert np.shape(data_in["y"]) == (self._system.p, 1)

        data_out = {}

        # unconstrained gradient descent
        step = self.du_phi_u(data_in["u"]) + self._system.du_h(data_in["u"]).T @ self.dy_phi_y(
            data_in["y"]
        )
        u_hat = data_in["u"] - self.alpha * step

        # linearized output constraints Y_u
        A_y_u = self._system.Y.A @ self._system.du_h(data_in["u"])
        b_y_u = (
            self._system.Y.A @ (self._system.du_h(data_in["u"]) @ data_in["u"] - data_in["y"])
            + self._system.Y.b
        )
        Y_u = Polytope(A_y_u, b_y_u, self._system.m)

        # complete input constraints
        constraint_u = self._system.U.intersect_with(Y_u)

        # project input to constraints
        u_proj = constraint_u.proj_2(u_hat)
        if u_proj is not None:
            data_out["u"] = u_proj
        else:
            warnings.warn("ignoring y constraint linearization")
            data_out["u"] = self._system.U.proj_2(u_hat)

        assert data_out["u"] is not None
        return data_out

    def data_step(self, data_in: dict) -> dict:
        # assert minimum input requirements
        assert "u" in data_in
        assert "y" in data_in

        # primal step
        data_out = self.primal_step(data_in)

        # NO ACTOR OR PRICE INTERPRETATION!

        return data_out
