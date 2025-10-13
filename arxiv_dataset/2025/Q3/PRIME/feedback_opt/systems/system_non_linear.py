import types

import numpy as np

from feedback_opt.systems.system_base import SystemBase


class SystemNonLinear(SystemBase):
    """
    unconstrained system defined by io map:
    y = h(u)

    The class is intended to be immutable after initialization!
    """

    def __init__(self, params):
        super().__init__(params)

        assert hasattr(params, "h"), (
            "steady state map y = h(u) must be defined within the system parameters!"
        )
        assert isinstance(params.h, types.LambdaType)
        self._h = params.h

        assert hasattr(params, "du_h"), (
            "steady state map sensitivity nabla_u h(u) must be defined within the system parameters!"
        )
        assert isinstance(params.du_h, types.LambdaType)
        self._du_h = params.du_h

    def h(self, u: np.ndarray) -> np.ndarray:
        assert np.shape(u) == (self.m, 1)

        y = self._h(u)

        assert np.shape(y) == (self.p, 1)
        return y

    def du_h(self, u: np.ndarray) -> np.ndarray:
        assert np.shape(u) == (self.m, 1)

        sensitivity = self._du_h(u)

        assert np.shape(sensitivity) == (self.p, self.m)
        return sensitivity
