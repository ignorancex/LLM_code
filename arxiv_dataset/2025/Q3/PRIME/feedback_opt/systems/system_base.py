from abc import ABC, abstractmethod

import numpy as np

from feedback_opt.utils import Polytope


class SystemBase(ABC):
    """
    unconstrained system defined by io map:
    y = h(u)

    The class is intended to be immutable after initialization!
    """

    def __init__(self, params):
        # system dimensions
        assert isinstance(params.m, int)
        self.m = params.m  # input dimension
        assert isinstance(params.p, int)
        self.p = params.p  # output dimension

        # system constraints
        if hasattr(params, "A_u") and hasattr(params, "b_u"):
            self.U = Polytope(params.A_u, params.b_u, params.m)
        else:
            self.U = Polytope.full_space(params.m)

        if hasattr(params, "A_y") and hasattr(params, "b_y"):
            self.Y = Polytope(params.A_y, params.b_y, params.p)
        else:
            self.Y = Polytope.full_space(params.p)

    ### steady state map
    @abstractmethod
    def h(self, u: np.ndarray) -> np.ndarray:
        """
        steady state map y = h(u) of unconstrained system
        (online measurement of system after reaching equilibrium)

        :param np.array u: input vector (m,1)
        :return np.array: steady-state output vector (p,1)
        """
        raise NotImplementedError

    @abstractmethod
    def du_h(self, u: np.ndarray) -> np.ndarray:
        """
        steady state map Jacobian nabla h(u) of unconstrained system

        :param np.array u: input vector (m,1)
        :return np.array: steady-state map sensitivity (p,m)
        """
        raise NotImplementedError
