from abc import ABC, abstractmethod

import numpy as np

from feedback_opt.systems.system_base import SystemBase


class OptimizerBase(ABC):
    """
    The class is intended to be immutable after initialization!
    """

    def __init__(
        self,
        params,
        system: SystemBase,
    ):
        # optimizer name
        if hasattr(params, "name"):
            assert isinstance(params.name, str)
            self.name = params.name
        else:
            self.name = self.__class__.__name__

        # system
        assert isinstance(system, SystemBase)
        self._system = system

        # input objective cost function
        if hasattr(params, "quad_u"):
            assert np.shape(params.quad_u) == (system.m, system.m)
            assert np.all(np.linalg.eigvals(params.quad_u) >= 0)
            assert np.all(params.quad_u == params.quad_u.T)
            self.quad_u = params.quad_u
        else:
            self.quad_u = np.zeros((system.m, system.m))
        if hasattr(params, "lin_u"):
            assert np.shape(params.lin_u) == (system.m, 1)
            self.lin_u = params.lin_u
        else:
            self.lin_u = np.zeros((system.m, 1))

        # output objective cost function
        if hasattr(params, "quad_y"):
            assert np.shape(params.quad_y) == (system.p, system.p)
            assert np.all(np.linalg.eigvals(params.quad_y) >= 0)
            assert np.all(params.quad_y == params.quad_y.T)
            self.quad_y = params.quad_y
        else:
            self.quad_y = np.zeros((system.p, system.p))
        if hasattr(params, "lin_y"):
            assert np.shape(params.lin_y) == (system.p, 1)
            self.lin_y = params.lin_y
        else:
            self.lin_y = np.zeros((system.p, 1))

    def phi_u(self, u: np.ndarray) -> float:
        """
        input cost function phi_u(u)

        :param np.array u: input vector (m,1)
        :return np.array: cost (1,1)
        """
        assert np.shape(u) == (self._system.m, 1)
        return u.T @ self.quad_u @ u + self.lin_u.T @ u

    def phi_y(self, y: np.ndarray) -> float:
        """
        output cost function phi_y(y)

        :param np.array y: output vector (p,1)
        :return np.array: cost (1,1)
        """
        assert np.shape(y) == (self._system.p, 1)
        return y.T @ self.quad_y @ y + self.lin_y.T @ y

    def phi(self, u: np.ndarray, y: np.ndarray) -> float:
        """
        complete optimization cost function phi(u, y)

        :param np.array u: input vector (m,1)
        :param np.array y: output vector (p,1)
        :return np.array: cost (1,1)
        """
        assert np.shape(u) == (self._system.m, 1)
        assert np.shape(y) == (self._system.p, 1)
        return self.phi_u(u) + self.phi_y(y)

    def du_phi_u(self, u: np.ndarray) -> np.ndarray:
        """
        input cost sensitivity function du_phi_u(u)

        :param np.array u: input vector (m,1)
        :return np.array: gradient (m,1)
        """
        assert np.shape(u) == (self._system.m, 1)
        return 2 * self.quad_u @ u + self.lin_u

    def dy_phi_y(self, y: np.ndarray) -> np.ndarray:
        """
        output cost sensitivity function dy_phi_y(y)

        :param np.array y: output vector (p,1)
        :return np.array: gradient (p,1)
        """
        assert np.shape(y) == (self._system.p, 1)
        return 2 * self.quad_y @ y + self.lin_y

    def data_initial(self, u_0: np.ndarray = None) -> dict:
        """
        packages the first data dict to be used inducetively

        :return dict: data_k0 data at timestep k=0
        """
        if u_0 is None:
            u_0 = np.zeros((self._system.m, 1))
        assert np.shape(u_0) == (self._system.m, 1)

        data_out = {}
        data_out["u"] = u_0
        data_out["y"] = self._system.h(u_0)
        data_out = self.data_cost(data_out)
        data_out = self.data_y_violation(data_out)

        return data_out

    def data_cost(self, data_in: dict) -> dict:
        # assert minimum input requirements
        assert "u" in data_in
        assert "y" in data_in

        data_out = data_in
        data_out["phi"] = self.phi(data_in["u"], data_in["y"])
        return data_out

    def data_y_violation(self, data_in: dict) -> dict:
        # assert minimum input requirements
        assert "y" in data_in

        data_out = data_in
        z = self._system.Y.proj_2(data_in["y"])
        data_out["y_violation"] = np.linalg.norm(z - data_in["y"], keepdims=True)
        return data_out

    @abstractmethod
    def data_step(self, data_in: dict) -> dict:
        """
        performs a single update step

        :param dict data_in: data at timestep k
        :return dict: data_out data at timestep k+1
        """
        raise NotImplementedError


class OptimizerGradientStep(OptimizerBase):
    def __init__(
        self,
        params,
        system: SystemBase,
    ):
        super().__init__(params=params, system=system)

        # primal learning rate
        assert hasattr(params, "alpha"), (
            "primal learning rate alpha must be defined within the optimizer parameters!"
        )
        assert isinstance(params.alpha, (int, float))
        self.alpha = float(params.alpha)

        # dual learning rate
        if hasattr(params, "beta"):
            assert isinstance(params.beta, (int, float))
            self.beta = float(params.beta)
        else:
            self.beta = self.alpha


class OptimizerProximal(OptimizerBase):
    def __init__(
        self,
        params,
        system: SystemBase,
    ):
        super().__init__(params=params, system=system)

        # dual learning rate
        assert hasattr(params, "rho"), (
            "dual learning rate rho must be defined within the optimizer parameters!"
        )
        assert isinstance(params.rho, (int, float))
        self.rho = float(params.rho)

        # proximal u deviation cost
        if hasattr(params, "gamma_u"):
            assert isinstance(params.gamma_u, (int, float))
            self.gamma_u = float(params.gamma_u)
        else:
            self.gamma_u = 0.0
