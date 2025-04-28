from dataclasses import dataclass, field

import numpy as np

from src.common_utils.custom_vars import Opd, Acq
from src.inverse_model.operators import LinearOperator, ProximalOperator


@dataclass(frozen=True)
class LorisVerhoevenIteration:
    # TODO:
    #  - An alternative way for post-initializing dependent attributes of a frozen dataclass is to instantiate the
    #  object from a factory function (recommended) or a class-method.
    transfer_matrix: LinearOperator
    domain_transform: LinearOperator
    prox_functional: ProximalOperator
    regularization_parameter: float
    interferogram: np.ndarray[tuple[Opd, Acq], np.dtype[np.float64]]
    is_compute_cost: bool = False

    tau: float = field(init=False)
    eta: float = field(init=False)
    rho: float = field(init=False)
    rho_tau: float = field(init=False)

    def __post_init__(self):
        """
        Post-initializing the convergence and over-relaxation parameters.
          - Originally, rho is a sequence of over-relaxation parameters, with respect to the number of iterations.
          - Here, so far, rho is considered constant for all iterations.
          - If rho=1, the algorithm is equivalent to the original Loris-Verhoeven one (i.e., without over-relaxation).
          - tau_init=1. and rho=1.9 have been set by default.
        """
        tau, eta = self.__convergence_params(tau_init=1.)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "rho", 1.9)
        object.__setattr__(self, "rho_tau", self.rho * self.tau)

    def __convergence_params(self, tau_init: float = 1.) -> tuple[float, float]:
        """
        In case the conditions ||A||<sqrt(2) and/or ||L||<1 are not satisfied, it is possible to rescale the matrices.
          - tau < 2 / transfer_matrix.norm ** 2  =>  Transfer Matrix norm rescale parameter.
          - sigma < 1 / domain_transform.norm ** 2  =>  Domain Transform norm rescale parameter.
          - eta < sigma / tau  =>  Caching the division sigma/tau.
        """
        tau = tau_init / self.transfer_matrix.norm ** 2
        eta = 0.99 / tau / self.domain_transform.norm ** 2
        return tau, eta

    def update(
            self, prim: np.ndarray, dual: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        - In the generalized Loris-Verhoeven version [Condat et al. (2023)], the derivative of h(x) is computed, which
          is the data fidelity in this case, that is, h(x)=1/2||Ax-y||^2, leading to A^T(Ax-y).
        """
        data_fidelity = self.transfer_matrix.direct(prim) - self.interferogram
        data_fidelity_derivative = self.transfer_matrix.adjoint(data_fidelity)
        prim_half = prim - self.tau * (data_fidelity_derivative + self.domain_transform.adjoint(dual))
        dual_half = self.prox_functional.proximal_conjugate(
            dual + self.eta * self.domain_transform.direct(prim_half),
            self.eta,
            self.regularization_parameter,
        )
        prim = prim - self.rho_tau * (data_fidelity_derivative + self.domain_transform.adjoint(dual_half))
        dual = dual + self.rho * (dual_half - dual)

        if self.is_compute_cost:
            fidelity_term = np.linalg.norm(x=data_fidelity.flatten(), ord=2)
            scalar_functional = self.prox_functional.direct(x=self.domain_transform.direct(prim.flatten()), lambdaa=1.)
            cost = 0.5 * fidelity_term + self.regularization_parameter * scalar_functional
        else:
            cost = 0

        return prim, dual, cost
