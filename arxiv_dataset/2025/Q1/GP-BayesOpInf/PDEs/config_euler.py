# config_heatcubiclifted.py
"""Configuration for Euler equations in specific volume variables.

This experiment reduces the lifted variables (u, q, 1/rho) jointly and learns
a ROM with the quadratic structure dq/dt = H[q x q].
"""

__all__ = [
    # Simulation specifics
    "spatial_domain",
    "time_domain",
    "init_params",
    "initial_conditions",
    # Simulation classes
    "FullOrderModel",
    "Basis",
    "ReducedOrderModel",
    # GP kernel fitting hyperparameters
    "CONSTANT_VALUE_BOUNDS",
    "LENGTH_SCALE_BOUNDS",
    "NOISE_LEVEL_BOUNDS",
    "N_RESTARTS_OPTIMIZER",
]

import numpy as np

import opinf

import pde_models as pdes


# Simulation specifications  --------------------------------------------------
spatial_domain = np.linspace(0, 2, 201)[:-1]  # Spatial domain x.
time_domain = np.linspace(0, 0.15, 401)  # Temporal domain t.
init_params = [22, 20, 24, 95, 105, 100]
initial_conditions = pdes.Euler(spatial_domain).initial_conditions(
    init_params=init_params,
    plot=False,
)  # Initial conditions q(x, 0).


# Simulation classes ----------------------------------------------------------
class FullOrderModel(pdes.Euler):
    """Full-order model for this problem."""

    def __init__(self):
        super().__init__(spatial_domain)


class Basis(opinf.basis.PODBasis):
    """Basis for this problem: POD, treating all three variables jointly,
    with some nondimensionaliziation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _v, _rho = 100, 10
        self.__scalers = np.array([_v, _rho * _v**2, 1 / _rho])

    @property
    def scalers(self):
        return self.__scalers

    def nondimensionalize(self, states):
        return np.concatenate(
            [var / s for var, s in zip(np.split(states, 3), self.scalers)]
        )

    def redimensionalize(self, states):
        return np.concatenate(
            [var * s for var, s in zip(np.split(states, 3), self.scalers)]
        )

    def fit(self, states):
        states, self.shift_ = opinf.pre.shift(states)
        return super().fit(self.nondimensionalize(states))

    def compress(self, states):
        states = opinf.pre.shift(states, shift_by=self.shift_)
        return super().compress(self.nondimensionalize(states))

    def decompress(self, states):
        states = self.redimensionalize(super().decompress(states))
        return opinf.pre.shift(states, shift_by=-self.shift_)


class ReducedOrderModel(opinf.models.ContinuousModel):
    """Reduced-order model for this problem."""

    ivp_method = "RK45"
    input_dimension = 0

    def __init__(self):
        super().__init__("cAH")

    input_func = None


# Gaussian process kernel fitting hyperparameters -----------------------------
CONSTANT_VALUE_BOUNDS = (1e-5, 1e5)
LENGTH_SCALE_BOUNDS = (1e-5, 1e2)
NOISE_LEVEL_BOUNDS = (1e-16, 1e2)
N_RESTARTS_OPTIMIZER = 100
