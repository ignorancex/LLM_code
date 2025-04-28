# config_heatcubiclifted.py
"""Configuration for cubic heat equation experiments with quadratic lifting.

This experiment reduces the lifted variables (q, q^2) jointly and learns a ROM
with the quadratic structure dq/dt = c + Aq + H[q x q] + B[u] + N[u x q].
"""

__all__ = [
    # Simulation specifics
    "spatial_domain",
    "time_domain",
    "initial_conditions",
    "input_parameters",
    # Simulation classes
    "FullOrderModel",
    "Basis",
    "ReducedOrderModel",
    "test_parameters",
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
spatial_domain = np.linspace(0, 1, 500)  # Spatial domain x.
time_domain = np.linspace(0, 2, 500)  # Temporal domain t.
left_bc = 0  # q(x[0], t) = left_bc.
right_bc = 1  # q(x[-1], t) = right_bc.
diffusion = 1e-2  # Diffusion constant kappa.
initial_conditions = pdes.HeatBimodal.initial_conditions(
    spatial_domain, left_bc, right_bc
)  # q(x, 0).

input_parameters = (
    (-2, 0),
    (-1, -2),
    (0, 1),
    (1, -1),
    (2, 2),
)
test_parameters = (1.5, 0.5)


# Simulation classes ----------------------------------------------------------
class FullOrderModel(pdes.CubicHeatBimodal):
    """Full-order model for this problem."""

    def __init__(self, params):
        a, b = params
        super().__init__(
            spatial_domain,
            left_bc,
            right_bc,
            diffusion=diffusion,
            a=a,
            b=b,
        )


class Basis(opinf.basis.PODBasis):
    """Basis for states of the form (q, q^2).
    A single POD basis is used for the joint state.
    """

    def fit(self, states):
        """Construct the bases."""
        states = np.concatenate((states, states**2))
        states, self.shift_ = opinf.pre.shift(states)
        return super().fit(states)

    def compress(self, states):
        """Map high-dimensional states to low-dimensional coordinates."""
        states = np.concatenate((states, states**2))
        states = opinf.pre.shift(states, shift_by=self.shift_)
        return super().compress(states)

    def decompress(self, states_compressed):
        """Map low-dimensional coordinates to high-dimensional states."""
        states = super().decompress(states_compressed)
        states = opinf.pre.shift(states, shift_by=-self.shift_)
        return np.split(states, 2, axis=0)[0]


class ReducedOrderModel(opinf.models.ContinuousModel):
    """Reduced-order model for this problem."""

    ivp_method = "BDF"
    input_dimension = 2

    def __init__(self):
        super().__init__("cAHBN")


def input_func_factory(params):
    """Create a function handle to the input function u(t) for a
    given set of input parameters.
    """
    a, b = params

    def input_func(t):
        """Left Neumann BC with the given input parameters."""
        return FullOrderModel.oscillators(t, a, b)

    return input_func


# Gaussian process kernel fitting hyperparameters -----------------------------
CONSTANT_VALUE_BOUNDS = (1e-5, 1e5)
LENGTH_SCALE_BOUNDS = (1e-5, 1e2)
NOISE_LEVEL_BOUNDS = (1e-16, 1e2)
N_RESTARTS_OPTIMIZER = 100
