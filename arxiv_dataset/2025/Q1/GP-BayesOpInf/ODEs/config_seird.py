# config_seird.py
"""Configuration for SEIRD system experiments."""

__all__ = [
    "time_domain",
    "test_initial_conditions",
    "Model",
]

import numpy as np

import ode_models as odes


time_domain = np.linspace(0, 200, 500)
true_parameters = np.array([1.0, 0.25, 0.1, 0.1, 0.05, 0.05])
initial_conditions = np.array([0.994, 0.005, 0.001, 0, 0])
test_initial_conditions = np.array([0.722, 0.208, 0.070, 0, 0])


class Model(odes.SEIRD2):
    num_equations = 1

    def __init__(self):
        """Set the system parameters."""
        super().__init__(super().convert_parameters(true_parameters))

    @staticmethod
    def data_matrix(states: np.ndarray) -> np.ndarray:
        """Construct the 5k x 4 data matrix for the single coupled problem."""
        S, E, I, _, _ = states
        SI = S * I
        Z = np.zeros_like(S)

        data_dSdt = np.column_stack((-SI, Z, Z, Z))
        data_dEdt = np.column_stack((SI, -E, Z, Z))
        data_dIdt = np.column_stack((Z, E, -I, -I))
        data_dRdt = np.column_stack((Z, Z, I, Z))
        data_dDdt = np.column_stack((Z, Z, Z, I))

        return np.vstack(
            [data_dSdt, data_dEdt, data_dIdt, data_dRdt, data_dDdt]
        )
