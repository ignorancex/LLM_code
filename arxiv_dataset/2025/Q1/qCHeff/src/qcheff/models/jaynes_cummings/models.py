# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Jaynes-Cummings self with a single qubit and a resonator,
# implemented in SCQubits, QuTiP and quCHeff.
from dataclasses import dataclass

import numpy as np
import scqubits as scq


@dataclass
class JCModel:
    """
    Everything is measured in units of the coupling constant g.
    """

    resonator_freq: np.float64 = 5
    detuning: np.float64 = 0.1
    resonator_levels: np.int64 = 5

    def jc_scqubits_hilbertspace(
        self,
    ):
        """
        Returns the SCQubits Hilbertspace of a Jaynes-Cummings model
        with a single qubit and a resonator.

        """
        resonator = scq.Oscillator(
            E_osc=self.resonator_freq,
            truncated_dim=self.resonator_levels,
            id_str="r",
        )
        qubit = scq.KerrOscillator(
            E_osc=self.resonator_freq - self.detuning,
            K=0,
            truncated_dim=2,
            id_str="q",
        )
        hs = scq.HilbertSpace([qubit, resonator], evals_method="evals_scipy_sparse")
        hs.add_interaction(
            g_strength=1.0,
            op1=(qubit.annihilation_operator, qubit),
            op2=(resonator.creation_operator, resonator),
            add_hc=True,
        )
        hs.generate_lookup()
        return hs

    def critical_chemical_potential(self, delta, n):
        return np.sqrt(n + delta * delta / 4) - np.sqrt(n + 1 + delta * delta / 4)
