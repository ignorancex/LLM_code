# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import itertools
from dataclasses import dataclass, field

import more_itertools
import numpy as np
import scqubits as scq

import qcheff.operators.operators as qops


@dataclass
class JCHModel:
    """
    A class to represent a Jaynes-Cummings-Hubbard model.

    Parameters:

    resonator_freqs : list[float]
        A list of resonator frequencies.
    detunings : list[float]
        A list of detunings between resonators and their embedded atoms.
    g : np.float64
        The coupling strength between the qubit and resonator.
    kappa : np.float64
        The coupling strength between resonators.
    nr : np.int64
        The number of levels in the resonator.
    mu : np.float64
        Chemical potential coupling to the polariton terms.

    """

    resonator_freqs: list[float]
    detunings: list[float]
    g: np.float64 = 0.5
    kappa: np.float64 = 0.0
    nr: np.int64 = 5
    mu: np.float64 = 0.0
    n: np.int64 = field(init=False)

    def __post_init__(self):
        if len(self.resonator_freqs) != len(self.detunings):
            msg = f"The number of resonator frequencies ({len(self.resonator_freqs)})and\
                  detunings ({len(self.detunings)}) must be the same."
            raise ValueError(msg)
        self.n = len(self.resonator_freqs)
        # If only one value is given, repeat it for all
        if self.n == 1:
            self.n = 2
            self.resonator_freqs = [self.resonator_freqs[0]] * self.n
            self.detunings = [self.detunings[0]] * self.n

    def jch_hamiltonian(self):
        """
        Returns the Hamiltonian of a Jaynes-Cummings-Hubbard model.
        The tensor structure is {qubits, resonators}.


        """
        # All qubits first, all resonators second
        system_dims = [2] * self.n + [self.nr] * self.n
        qubit_creation_ops = [qops.sigma_plus() for _ in range(self.n)]
        resonator_creation_ops = [qops.create() for _ in range(self.n)]

        pass

    def jch_scqubits_hilbertspace(self):
        """
        Returns the SCQubits Hilbertspace of a Jaynes-Cummings-Hubbard
        model with a single qubit and a resonator.

        scQubits currently only produces dense matrices, so this should not be used to scale.

        """
        resonators = [
            scq.Oscillator(E_osc=wr, truncated_dim=self.nr, id_str=f"r_{i}")
            for i, wr in enumerate(self.resonator_freqs)
        ]

        qubits = [
            scq.KerrOscillator(E_osc=wr, K=0, truncated_dim=2, id_str=f"q_{i}")
            for i, wr in enumerate(self.resonator_freqs)
        ]
        hs = scq.HilbertSpace(list(itertools.chain(qubits, resonators)))
        for qubit, resonator in zip(qubits, resonators, strict=True):
            hs.add_interaction(
                g_strength=self.g,
                op1=(qubit.annihilation_operator, qubit),
                op2=(resonator.creation_operator, resonator),
                add_hc=True,
            )

        for resL, resR in itertools.pairwise([*resonators, resonators[0]]):
            hs.add_interaction(
                g_strength=self.kappa,
                op1=(resL.creation_operator, resL),
                op2=(resR.annihilation_operator, resR),
                add_hc=True,
            )

        # Add the polariton terms
        hs.add_interaction(
            qobj=self.mu
            * sum(
                hs.diag_operator(range(2), hs[f"q_{i}"])  # two levels per qubit
                + hs.diag_operator(
                    range(self.nr), hs[f"r_{i}"]
                )  # nr levels per resonator
                for i in range(self.n)  # n subsystems
            )
        )

        hs.generate_lookup()
        return hs
