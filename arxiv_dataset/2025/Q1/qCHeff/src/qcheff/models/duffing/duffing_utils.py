# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field

import cupy as np
import qutip as qt
import sympy


@dataclass(frozen=True)
class DuffingOscQubit:
    omega: float
    alpha: float
    ntrunc: int = field(default=5)


@dataclass(frozen=True)
class CoupledDuffingOsc:
    qubit1: DuffingOscQubit
    qubit2: DuffingOscQubit
    g: float

    def delta(self):
        return 0.5 * (self.qubit1.omega - self.qubit2.omega + self.qubit1.alpha)

    def Delta(self):
        return (
            1.5 * (self.qubit1.omega - self.qubit2.omega)
            - self.qubit2.alpha
            + 0.5 * self.qubit1.alpha
        )

    def E2(self):
        g1 = np.sqrt(2) * self.g
        return self.delta() * (np.sqrt(1 + (g1 / self.delta()) ** 2))

    def c01(self):
        g1 = np.sqrt(2) * self.g
        return 1 / np.sqrt(((self.E2() - self.delta()) / g1) ** 2 + 1)


def create_system_NPAD(
    detuning: float,
    ntrunc: int = 3,
    alpha1: float = -0.3,  # GHz
    alpha2: float = -0.3,  # GHz
    g: float = 0.1,  # GHz
) -> CoupledDuffingOsc:
    """
    Given a detuning, creates some sensible default parameters.
    """
    w2 = 5  # GHz #probably a sensible value for a transmon
    w1 = w2 + detuning

    return CoupledDuffingOsc(
        qubit1=DuffingOscQubit(omega=w1, alpha=alpha1, ntrunc=ntrunc),
        qubit2=DuffingOscQubit(omega=w2, alpha=alpha2, ntrunc=ntrunc),
        g=g,
    )


def qutip2sympy(op: qt.Qobj):
    """
    This function converts QuTiP operators into SymPy matrices.
    """
    return sympy.Matrix(sympy.sympify(op[:]).applyfunc(sympy.nsimplify))
