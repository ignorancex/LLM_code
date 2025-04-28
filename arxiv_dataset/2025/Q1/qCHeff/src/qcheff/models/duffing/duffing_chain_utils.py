# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import qutip as qt
import scqubits as scq

import qcheff.operators as qcops
from qcheff.duffing.duffing_utils import DuffingOscQubit
from qcheff.operators import embed_operator


@dataclass(frozen=True)
class DuffingChain:
    """
    Implements a linear chain of N Duffing Oscillator qubits.
    This simple example assumes the qubits are all coupled to their nearest neighbors
    with the different coupling strenghs g.

    """

    qubits: list[DuffingOscQubit]
    couplings: list[float]
    ntrunc: int


def create_duffing_chain_system(
    omegas,
    alphas,
    couplings,
    ntrunc: int = 3,
):
    """
    Factory function for dduffing chain
    """
    if len(omegas) != len(alphas) or len(omegas) != len(couplings) + 1:
        raise ValueError(
            "The number of qubit freqs should be equal to the number of anharmonicities and one less than the number of couplings"
        )
    DuffingQubits_list = [
        DuffingOscQubit(omega=w, alpha=a, ntrunc=ntrunc)
        for w, a in zip(omegas, alphas, strict=True)
    ]

    return DuffingChain(qubits=DuffingQubits_list, couplings=couplings, ntrunc=ntrunc)


def create_linear_spectrum_zz_chain(
    delta1: float,
    delta2: float,
    alpha1: float = -0.3,  # GHz
    alpha2: float = -0.3,  # GHz
    num_resonators: int = 1,
    omega_res: float = 5,  # GHz
    delta_omega_res: float = 0.6,  # GHz
    g: float = 0.1,  # GHz
    ntrunc: int = 3,
    debug: bool = False,
) -> DuffingChain:
    """
    Creates a linear chain of Duffing Oscillators. The ends have anharmonicity alpha, modeling transmons.
    The remaining are linear resonators, with their frequencies distributed in a V shape.

    """
    if num_resonators % 2 == 0 or num_resonators < 1:
        raise ValueError("The chain size should be a positive odd number.")

    n = (num_resonators - 1) // 2

    res_deltas = np.abs(np.linspace(-n, n, num_resonators)) * delta_omega_res
    all_omegas = (
        [omega_res + delta1] + list(res_deltas + omega_res) + [omega_res + delta2]
    )
    all_alphas = [alpha1] + [0] * num_resonators + [alpha2]
    all_couplings = [g] * 2 * (n + 1)

    if debug:
        print(f"omegas: {all_omegas}\nalphas: {all_alphas}\ncouplings: {all_couplings}")
    return create_duffing_chain_system(
        all_omegas, all_alphas, all_couplings, ntrunc=ntrunc
    )


def duffing_chain_num_ham(
    example: DuffingChain,
):
    """
    Numerical Hamiltonian in full Hilbert space for a Duffing Oscillator chain.

    example: DuffingChain
    The example system
    """
    freqs_num = [qubit.omega for qubit in example.qubits]
    alphas_num = [qubit.alpha for qubit in example.qubits]
    g_num = example.couplings

    if not isinstance(example, DuffingChain):
        msg = "example must be a DuffingChain"
        raise ValueError(msg)

    ntrunc = example.ntrunc
    nsystems = len(example.qubits)

    # annihilation operators
    b_ops = [
        embed_operator(
            op=qcops.destroy(ntrunc), pos=idx, ntrunc=ntrunc, nsystems=nsystems
        )
        for idx in range(nsystems)
    ]
    # number operators
    num_ops = [b.conj().T @ b for b in b_ops]
    # Kerr terms
    kerr_ops = [(b.conj().T) ** 2 @ (b) ** 2 for b in b_ops]

    # Self energies of the qubits
    H0 = sum(
        (
            (w * N + (a * K) / 2)
            for w, a, N, K in zip(
                freqs_num, alphas_num, num_ops, kerr_ops, strict=False
            )
        )
    )
    Hc = sum(
        (
            coupl * (b1 @ b2.conj().T + b1.conj().T @ b2)
            for coupl, b1, b2 in zip(g_num, b_ops, b_ops[1:], strict=False)
        )
    )
    H = H0 + Hc
    return H


def duffing_chain_qutip_ham(
    example: DuffingChain,
):
    """
    Numerical Hamiltonian in full Hilbert space for a Duffing Oscillator chain.

    example: DuffingChain
    The example system
    """
    freqs_num = [qubit.omega for qubit in example.qubits]
    alphas_num = [qubit.alpha for qubit in example.qubits]
    g_num = example.couplings

    if not isinstance(example, DuffingChain):
        msg = "example must be a DuffingChain"
        raise ValueError(msg)

    ntrunc = example.ntrunc
    nsystems = len(example.qubits)

    # annihilation operators
    b_ops = [
        qt.tensor(
            [qt.destroy(ntrunc) if i == j else qt.qeye(ntrunc) for i in range(nsystems)]
        )
        for j in range(nsystems)
    ]

    # number operators
    num_ops = [b.dag() @ b for b in b_ops]
    # Kerr terms
    kerr_ops = [(b.dag()) ** 2 @ (b) ** 2 for b in b_ops]

    # Self energies of the qubits
    H0 = sum(
        (
            (w * N + (a * K) / 2)
            for w, a, N, K in zip(
                freqs_num, alphas_num, num_ops, kerr_ops, strict=False
            )
        )
    )
    Hc = sum(
        (
            coupl * (b1 @ b2.dag() + b1.dag() @ b2)
            for coupl, b1, b2 in zip(g_num, b_ops, b_ops[1:], strict=False)
        )
    )
    H = H0 + Hc
    return H


def duffing_chain_scq_hilbertspace(
    example: DuffingChain,
):
    """
    SCQubits Hamiltonian for a Duffing Oscillator chain.

    Since SCQubits comes with a Kerr oscillator class, we can use it to define the qubits.
    However, the Kerr oscillator Hamiltonina has different constants, so we need to adjust the sign of the anharmonicity
    and divide by 2.

    """
    freqs_num = [qubit.omega for qubit in example.qubits]
    alphas_num = [qubit.alpha for qubit in example.qubits]
    # If the chain is uniform, then all the coupling strengths are the same
    if isinstance(example, DuffingChain):
        g_num = example.couplings
    else:
        msg = "example must be a DuffingChain"
        raise ValueError(msg)

    tmon_array = [
        scq.KerrOscillator(
            E_osc=omega,
            K=-alpha
            / 2,  # The minus sign is because of the definition of the Hamiltonian
            truncated_dim=example.ntrunc,
            id_str=f"tmon_{idx}",
        )
        for idx, (omega, alpha) in enumerate(zip(freqs_num, alphas_num, strict=False))
    ]
    hs = scq.HilbertSpace(tmon_array)
    for (left_tmon, right_tmon), g in zip(pairwise(tmon_array), g_num, strict=False):
        left_a = left_tmon.creation_operator()
        right_adag = right_tmon.annihilation_operator()
        hs.add_interaction(
            g=g,
            op1=(left_a, left_tmon),
            op2=(right_adag, right_tmon),
            add_hc=True,
            id_str=f"g_{left_tmon.id_str[-1]}{right_tmon.id_str[-1]}",
        )
    hs.generate_lookup()
    return hs
