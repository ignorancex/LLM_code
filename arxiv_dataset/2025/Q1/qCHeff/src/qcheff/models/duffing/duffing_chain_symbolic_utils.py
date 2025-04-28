# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import qutip as qt
import sympy

from qcheff.duffing.duffing_chain_utils import UniformlyCoupledDuffingChain
from qcheff.duffing.duffing_utils import qutip2sympy


def duffing_chain_full_ham(
    example: UniformlyCoupledDuffingChain,
    return_subs: bool = False,
):
    """
    Hamiltonian in full Hilbert space for a Duffing Oscillator chain.

    example: DuffingChain
    The example system

    return_subs: bool, default False
    Whether to return the substitution params.

    """
    freqs_num = [qubit.omega for qubit in example.qubits]
    alphas_num = [qubit.alpha for qubit in example.qubits]
    g_num = example.g

    ntrunc = example.ntrunc
    nsystems = len(example.qubits)
    freqs_sym = sympy.symbols(f"omega1:{nsystems+1}", positive=True)
    alphas_sym = sympy.symbols(f"alpha1:{nsystems+1}", real=True)
    g = sympy.symbols("g", positive=True)

    param_sub_vals = (
        dict(zip(freqs_sym, freqs_num, strict=False))
        | dict(zip(alphas_sym, alphas_num, strict=False))
        | {g: g_num}
    )
    # annihilation operators
    b_ops = [
        qutip2sympy(
            embed_operator(
                op=qt.destroy(ntrunc), pos=idx, ntrunc=ntrunc, nsystems=nsystems
            )
        )
        for idx in range(nsystems)
    ]
    # number operators
    num_ops = [b.adjoint() @ b for b in b_ops]
    # Kerr terms
    kerr_ops = [(b.adjoint()) ** 2 @ (b) ** 2 for b in b_ops]

    # Self energies of the qubits
    H0 = sympy.Add(
        *(
            (w * N + (a * K) / 2)
            for w, a, N, K in zip(
                freqs_sym, alphas_sym, num_ops, kerr_ops, strict=False
            )
        )
    )
    Hc = sympy.Add(
        *(
            g * (b1 @ b2.adjoint() + b1.adjoint() @ b2)
            for b1, b2 in zip(b_ops, b_ops[1:], strict=False)
        )
    )
    H = H0 + Hc
    return (H, param_sub_vals) if return_subs else H
