# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import functools
import pprint
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import cupy as cp
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import qutip as qt

from qcheff.utils.pulses import FourierPulse
from qcheff.utils.system import QuTiPSystem


@dataclass(frozen=True)
class DegenerateSpinChain:
    """
    Describes a degenerate spin chain of size N. All qubits have the same
    frequency = 1 with nearest ne ighbor coupling strength J (in units of
    frequency) and next-nearest neighbor coupling strength g (in units of
    frequency).
    """

    chain_size: int  # N
    nearest_couping: float  # J
    next_nearest_couping: float  # g


def create_degen_chain(N: int = 3, J: float = 5e-2, g: float = 5e-3):
    """
    Creates a spin chain with given parameters.

    N: int, default 3
    size of chain

    J: float, default 1/20
    NN coupling strength

    g: float, default 1/200
    NNN coupling strength. The default value is 1/10 the
    default value for J.
    """
    return DegenerateSpinChain(chain_size=N, nearest_couping=J, next_nearest_couping=g)


def embed_operator(op: qt.Qobj, pos: int, ntrunc: int, nsystems: int):
    """
    Identity wrapping for the appropriate operator

    op: qt.Qobj
    The operator to be wrapped. Ideally, it should have dimensions=ntrunc

    pos: int
    The position of the operator

    ntrunc: int
    The truncation level of all operators

    nsystems: int
    The total number of systems
    Should be larger than pos.

    """

    # Fully tensored identity in the Full Hilbert Space.
    wrapped_op = [op if idx == pos else qt.qeye(ntrunc) for idx in range(nsystems)]
    return qt.tensor(wrapped_op)


def QuTiP_level_labels(chain: DegenerateSpinChain):
    """
    Returns strings showing labels for each level in the
    preferred ordering in QuTiP.
    """
    N = chain.chain_size
    chain_dims = [2] * N
    labels = [
        ("{}," * N).format(*state)[:-1]
        for state in qt.state_number_enumerate(chain_dims)
    ]
    return labels


def QuTiP_drift_ham_degen_chain(chain: DegenerateSpinChain):
    """
    Creates a QuTiP drift Hamiltonian given a chain.
    Energies are measured in units of omega.
    """
    N = chain.chain_size  # number of spins
    J = chain.nearest_couping
    g = chain.next_nearest_couping
    # Wrapper for Pauli matrices
    embed_pauli = partial(embed_operator, ntrunc=2, nsystems=N)
    [embed_pauli(op=qt.sigmax(), pos=idx) for idx in range(N)]
    [embed_pauli(op=qt.sigmay(), pos=idx) for idx in range(N)]
    sz_ops = [embed_pauli(op=qt.sigmaz(), pos=idx) for idx in range(N)]

    Hself = 0.5 * (sum(sz_ops))
    Hneigbors = sum(
        -J * sz_ops[j % N] @ sz_ops[(j + 1) % N]
        - g * sz_ops[j % N] @ sz_ops[(j + 2) % N]
        for j in range(N)
    )
    return (Hself, Hneigbors)


def QuTiP_chain_prodstate(chain_size: int, state_label: tuple[int, ...]):
    """
    Returns an initial state of the given label for a product state.
    """
    return qt.basis(dims=[2] * chain_size, n=state_label)


def QuTiP_chain_prodstate_index(chain_size: int, state_index: int):
    """
    Returns an initial state of the given label for a product state.
    """
    return qt.basis(
        dims=[2] * chain_size,
        n=list(qt.state_index_number(dims=[2] * chain_size, index=state_index)),
    )


def QuTiP_total_ham_degen_chain(
    chain: DegenerateSpinChain,
    tlist: np.ndarray,
    xdrive: np.ndarray,
    ydrive: np.ndarray,
):
    """
    Creates a QuTiP total driven Hamiltonian given a chain.
    Energies are measured in units of omega.
    Times are measured in units of pi/omega.

    Two drives must be provided as arrays. For simplicity, we assume
    the maximum drive strength is 1 (in units of omega).
    """
    N = chain.chain_size  # number of spins
    J = chain.nearest_couping
    g = chain.next_nearest_couping
    # Wrapper for Pauli matrices
    embed_pauli = partial(embed_operator, ntrunc=2, nsystems=N)
    sx_ops = [embed_pauli(op=qt.sigmax(), pos=idx) for idx in range(N)]
    sy_ops = [embed_pauli(op=qt.sigmay(), pos=idx) for idx in range(N)]
    sz_ops = [embed_pauli(op=qt.sigmaz(), pos=idx) for idx in range(N)]

    Hself = 0.5 * (sum(sz_ops))
    # Make sure I'm not double counting here.
    Hneigbors = sum(
        -J * sz_ops[j % N] @ sz_ops[(j + 1) % N]
        - g * sz_ops[j % N] @ sz_ops[(j + 2) % N]
        for j in range(N)
    )

    Hdrive = qt.QobjEvo([[sum(sx_ops), xdrive], [sum(sy_ops), ydrive]], tlist=tlist)

    return (Hself, Hneigbors, Hdrive)


def simulate_chain_dynamics(
    chain: DegenerateSpinChain,
    tlist: np.ndarray,
    Hsim: qt.Qobj | qt.QobjEvo,
    psi0: qt.Qobj,
    plot=None,
):
    """
    Simulates the dynamics of the chain and plots Pauli operator expectation values
    """

    result = qt.sesolve(
        Hsim, psi0, tlist, options={"progress_bar": False, "method": "vern9"}
    )
    # Convert states to density matrices
    states = [s * s.dag() for s in result.states]

    if plot == "pauli":
        embed_pauli = partial(embed_operator, ntrunc=2, nsystems=chain.chain_size)
        sx_eops = [
            embed_pauli(op=qt.sigmax(), pos=idx) for idx in range(chain.chain_size)
        ]
        sy_eops = [
            embed_pauli(op=qt.sigmay(), pos=idx) for idx in range(chain.chain_size)
        ]
        sz_eops = [
            embed_pauli(op=qt.sigmaz(), pos=idx) for idx in range(chain.chain_size)
        ]
        # Expectation value
        exp_sx = np.array(qt.expect(states, sx_eops))
        exp_sy = np.array(qt.expect(states, sy_eops))
        exp_sz = np.array(qt.expect(states, sz_eops))
        # Plot the expecation value
        fig3, ax3 = plt.subplots(1, 3, layout="constrained", figsize=(20, 5))
        for axis, pauli_label, pauli_exp in zip(
            ax3, ["x", "y", "z"], [exp_sx, exp_sy, exp_sz], strict=False
        ):
            ymax = np.max(np.abs(pauli_exp))
            axis.plot(
                tlist,
                pauli_exp,
                label=[
                    rf"$\langle \sigma^{pauli_label}_{i} \rangle$"
                    for i in range(chain.chain_size)
                ],
                alpha=0.3,
                # marker="o",
                # markevery=20,
            )
            axis.legend(loc="lower right")
            axis.set(
                xlabel="Time",
                ylabel=rf"$\langle \sigma_{pauli_label} \rangle$",
                ylim=(-1 * ymax, 1 * ymax),
                xlim=(tlist.min(), tlist.max()),
            )
            fig3.suptitle("Dynamics of Spin Chain")
    elif plot == "pops":
        # need to implement
        pass
        # fig3, ax3 = plt.subplots(1, 1, layout="constrained", figsize=(10, 5))


# Here is a function that will set the system up for us.
def setup_magnus_chain_example(
    pulse_coeffs: Sequence[float],
    **kwargs,
):
    g = kwargs.get("g", 5e-3)
    J = kwargs.get("J", 5e-2)
    x_pulse_coeffs, y_pulse_coeffs = np.array_split(
        np.asarray(pulse_coeffs),
        indices_or_sections=2,
    )
    gate_time = kwargs.get("gate_time", 25)
    chain_size = kwargs.get("chain_size", 3)
    num_tlist = kwargs.get("num_tlist", 10**3)
    pulse_freq = kwargs.get("pulse_freq", 2 * np.pi * 5)
    max_amp = kwargs.get("pulse_amp", 5e-2)
    _device = kwargs.get("device", "cpu")
    _sparse = kwargs.get("sparse", False)

    _debug = kwargs.get("debug", False)

    if _debug:
        pprint.pprint(kwargs)  # noqa: T203

    control_pulses = [
        FourierPulse(
            pulse_coeff,
            gate_time=gate_time,
            frequency=pulse_freq,
            amplitude=max_amp,
            name=f"{drive_name}_drive",
        )
        for pulse_coeff, drive_name in zip(
            [x_pulse_coeffs, y_pulse_coeffs],
            ["X", "y"],
            strict=True,
        )
    ]
    test_tlist = np.linspace(0, gate_time, num_tlist)

    embed_pauli = functools.partial(embed_operator, ntrunc=2, nsystems=chain_size)
    sx_ops, sy_ops, sz_ops = (
        [embed_pauli(op=sigma_op, pos=idx) for idx in range(chain_size)]
        for sigma_op in [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    )

    # The chain has periodic boundary conditions
    H_nn = sum(
        -J * sz_ops[j % chain_size] @ sz_ops[(j + 1) % chain_size]
        for j in range(chain_size)
    )
    H_nnn = sum(
        -g * sz_ops[j % chain_size] @ sz_ops[(j + 2) % chain_size]
        for j in range(chain_size)
    )
    Hneigbors = H_nn + H_nnn
    # Create the drift Hamiltonian
    Hdrift =  Hneigbors
    # Create the control Hamiltonian
    Hcontrol = [sum(sx_ops), sum(sy_ops)]

    test_system = QuTiPSystem(Hdrift, control_pulses, Hcontrol)
    # yield test_system

    test_magnus = test_system.get_magnus_system(
        tlist=test_tlist,
        device=_device,
        sparse=_sparse,
    )
    return test_system, test_magnus


def state_transfer_infidelity(
    **kwargs,
):
    test_system, test_magnus = setup_magnus_chain_example(**kwargs)
    chain_size = kwargs.get("chain_size", None)
    num_magnus_intervals = kwargs.get("num_magnus_intervals", None)
    allzero_state = qt.basis(dimensions=[2] * chain_size, n=[0] * chain_size)
    allone_state = qt.basis([2] * chain_size, n=[1] * chain_size)

    P1 = allone_state.proj()

    test_psi0 = np.asarray((allzero_state).unit()[:])
    final_state = qt.Qobj(
        cp.asnumpy(
            more_itertools.last(
                test_magnus.evolve(
                    init_state=test_psi0,
                    num_intervals=num_magnus_intervals,
                )
            )
        ),
        dims=[[2] * chain_size, [1] * chain_size],
    )

    infidelity = 1 - qt.expect(P1, final_state)
    return infidelity
