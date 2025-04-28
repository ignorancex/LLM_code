from functools import partial

import cupy as cp
import numpy as np
import pytest
import qutip as qt
import yaml

from qcheff.magnus.pulses import ControlPulse, FourierPulse
from qcheff.magnus.system import QuTiPSystem
from qcheff.spin_chain.utils import (
    embed_operator,
)

with open("legate_magnus_params.yaml") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

##############################################################################
####### Change these parameters in the legate_magnus_params.yaml file ########
##############################################################################
test_rounds = data["benchmarking_parameters"]["test_rounds"]
test_iter = data["benchmarking_parameters"]["test_iter"]
test_warmup_rounds = data["benchmarking_parameters"]["test_warmup_rounds"]
global_test_chain_size = data["global_parameters"]["global_chain_size"]
global_test_gate_time = data["global_parameters"]["global_test_gate_time"]
global_test_num_tlist = data["global_parameters"]["global_test_num_tlist"]
global_num_interval_list = data["global_parameters"]["global_num_interval_list"]

# Use whichever GPU is available
# cp.cuda.Device(1).use()


##############################################################################
# Used to sync the GPU after running a function
##############################################################################
def sync_wrapper(func, *args, **kwargs):
    """
    Synchronizes the GPU after running the function.
    """
    res = func(*args, **kwargs)
    cp.cuda.Device().synchronize()
    return res


##############################################################################
# Auxialiary function for setup. Do not change.
# Functions called here will be used only once,
# and are not intended for optimization.
##############################################################################
def setup_magnus_chain_example(
    pulse_coeffs: cp.ndarray,
    gate_time: int = 20,  # ns
    chain_size: int = 5,
    num_tlist: int = 10**3,
):
    control_pulse = FourierPulse(
        pulse_coeffs,
        gate_time=gate_time,
        frequency=0,
        amplitude=0.2,
        backend="cpu",
    )
    test_tlist = cp.linspace(0, gate_time, num_tlist)

    test_g = 5e-3
    test_J = 5e-2

    embed_pauli = partial(embed_operator, ntrunc=2, nsystems=chain_size)
    sx_ops, sy_ops, sz_ops = (
        [embed_pauli(op=sigma_op, pos=idx) for idx in range(chain_size)]
        for sigma_op in [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    )

    Hself = 0  # RWA will remove this term
    # The chain has periodic boundary conditions
    H_nn = sum(
        -test_J * sz_ops[j % chain_size] @ sz_ops[(j + 1) % chain_size]
        for j in range(chain_size)
    )
    H_nnn = sum(
        -test_g * sz_ops[j % chain_size] @ sz_ops[(j + 2) % chain_size]
        for j in range(chain_size)
    )
    Hneigbors = H_nn + H_nnn
    # Create the drift Hamiltonian
    Hdrift = Hself + Hneigbors
    # Create the control Hamiltonian
    Hcontrol = [sum(sx_ops)]
    test_magnus = QuTiPSystem(Hdrift, [control_pulse], Hcontrol).get_magnus_system(
        test_tlist
    )
    return test_magnus


##############################################################################
# Setup function
# This function is used to prepare the data for the benchmark
# It is run once before the benchmark starts and is expected to be slow
# for large systems. Do not change the internals.
##############################################################################
@pytest.fixture()
def data_gpu():
    # This is an example set of pulse coeffs.
    # Changing the size of this list should not affect performance.
    # Do not change this.
    test_coeffs = np.asarray(
        [1.0, -0.10834602, 0.29862906, 1.0, 1.0, 0.1280348, 0.18021425, 1.0]
    )
    test_magnus = sync_wrapper(
        setup_magnus_chain_example,
        pulse_coeffs=test_coeffs,
        gate_time=global_test_gate_time,
        chain_size=global_test_chain_size,
        num_tlist=global_test_num_tlist,
    )
    test_psi0 = cp.asarray(
        qt.basis(
            dimensions=[2] * global_test_chain_size, n=[0] * global_test_chain_size
        ).unit()[:]
    )

    cp.cuda.Device().synchronize()
    return test_psi0, test_magnus


##############################################################################
# Benchmarking Magnus time evolution
# The number of intervals changes the accuracy of the Magnus expansion.
# This also affects how many multiplications are done.
##############################################################################
@pytest.mark.parametrize("num_intervals", global_num_interval_list)
@pytest.mark.parametrize("chain_size", [global_test_chain_size])
def test_magnus_chain_example_gpu(benchmark, data_gpu, num_intervals, chain_size):
    psi0, test_magnus = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=(test_magnus.evolve, psi0, num_intervals),
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )
