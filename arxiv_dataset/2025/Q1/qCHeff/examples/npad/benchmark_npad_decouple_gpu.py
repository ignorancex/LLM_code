import cupy as cp
import matplotlib as mpl
import pytest

# import numpy as np
import qutip as qt

from qcheff.duffing.duffing_chain_utils import (
    create_uneven_Duffing_Chain,
    duffing_chain_qutip_ham,
)
from qcheff.gpu import npad_decouple_sim

global_ntrunc = 3

cp.cuda.Device(1).use()

chain_size_list = list(range(2, 10))
batch_size_list = [5]  # , 10, 20, 50, 100]


def gen_decouple_data_warmup(chain_size):
    system_dims = [global_ntrunc] * chain_size
    example_chain = create_uneven_Duffing_Chain(
        chain_size=chain_size,
        ntrunc=global_ntrunc,
    )
    chain_full_ham = cp.asarray(duffing_chain_qutip_ham(example_chain)[:])
    all_levels = list(qt.state_number_enumerate(system_dims))
    qubit_chain_levels = [label for label in all_levels if max(label) <= 1]
    other_levels = [label for label in all_levels if label not in qubit_chain_levels]

    chain_idx = cp.asarray(
        [qt.state_number_index(system_dims, label) for label in qubit_chain_levels]
    )
    other_idx = cp.asarray(
        [qt.state_number_index(system_dims, label) for label in other_levels]
    )
    return chain_full_ham, chain_idx, other_idx


def run_gpu(**kwargs):
    res = npad_decouple_sim(**kwargs)
    cp.cuda.Device().synchronize()
    return res


@pytest.mark.skip("Slow and fails to converge for large matrices.")
@pytest.mark.parametrize("chain_size", chain_size_list)
def test_npad_decouple_gpu(benchmark, chain_size):
    """
    Sets up a chain of Duffing Oscillators, and then decouples the qubit chain subspace.


    """

    data_ham, incl_idx, excl_idx = gen_decouple_data_warmup(chain_size)

    kwarg_dict = {
        "H": data_ham,
        "incl_idx": incl_idx,
        "excl_idx": excl_idx,
        "eps": 1e-6,
        "ret_norm": False,
        "max_rots": 1000,
        "batch_size": 20,
        "debug": False,
    }
    # Warmup
    _ = run_gpu(**kwarg_dict).get()

    decH = benchmark.pedantic(run_gpu, kwargs=kwarg_dict, rounds=5)

    full_eigs = cp.sort(cp.linalg.eigvalsh(data_ham))
    subspace_eigs = cp.sort(cp.linalg.eigvalsh(decH[cp.ix_(incl_idx, incl_idx)]))

    assert numpy.all(
        [numpy.isclose(eigval.get(), full_eigs.get()).any() for eigval in subspace_eigs]
    ), "Did not converge"


@pytest.mark.parametrize("chain_size", chain_size_list)
@pytest.mark.parametrize("batch_size", batch_size_list)
def test_npad_decouple_iteration_speed_gpu(benchmark, chain_size, batch_size):
    """
    Sets up a chain of Duffing Oscillators, and then decouples the qubit chain subspace.


    """

    data_ham, incl_idx, excl_idx = gen_decouple_data_warmup(chain_size)

    kwarg_dict = {
        "H": data_ham,
        "incl_idx": incl_idx,
        "excl_idx": excl_idx,
        "eps": 1e-6,
        "ret_norm": False,
        "max_rots": 1,
        "batch_size": batch_size,
        "debug": False,
    }
    # Warmup
    _ = run_gpu(**kwarg_dict).get()

    benchmark.pedantic(run_gpu, kwargs=kwarg_dict, rounds=5)
