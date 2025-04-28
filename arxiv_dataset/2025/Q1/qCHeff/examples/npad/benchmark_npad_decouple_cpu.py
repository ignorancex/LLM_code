import matplotlib as mpl
import numpy as np
import pytest
import qutip as qt

from qcheff.cpu.npad import npad_decouple_sim
from qcheff.duffing.duffing_chain_utils import (
    create_uneven_Duffing_Chain,
    duffing_chain_qutip_ham,
)

global_ntrunc = 3

chain_size_list = list(range(2, 10))
batch_size_list = [5]  # , 10]


def gen_decouple_data_warmup(chain_size):
    system_dims = [global_ntrunc] * chain_size
    example_chain = create_uneven_Duffing_Chain(
        chain_size=chain_size, ntrunc=global_ntrunc
    )
    chain_full_ham = duffing_chain_qutip_ham(example_chain)[:]

    all_levels = list(qt.state_number_enumerate(system_dims))
    qubit_chain_levels = [label for label in all_levels if max(label) <= 1]
    other_levels = [label for label in all_levels if label not in qubit_chain_levels]

    chain_idx = np.asarray(
        [qt.state_number_index(system_dims, label) for label in qubit_chain_levels]
    )
    other_idx = np.asarray(
        [qt.state_number_index(system_dims, label) for label in other_levels]
    )
    return chain_full_ham, chain_idx, other_idx


@pytest.mark.skip("Slow for large matrices.")
@pytest.mark.parametrize("chain_size", chain_size_list)
def test_npad_decouple_cpu(benchmark, chain_size):
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
    _ = npad_decouple_sim(**kwarg_dict)

    decH = benchmark.pedantic(npad_decouple_sim, kwargs=kwarg_dict, rounds=5)

    full_eigs = np.sort(np.linalg.eigvalsh(data_ham))
    subspace_eigs = np.sort(np.linalg.eigvalsh(decH[np.ix_(incl_idx, incl_idx)]))
    assert np.all(
        [np.isclose(eigval, full_eigs).any() for eigval in subspace_eigs]
    ), "Did not converge"


@pytest.mark.parametrize("chain_size", chain_size_list)
@pytest.mark.parametrize("batch_size", batch_size_list)
def test_npad_decouple_iteration_speed_cpu(benchmark, chain_size, batch_size):
    """
    Sets up a chain of Duffing Oscillators, and then performs 10
    decoupling steps the qubit chain subspace.
    Checking performance v/s batch size


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
    _ = npad_decouple_sim(**kwarg_dict)

    benchmark.pedantic(npad_decouple_sim, kwargs=kwarg_dict, rounds=5)
