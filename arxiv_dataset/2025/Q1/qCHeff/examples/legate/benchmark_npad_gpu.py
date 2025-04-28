from itertools import combinations_with_replacement

import cupy as cp
import pytest
import qutip as qt
import yaml

import qcheff.npad.sparse.npad_gpu as npad_sparse
from qcheff.duffing.duffing_chain_utils import (
    create_duffing_chain_zz_system,
    duffing_chain_qutip_ham,
)
from qcheff.npad import npad_gpu

with open("legate_npad_params.yaml") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

##############################################################################
####### Change these parameters in the legate_npad_params.yaml file ##########
##############################################################################
test_rounds = data["benchmarking_parameters"]["test_rounds"]
test_iter = data["benchmarking_parameters"]["test_iter"]
test_warmup_rounds = data["benchmarking_parameters"]["test_warmup_rounds"]
global_chain_size = data["global_parameters"]["global_chain_size"]
global_ntrunc = data["global_parameters"]["global_ntrunc"]

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
# This function is used to prepare the data for the benchmark.
# It is run once before the benchmark starts.
# It is very slow for large n, and you should optimize/cache results here
# if needed,  but it doesn't affect the benchmark performance.
##############################################################################
def get_n_photon_couplings(n: int, system_dims, end_zz_levels):
    pm_gens = cp.asarray(
        cp.identity(global_chain_size) - cp.diag(cp.ones(global_chain_size - 1), k=1),
        dtype=int,
    )[:-1]
    mp_gens = -pm_gens
    all_1p_gens = cp.vstack([pm_gens, mp_gens])
    all_np_gens = cp.vstack(
        [sum(gens) for gens in combinations_with_replacement(all_1p_gens, n)]
    )
    fstates = cp.vstack(end_zz_levels)[:, None, :] + all_np_gens
    istates = cp.broadcast_to(cp.vstack(end_zz_levels)[:, None, :], fstates.shape)

    coupling_tuples = [
        (istate, fstate)
        for istate, fstate in zip(cp.vstack(istates), cp.vstack(fstates), strict=False)
        if cp.all(fstate >= 0)
        and cp.all(fstate < global_ntrunc)
        and not cp.all(fstate == 0)
    ]

    coupling_idx = cp.asarray(
        [
            (
                qt.state_number_index(system_dims, i),
                qt.state_number_index(system_dims, j),
            )
            for i, j in coupling_tuples
            if (i - j).any()
        ]
    )
    return coupling_idx


system_dims = [global_ntrunc] * global_chain_size
all_levels = list(qt.state_number_enumerate(system_dims))
end_zz_levels = [
    label
    for label in all_levels
    if label[0] in [0, 1] and label[-1] in [0, 1] and set(label[1:-1]) == {0}
]
other_levels = [label for label in all_levels if label not in end_zz_levels]
end_zz_idx = cp.asarray(
    [qt.state_number_index(system_dims, state) for state in end_zz_levels]
)
other_idx = cp.asarray(
    [qt.state_number_index(system_dims, state) for state in other_levels]
)


coupling_batches = [
    get_n_photon_couplings(n, system_dims, end_zz_levels)
    for n in range(1, global_chain_size)
]


##############################################################################
# Setup function
# This function is used to prepare the data for the benchmark
# It is run once before the benchmark starts.
# Do not change this.
##############################################################################
@pytest.fixture()
def prep_data_gpu():
    # Do NOT change these parameters
    # delp = del1 + del2
    # delm = del1 - del2
    # del1 = w1 - wr
    # del2 = w2 - wr
    delp = 0.1
    delm = 0.4
    test_sys = create_duffing_chain_zz_system(
        chain_size=global_chain_size,
        delp=delp,
        delm=delm,
        ntrunc=global_ntrunc,
    )

    cp.cuda.Device().synchronize()
    test_ham = duffing_chain_qutip_ham(test_sys)

    return test_ham, coupling_batches, end_zz_idx


##############################################################################
# This is the function under test for dense matrices.
# Do not change code below.
# Please optimize the function called within.
##############################################################################
def npad_chain_auto_zz_gpu_dense(test_ham, coupling_batches, end_zz_idx):
    for batch in coupling_batches:
        test_ham = npad_gpu.npad_eliminate_couplings_simultaneous(test_ham, batch)
    npad_cz_ham = npad_gpu.npad_diagonalize(
        test_ham[cp.ix_(end_zz_idx, end_zz_idx)], eps=1e-12
    )
    E00, E01, E10, E11 = cp.diag(npad_cz_ham)
    return E11 + E00 - E01 - E10


##############################################################################
# The code below runs the benchmark for the dense implementation.
# Do not change this code.
##############################################################################
@pytest.mark.parametrize("chain_size, ntrunc", [(global_chain_size, global_ntrunc)])
def test_npad_auto_chain_zz_gpu_dense(benchmark, prep_data_gpu, chain_size, ntrunc):
    test_ham, coupling_batches, end_zz_idx = prep_data_gpu
    test_ham = cp.asarray(test_ham[:])

    benchmark.pedantic(
        sync_wrapper,
        args=(npad_chain_auto_zz_gpu_dense, test_ham, coupling_batches, end_zz_idx),
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


##############################################################################
# This is the function under test for sparse matrices.
# Do not change code below.
# Please optimize the function called within.
##############################################################################
def npad_chain_auto_zz_gpu_sparse(test_ham, coupling_batches, end_zz_idx):
    for batch in coupling_batches:
        test_ham = npad_sparse.npad_eliminate_couplings_simultaneous(test_ham, batch)
    npad_cz_ham = npad_gpu.npad_diagonalize(
        test_ham[cp.ix_(end_zz_idx, end_zz_idx)], eps=1e-12
    )
    E00, E01, E10, E11 = cp.diag(npad_cz_ham)
    return E11 + E00 - E01 - E10


##############################################################################
# The code below runs the benchmark for sparse matrices.
# Do not change this code.
##############################################################################
@pytest.mark.parametrize("chain_size, ntrunc", [(global_chain_size, global_ntrunc)])
def test_npad_auto_chain_zz_gpu_sparse(benchmark, prep_data_gpu, chain_size, ntrunc):
    test_ham, coupling_batches, end_zz_idx = prep_data_gpu
    test_ham = cp.asarray(test_ham[:])

    benchmark.pedantic(
        sync_wrapper,
        args=(npad_chain_auto_zz_gpu_sparse, test_ham, coupling_batches, end_zz_idx),
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )
