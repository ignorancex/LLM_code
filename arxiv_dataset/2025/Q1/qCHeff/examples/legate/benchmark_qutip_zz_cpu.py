from itertools import product

import numpy as np
import pytest
import qutip as qt
import yaml

from qcheff.duffing.duffing_chain_utils import (
    create_duffing_chain_zz_system,
    duffing_chain_qutip_ham,
)

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

##############################################################################


system_dims = [global_ntrunc] * global_chain_size
all_levels = list(qt.state_number_enumerate(system_dims))
end_zz_levels = [
    list(label)
    for label in all_levels
    if label[0] in [0, 1] and label[-1] in [0, 1] and set(label[1:-1]) == {0}
]

# other_levels = [label for label in all_levels if label not in end_zz_levels]
# end_zz_idx = np.asarray([qt.state_number_index(system_dims, state) for state in end_zz_levels])
# other_idx = np.asarray([qt.state_number_index(system_dims, state) for state in other_levels])
end_zz_states = [qt.basis(system_dims, level) for level in end_zz_levels]


##############################################################################
# Setup function
# This function is used to prepare the data for the benchmark
# It is run once before the benchmark starts.
# Do not change this.
##############################################################################
@pytest.fixture()
def prep_data_cpu():
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

    test_ham = duffing_chain_qutip_ham(test_sys)

    return test_ham


##############################################################################
# This is the function under test for sparse matrices.
# Do not change code below.
# Please optimize the function called within.
##############################################################################
def qutip_chain_auto_zz_cpu_sparse(test_ham):
    test_evals, test_esys = test_ham.eigenstates()
    overlap_mat = np.asarray(
        [
            np.abs(test_estate.overlap(state)) ** 2
            for state, test_estate in product(end_zz_states, test_esys)
        ]
    ).reshape(4, -1)
    E00, E01, E10, E11 = test_evals[np.argmax(overlap_mat, axis=1)]
    return E11 + E00 - E01 - E10


##############################################################################
# The code below runs the benchmark for sparse matrices.
# Do not change this code.
##############################################################################
@pytest.mark.parametrize("chain_size, ntrunc", [(global_chain_size, global_ntrunc)])
def test_qutip_chain_zz_cpu_sparse(benchmark, prep_data_cpu, chain_size, ntrunc):
    test_ham = prep_data_cpu

    benchmark.pedantic(
        qutip_chain_auto_zz_cpu_sparse,
        args=(test_ham,),
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )
