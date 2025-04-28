import itertools

import cupy as cp
import cupyx.scipy.sparse as cpsparse
import numpy as np
import pytest
import scipy.sparse as spsparse

from qcheff.magnus import magnus
from qcheff.operators import DenseOperator, SparseOperator, sigmax, sigmaz

dtypes = [np.float32, np.float64, np.complex64, np.complex128]
devices = ["cpu", "gpu"]
sparsity = ["sparse", "dense"]

configs = [
    {"dtype": dtype, "device": device, "sparsity": sparse}
    for dtype, device, sparse in itertools.product(dtypes, devices, sparsity)
]


def config_id_func(config):
    return f"{config['dtype'].__name__}_{config['device']}_{config['sparsity']}"


@pytest.fixture(params=configs, ids=config_id_func)
# @pytest.mark.parametrize("request2", configs)
def test_magnus(request):
    config = request.param
    if config["sparsity"] == "sparse":
        test_drift_ham = SparseOperator(sigmaz().astype(config["dtype"]))
        test_control_ham = [SparseOperator(sigmax().astype(config["dtype"]))]

    else: # config["sparsity"] == "dense"
        test_drift_ham = DenseOperator(sigmaz().astype(config["dtype"]))
        test_control_ham = [DenseOperator(sigmax().astype(config["dtype"]))]

    if config["device"] == "gpu":
        test_drift_ham.to("gpu")
        for control in test_control_ham:
            control.to("gpu")
    xp = cp.get_array_module(test_drift_ham.op)
    test_tlist = xp.linspace(0, 5, 100)
    test_control_sigs = xp.linspace(0, 5, 100)
    test_magnus = magnus(
        tlist=test_tlist,
        drift_ham=test_drift_ham,
        control_sigs=test_control_sigs,
        control_hams=test_control_ham,
    )

    yield test_magnus


def test_magnus_hamiltonial_is_hermitian(test_magnus):
    # Check if the magnus1_hams are Hermitian
    magnus1_hams = test_magnus.magnus_hamiltonians(num_intervals=5)
    for matrix in magnus1_hams:
        non_hermiticity = matrix - matrix.conj().transpose()
        if cpsparse.issparse(matrix) or spsparse.issparse(matrix):
            non_hermiticity = cp.asnumpy(non_hermiticity.toarray())
        else:
            non_hermiticity = cp.asnumpy(non_hermiticity)
    np.testing.assert_allclose(
        non_hermiticity,
        np.zeros_like(non_hermiticity),
        atol=1e-10,
    )


def test_magnus_propagator_is_unitary(test_magnus):
    # Check if the magnus1_hams are Hermitian
    magnus1_props = test_magnus.magnus_propagators(num_intervals=5)

    for matrix in magnus1_props:
        unitarity = matrix @ matrix.conj().transpose()
        if cpsparse.issparse(matrix) or spsparse.issparse(matrix):
            unitarity = cp.asnumpy(unitarity.toarray())
        else:
            unitarity = cp.asnumpy(unitarity)
    np.testing.assert_allclose(
        unitarity,
        np.eye(matrix.shape[0], dtype=complex),
        atol=1e-10,
    )
