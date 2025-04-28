# tests/iswt/test_iswt.py
import cupy as cp
import cupyx.scipy.sparse as cpsparse
import numpy as np
import pytest
import scipy.sparse as spsparse

from qcheff.iswt import NPAD
from qcheff.iswt.iswt import NPADCupySparse, NPADScipySparse
from qcheff.operators import DenseOperator, SparseOperator

# Define test parameters
test_devices = ["cpu", "gpu"]
test_sparsities = ["sparse", "dense"]
dtypes = [np.float32, np.float64, np.complex64, np.complex128]
test_dims = [10, 20, 30]


@pytest.mark.parametrize("device", test_devices)
@pytest.mark.parametrize("sparsity", test_sparsities)
@pytest.mark.parametrize("dtype", dtypes)
def test_npad_factory(device, sparsity, dtype):
    # Create an instance of NPADCupySparse or NPADScipySparse

    # Creat an operator
    H = np.random.rand(10, 10).astype(dtype)
    operator_class = SparseOperator if sparsity == "sparse" else DenseOperator
    operator = operator_class(H)
    if device == "gpu":
        operator.to("gpu")

    npad_instance = NPAD(operator)
    expected_npad_instance = NPADCupySparse if device == "gpu" else NPADScipySparse
    assert isinstance(npad_instance, expected_npad_instance)
    # Also check that the input has been sparsified
    if device == "gpu":
        assert cpsparse.issparse(npad_instance.H.op)
    else:
        assert spsparse.issparse(npad_instance.H.op)


# Define test functions
@pytest.mark.parametrize("device", test_devices)
@pytest.mark.parametrize("sparsity", test_sparsities)
@pytest.mark.parametrize("dtype", dtypes)
def test_givens_rotation_matrix(device, sparsity, dtype):
    H = np.ones((10, 10)).astype(dtype)
    operator_class = SparseOperator if sparsity == "sparse" else DenseOperator
    operator = operator_class(H)
    if device == "gpu":
        operator.to("gpu")

    npad_instance = NPAD(operator)

    # Test the givens_rotation_matrix method
    i, j = 0, 1
    givens_matrix = npad_instance.givens_rotation_matrix(i, j)

    assert givens_matrix.shape == (10, 10)
    # Check that the givens matrix is unitary
    _unitary_check_mat = cp.asnumpy(
        (givens_matrix @ givens_matrix.conj().transpose()).toarray()
    )

    np.testing.assert_allclose(
        _unitary_check_mat, np.eye(10), atol=np.finfo(dtype).resolution
    )


@pytest.mark.parametrize("device", test_devices)
@pytest.mark.parametrize("sparsity", test_sparsities)
@pytest.mark.parametrize("dtype", dtypes)
def test_eliminate_coupling(device, sparsity, dtype):
    H = np.diag(
        np.linspace(
            0,
            0.9,
            10,
        )
    ).astype(dtype)
    H[0, 1] = H[1, 0] = 0.05
    operator_class = SparseOperator if sparsity == "sparse" else DenseOperator
    operator = operator_class(H)
    if device == "gpu":
        operator.to("gpu")

    npad_instance = NPAD(operator)
    i, j = 0, 1
    npad_instance.eliminate_coupling(i, j)
    npad_instance.H.to("cpu")

    result_H = npad_instance.H.op

    # Check that the coupling has been eliminated
    np.testing.assert_allclose(result_H[i, j], 0, atol=np.finfo(dtype).resolution)

    np.testing.assert_allclose(result_H[j, i], 0, atol=np.finfo(dtype).resolution)


@pytest.mark.parametrize("device", test_devices)
@pytest.mark.parametrize("sparsity", test_sparsities)
@pytest.mark.parametrize("dtype", dtypes)
def test_unitary_transformation(device, sparsity, dtype):
    H = np.diag(
        np.linspace(
            0,
            0.9,
            10,
        )
    ).astype(dtype)
    H[0, 1] = H[1, 0] = 0.05
    operator_class = SparseOperator if sparsity == "sparse" else DenseOperator
    operator = operator_class(H)
    reference_operator = SparseOperator(H)
    if device == "gpu":
        operator.to("gpu")
        reference_operator.to("gpu")

    npad_instance = NPAD(operator)
    test_U = npad_instance.givens_rotation_matrix(0, 1)
    npad_instance.unitary_transformation(U=test_U)
    expected_H = SparseOperator(
        test_U @ reference_operator.op @ test_U.conj().transpose()
    )

    npad_instance.H.to("cpu")
    expected_H.to("cpu")
    result_H = npad_instance.H.op.toarray()
    expected_H = expected_H.op.toarray()

    np.testing.assert_allclose(result_H, expected_H, atol=np.finfo(dtype).resolution)


@pytest.mark.parametrize("device", test_devices)
@pytest.mark.parametrize("sparsity", test_sparsities)
@pytest.mark.parametrize("dtype", dtypes)
def test_npad_eliminate_couplings(device, sparsity, dtype):
    H = np.diag(
        np.linspace(
            0,
            0.9,
            10,
        )
    ).astype(dtype)
    H[0, 1] = H[1, 0] = 0.05
    H[4, 5] = H[5, 4] = 0.05
    operator_class = SparseOperator if sparsity == "sparse" else DenseOperator
    operator = operator_class(H)
    if device == "gpu":
        operator.to("gpu")

    npad_instance = NPAD(operator)
    npad_instance.eliminate_couplings([(0, 1), (4, 5)])
    npad_instance.H.to("cpu")

    result_H = npad_instance.H.op

    for i, j in [(0, 1), (4, 5)]:
        # Check that the coupling has been eliminated
        np.testing.assert_allclose(result_H[i, j], 0, atol=np.finfo(dtype).resolution)
        np.testing.assert_allclose(result_H[j, i], 0, atol=np.finfo(dtype).resolution)


@pytest.mark.parametrize("device", test_devices)
@pytest.mark.parametrize("sparsity", test_sparsities)
@pytest.mark.parametrize("dtype", dtypes)
def test_npad_largest_couplings(device, sparsity, dtype):
    H = np.diag(
        np.linspace(
            0,
            0.9,
            10,
        )
    ).astype(dtype)
    H[0, 1] = H[1, 0] = 0.05
    H[4, 5] = H[5, 4] = 0.05
    operator_class = SparseOperator if sparsity == "sparse" else DenseOperator
    operator = operator_class(H)
    if device == "gpu":
        operator.to("gpu")

    npad_instance = NPAD(operator)
    cpls, _ = next(npad_instance.largest_couplings(2))

    if device == "gpu":
        cpls = cpls.get()

    np.testing.assert_array_equal(cpls, np.array([(0, 1), (4, 5)]))


@pytest.mark.parametrize("device", test_devices)
@pytest.mark.parametrize("sparsity", test_sparsities)
@pytest.mark.parametrize("dtype", dtypes)
def test_npad_largest_couplings_with_levels(device, sparsity, dtype):
    H = np.diag(
        np.linspace(
            0,
            0.9,
            10,
        )
    ).astype(dtype)
    H[0, 1] = H[1, 0] = 0.05
    H[4, 5] = H[5, 4] = 0.05
    operator_class = SparseOperator if sparsity == "sparse" else DenseOperator
    operator = operator_class(H)
    if device == "gpu":
        operator.to("gpu")

    npad_instance = NPAD(operator)
    cpls, _ = next(npad_instance.largest_couplings(2, levels=(0, 1, 2)))

    if device == "gpu":
        cpls = cpls.get()

    np.testing.assert_array_equal(cpls, np.array([0, 1]))
