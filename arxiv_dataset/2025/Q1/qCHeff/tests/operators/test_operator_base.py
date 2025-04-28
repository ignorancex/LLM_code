import tempfile

import cupy as cp
import cupyx
import cupyx.scipy.sparse as cpsparse
import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as spsparse

from qcheff.operators import qcheffOperator
from qcheff.operators.dense_operator import DenseOperator
from qcheff.operators.operator_base import OperatorMatrix
from qcheff.operators.sparse_operator import SparseOperator

dtypes = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("sparsity", ["sparse", "dense"])
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_qcheff_operator_factory(dtype, sparsity, backend):
    if backend == "cpu":
        xp = np
        xpsparse = spsparse
    else:
        xp = cp
        xpsparse = cpsparse

    sparse = True if sparsity == "sparse" else False
    op = xp.array([[1, 2], [3, 4]], dtype=dtype)
    if sparse:
        op = spsparse.csr_array(op) if backend == "cpu" else cpsparse.csr_matrix(op)

    operator = qcheffOperator(op)
    if sparse:
        assert isinstance(operator, SparseOperator)
        if backend == "cpu":
            with pytest.raises(TypeError):
                qcheffOperator(spsparse.csr_matrix(op))
    else:
        assert isinstance(operator, DenseOperator)


@pytest.mark.parametrize("dtype", dtypes)
def test_operator_matrix_init_directly(dtype):
    op = np.array([[1, 2], [3, 4]], dtype=dtype)
    with pytest.raises(TypeError):
        OperatorMatrix(op)


@pytest.mark.parametrize("dtype", dtypes)
def test_operator_matrix_init_factory(dtype):
    op = np.array([[1, 2], [3, 4]], dtype=dtype)
    operator = qcheffOperator(op)
    npt.assert_allclose(operator.op, op)


@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_operator_init(dtype):
    op = np.array([[1, 2], [3, 4]], dtype=dtype)
    sparse_operator = SparseOperator(op)
    npt.assert_allclose(sparse_operator.op.toarray(), op)
    assert sparse_operator.backend_module == cupyx.scipy.get_array_module(op).sparse


@pytest.mark.parametrize("dtype", dtypes)
def test_dense_operator_init(dtype):
    op = np.array([[1, 2], [3, 4]], dtype=dtype)
    dense_operator = DenseOperator(op)
    npt.assert_allclose(dense_operator.op, op)
    assert dense_operator.backend_module == cp.get_array_module(op)


@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_operator_diagonals(dtype):
    op = np.array([[1, 2], [3, 4]], dtype=dtype)
    sparse_operator = SparseOperator(op)
    diagonals = sparse_operator.diagonals()
    npt.assert_allclose(diagonals, np.array([1, 4]))


@pytest.mark.parametrize("dtype", dtypes)
def test_dense_operator_diagonals(dtype):
    op = np.array([[1, 2], [3, 4]], dtype=dtype)
    dense_operator = DenseOperator(op)
    diagonals = dense_operator.diagonals()
    npt.assert_allclose(diagonals, np.array([1, 4]))


@pytest.mark.parametrize("dtype", dtypes)
def test_sparse_operator_couplings(dtype):
    op = np.array([[1, 2], [3, 4]], dtype=dtype)
    sparse_operator = SparseOperator(op)
    couplings = sparse_operator.couplings()
    npt.assert_allclose(couplings.toarray(), np.array([[0, 2], [0, 0]]))


@pytest.mark.parametrize("dtype", dtypes)
def test_dense_operator_couplings(dtype):
    op = np.array([[1, 2], [3, 4]], dtype=dtype)
    dense_operator = DenseOperator(op)
    couplings = dense_operator.couplings()
    npt.assert_allclose(couplings, np.array([[0, 2], [0, 0]]))


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("operator_class", [SparseOperator, DenseOperator])
def test_operator_matrix_save_load(dtype, operator_class):
    op = np.array([[1, 2], [3, 4]], dtype=dtype)
    operator = operator_class(op)
    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        filename = tmp.name
        operator.save(filename)
        loaded_operator = operator_class.load(filename)
    if operator_class == SparseOperator:
        npt.assert_allclose(loaded_operator.op.toarray(), op)
    else:
        npt.assert_allclose(loaded_operator.op, operator.op)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("operator_class", [SparseOperator, DenseOperator])
def test_operator_matrix_add(dtype, operator_class):
    op1 = np.array([[1, 2], [3, 4]], dtype=dtype)
    op2 = np.array([[5, 6], [7, 8]], dtype=dtype)
    operator1 = operator_class(op1)
    operator2 = operator_class(op2)
    result = operator1 + operator2
    if operator_class == SparseOperator:
        npt.assert_allclose(result.op.toarray(), op1 + op2)
    else:
        npt.assert_allclose(result.op, (op1 + op2))


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("operator_class", [SparseOperator, DenseOperator])
def test_operator_matrix_sub(dtype, operator_class):
    op1 = np.array([[1, 2], [3, 4]], dtype=dtype)
    op2 = np.array([[5, 6], [7, 8]], dtype=dtype)
    operator1 = operator_class(op1)
    operator2 = operator_class(op2)
    result = operator1 - operator2
    if operator_class == SparseOperator:
        npt.assert_allclose(result.op.toarray(), op1 - op2)
    else:
        npt.assert_allclose(result.op, (op1 - op2))


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("operator_class", [SparseOperator, DenseOperator])
def test_operator_matrix_mul(dtype, operator_class):
    op = np.array([[1, 2], [3, 4]], dtype=dtype)
    scalar = 2
    operator = operator_class(op)
    result = operator * scalar
    if operator_class == SparseOperator:
        npt.assert_allclose(result.op.toarray(), op * scalar)
    else:
        npt.assert_allclose(result.op, op * scalar)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("operator_class", [SparseOperator, DenseOperator])
def test_operator_matrix_matmul(dtype, operator_class):
    op1 = np.array([[1, 2], [3, 4]], dtype=dtype)
    op2 = np.array([[5, 6], [7, 8]], dtype=dtype)
    operator1 = operator_class(op1)
    operator2 = operator_class(op2)
    result = operator1 @ operator2
    if operator_class == SparseOperator:
        npt.assert_allclose(result.op.toarray(), op1 @ op2)
    else:
        npt.assert_allclose(result.op, (op1 @ op2))


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("operator_class", [SparseOperator, DenseOperator])
def test_to_cpu(dtype, operator_class):
    # Create a sparse operator on GPU
    if "complex" in dtype.__name__:
        op = 1j * cp.random.rand(10, 10, dtype=np.float64)
    else:
        op = cp.random.rand(10, 10, dtype=dtype)
    operator = operator_class(op)

    # Convert to CPU
    operator.to("cpu")

    # Check that the operator is now on CPU
    if operator_class == SparseOperator:
        assert spsparse.issparse(operator.op)
    else:
        assert isinstance(operator.op, np.ndarray)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("operator_class", [SparseOperator, DenseOperator])
def test_to_gpu(dtype, operator_class):
    # Create a sparse operator on CPU

    if "complex" in dtype.__name__:
        op = 1j * np.random.rand(10, 10)
    else:
        op = np.random.rand(10, 10)
    operator = operator_class(op)

    # Convert to GPU
    operator.to("gpu")

    # Check that the operator is now on GPU
    if operator_class == SparseOperator:
        assert cpsparse.issparse(operator.op)
    else:
        assert isinstance(operator.op, cp.ndarray)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("operator_class", [SparseOperator, DenseOperator])
def test_to_invalid_backend(dtype, operator_class):
    if "complex" in dtype.__name__:
        op = 1j * np.random.rand(10, 10)
    else:
        op = np.random.rand(10, 10)
    operator = operator_class(op)

    # Try to convert to an invalid backend
    with pytest.raises(ValueError):
        operator.to("invalid")


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("operator_class", [SparseOperator, DenseOperator])
def test_to_same_backend_gpu(dtype, operator_class):
    # Create a sparse operator on GPU
    if "complex" in dtype.__name__:
        op = 1j * np.random.rand(10, 10)
    else:
        op = np.random.rand(10, 10)
    operator = operator_class(op)

    # Try to convert to the same backend
    operator.to("gpu")

    # Check that the operator is still on GPU
    if operator_class == SparseOperator:
        assert cpsparse.issparse(operator.op)
    else:
        assert isinstance(operator.op, cp.ndarray)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("operator_class", [SparseOperator, DenseOperator])
def test_to_same_backend_cpu(dtype, operator_class):
    # Create a sparse operator on CPU
    if "complex" in dtype.__name__:
        op = 1j * np.random.rand(10, 10)
    else:
        op = np.random.rand(10, 10)
    operator = operator_class(op)

    # Try to convert to the same backend
    operator.to("cpu")

    # Check that the operator is still on CPU
    if operator_class == SparseOperator:
        assert spsparse.issparse(operator.op)
    else:
        assert isinstance(operator.op, np.ndarray)
