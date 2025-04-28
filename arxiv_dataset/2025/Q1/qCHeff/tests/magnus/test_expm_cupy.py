from itertools import product

import cupy as cp
import pytest
import scipy.linalg as la

from qcheff.magnus.utils_gpu import expm_taylor

dim_list = [2, 10, 100]
batch_sizes = [1, 10, 100]


@pytest.mark.parametrize("k, N", product(batch_sizes, dim_list))
def test_expm(k: int, N: int):
    # Create a complex tensor of shape (k, N, N)
    rng = cp.random.default_rng(12345)
    tensor = rng.random((k, N, N), dtype=cp.cfloat)

    # Compute the matrix exponentiation along axis 0
    result = expm_taylor(tensor)
    expected_result = la.expm(tensor.get())

    # Check the shape of the result
    assert result.shape == (k, N, N)

    # Check that the result is close to the expected result
    cp.testing.assert_allclose(result, expected_result, atol=1e-10)


@pytest.mark.parametrize("N", dim_list)
def test_expm_unbatched(N: int):
    # Create a tensor of shape (N, N)
    rng = cp.random.default_rng(12345)
    tensor = rng.random((N, N), dtype=cp.cfloat)

    # Compute the matrix exponentiation
    result = expm_taylor(tensor)
    expected_result = la.expm(tensor.get())

    # Check the shape of the result
    assert result.shape == (N, N)

    # Check that the result is close to the expected result
    cp.testing.assert_allclose(result, expected_result, atol=1e-15)
