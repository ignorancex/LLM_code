import cupy as cp
import cupy.linalg as la
import numpy
import pytest
import qutip as qt

from qcheff.rswt.utils import split_diag_offdiag_matrix

dim_list = list(numpy.geomspace(20, 100, 5, dtype="int"))


@pytest.mark.parametrize("dim", dim_list)
def test_matrix_split_returns_diag(dim: int):
    """
    Test that the split gives the correct diag matrix.
    """
    M = cp.asarray(qt.rand_herm(dim)[:])

    D, _ = split_diag_offdiag_matrix(M)

    assert cp.allclose(D, cp.diag(cp.diagonal(M))), "Incorrect diagonal matrix returned"


@pytest.mark.parametrize("dim", dim_list)
def test_matrix_split_returns_offdiag(dim: int):
    """
    Test that the split gives the correct offdiag matrix.
    """
    M = cp.asarray(qt.rand_herm(dim)[:])

    _, V = split_diag_offdiag_matrix(M)

    assert cp.allclose(
        V, M - cp.diag(cp.diagonal(M))
    ), "Incorrect offdiagonal matrix returned"
