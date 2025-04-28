# Tests for RSWT
import numpy as np
import pytest
import qutip as qt
import scipy.linalg as la

from qcheff.rswt.rswt_gpu import rswt_nested_commutator

dim_list = np.geomspace(2, 100, 5, dtype="int")


@pytest.mark.parametrize("dim", dim_list)
def test_nested_commutator_base_case(dim: int):
    """
    Test the base case for nested commutator.
    """

    M1 = qt.rand_herm(dim)[:]
    M2 = qt.rand_herm(dim)[:]

    assert np.allclose(
        rswt_nested_commutator(M1, M2, order=0), M2
    ), "Nested commutator base case is wrong."


@pytest.mark.parametrize("dim", dim_list)
def test_nested_commutator_order_recurrence(dim: int):
    """
    Test that the recurrence relation is accurate.
    """

    M1 = qt.rand_herm(dim)[:]
    M2 = qt.rand_herm(dim)[:]

    rng = np.random.default_rng()
    order = rng.integers(low=0, high=100)

    assert np.allclose(
        rswt_nested_commutator(M1, M2, order=order + 1),
        qt.commutator(M1, rswt_nested_commutator(M1, M2, order=order)),
    ), "Nested commutator does not recurse correctly."


@pytest.mark.skip(reason="NotImplemented")
# parametrize("dim", dim_list)
def test_rswt_generator_one_coupling(dim: int):
    pass


@pytest.mark.skip(reason="NotImplemented")
# parametrize("dim", dim_list)
def test_rswt_one_step(dim: int):
    pass


@pytest.mark.skip(reason="NotImplemented")
# parametrize("dim", dim_list)
def test_rswt(dim: int):
    pass
