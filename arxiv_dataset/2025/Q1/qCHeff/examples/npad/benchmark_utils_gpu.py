import logging

import cupy as cp
import pytest

from qcheff.gpu.utils import (
    extract_num_params,
    find_largest_coupling,
    givens_rot,
    givens_rot_numeric,
    split_diag_offdiag_matrix,
    unitary_transformation,
)

test_dim_list = [10, 100, 1000, 10000, 20000]
test_rounds = 5
test_iter = 1
test_warmup_rounds = 2

logging.basicConfig(level=logging.INFO)

cp.cuda.Device(1).use()


def sync_wrapper(func, *args, **kwargs):
    """
    Synchronizes the GPU after running the function.
    """
    res = func(*args, **kwargs)
    cp.cuda.Device().synchronize()
    return res


# Prepare some data for the benchmarks
@pytest.fixture(params=test_dim_list)
def data_gpu(request):
    rng = cp.random.default_rng(0)
    test_dim = request.param
    test_dims = (test_dim, test_dim)
    A, B, C, D = rng.random(size=(4, *test_dims), dtype=cp.complex128)
    cp.cuda.Device().synchronize()
    return A, B, C, D


def test_full_givens_rot_gpu(benchmark, data_gpu):
    logging.info("Running test_full_givens_rot_gpu")
    A, _, _, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[givens_rot, A, 1, 0],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_extract_num_params_gpu(benchmark, data_gpu):
    logging.info("Running test_extract_num_params_gpu")
    A, _, _, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[extract_num_params, A, 1, 0],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


@pytest.mark.parametrize("dim", test_dim_list)
def test_givens_rot_creation_gpu(benchmark, dim):
    logging.info("Running test_givens_rot_creation_gpu")
    benchmark.pedantic(
        sync_wrapper,
        args=[givens_rot_numeric, 1, 0, cp.pi / 2, 1j, dim],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_unitary_transformation_gpu(benchmark, data_gpu):
    logging.info("Running test_unitary_transformation_gpu")
    A, B, _, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[unitary_transformation, A, B],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_find_largest_coupling_gpu(benchmark, data_gpu):
    logging.info("Running test_find_largest_coupling_gpu")
    A, _, _, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[find_largest_coupling, A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_split_diag_offdiag_matrix_gpu(benchmark, data_gpu):
    logging.info("Running test_split_diag_offdiag_matrix_gpu")
    A, _, _, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[split_diag_offdiag_matrix, A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )
