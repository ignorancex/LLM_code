import logging

import numpy as np
import pytest

from qcheff.cpu.utils import (
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


# Prepare some data for the benchmarks
@pytest.fixture(params=test_dim_list)
def data_cpu(request):
    rng = np.random.default_rng(0)
    test_dim = request.param
    test_dims = (test_dim, test_dim)
    A, B, C, D = rng.random(size=(4, *test_dims)) * np.exp(
        1j * rng.random(size=(4, *test_dims)) * 2 * np.pi
    )
    return A, B, C, D


def test_full_givens_rot_cpu(benchmark, data_cpu):
    logging.info("Running test_full_givens_rot_cpu")
    A, _, _, _ = data_cpu
    benchmark.pedantic(
        givens_rot,
        args=[A, 1, 0],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_extract_num_params_cpu(benchmark, data_cpu):
    logging.info("Running test_extract_num_params_cpu")
    A, _, _, _ = data_cpu
    benchmark.pedantic(
        extract_num_params,
        args=[A, 1, 0],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


@pytest.mark.parametrize("test_dim", test_dim_list)
def test_givens_rot_creation_cpu(benchmark, test_dim):
    logging.info("Running test_givens_rot_creation_cpu")
    benchmark.pedantic(
        givens_rot_numeric,
        args=[1, 0, np.pi / 2, 1j, test_dim],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_unitary_transformation_cpu(benchmark, data_cpu):
    logging.info("Running test_unitary_transformation_cpu")
    A, B, _, _ = data_cpu
    benchmark.pedantic(
        unitary_transformation,
        args=[A, B],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_find_largest_coupling_cpu(benchmark, data_cpu):
    logging.info("Running test_find_largest_coupling_cpu")
    A, _, _, _ = data_cpu
    benchmark.pedantic(
        find_largest_coupling,
        args=[A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_split_diag_offdiag_matrix_cpu(benchmark, data_cpu):
    logging.info("Running test_split_diag_offdiag_matrix_cpu")
    A, _, _, _ = data_cpu
    benchmark.pedantic(
        split_diag_offdiag_matrix,
        args=[A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )
