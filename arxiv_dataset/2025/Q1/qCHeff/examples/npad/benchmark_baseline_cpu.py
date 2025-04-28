import logging

import numpy as np
import pytest

test_dim_list = [10, 100, 1000, 10000, 20000]
test_rounds = 5
test_iter = 3
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


def test_transpose_cpu(benchmark, data_cpu):
    logging.info("Running test_transpose_cpu")
    A, _, _, _ = data_cpu
    benchmark.pedantic(
        np.transpose,
        args=[A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_dagger_cpu(benchmark, data_cpu):
    logging.info("Running test_dagger_cpu")
    A, _, _, _ = data_cpu

    def dagger(x):
        return x.conj().transpose()

    benchmark.pedantic(
        dagger,
        args=[A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_matmul_cpu(benchmark, data_cpu):
    logging.info("Running test_matmul_cpu")
    A, B, _, _ = data_cpu
    benchmark.pedantic(
        np.matmul,
        args=[A, B],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_matmul_inplace_cpu(benchmark, data_cpu):
    logging.info("Running test_matmul_inplace_cpu")
    A, B, C, _ = data_cpu
    benchmark.pedantic(
        np.matmul,
        args=[A, B],
        kwargs={"out": C},
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_consec_matmul_cpu(benchmark, data_cpu):
    logging.info("Running test_matmul_cpu")
    A, B, C, _ = data_cpu

    def consec_matmul(a, b, c):
        return a @ b @ c

    benchmark.pedantic(
        consec_matmul,
        args=[A, B, C],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_consec_matmul_inplace_cpu(benchmark, data_cpu):
    logging.info("Running test_matmul_inplace_cpu")
    A, B, C, D = data_cpu

    def consec_matmul_ip(A, B, C, out):
        return np.matmul(np.matmul(A, B, out=out), C, out=out)

    benchmark.pedantic(
        consec_matmul_ip,
        args=[A, B, C],
        kwargs={"out": D},
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


@pytest.mark.skip("")
def test_einsum_2_cpu(benchmark, data_cpu):
    logging.info("Running test_einsum_2_cpu")
    A, B, _, _ = data_cpu
    benchmark.pedantic(
        np.einsum,
        args=["ij,jk->ik", A, B],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


@pytest.mark.skip("")
def test_einsum_3_cpu(benchmark, data_cpu):
    logging.info("Running test_einsum_3_cpu")
    A, B, C, _ = data_cpu
    benchmark.pedantic(
        np.einsum,
        args=["ij,jk,kl->il", A, B, C],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


@pytest.mark.skip("")
def test_einsum_4_cpu(benchmark, data_cpu):
    logging.info("Running test_einsum_4_cpu")
    A, B, C, D = data_cpu
    benchmark.pedantic(
        np.einsum,
        args=["ij,jk,kl,lm->im", A, B, C, D],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_argmax_cpu(benchmark, data_cpu):
    logging.info("Running test_argmax_cpu")
    A, _, _, _ = data_cpu
    benchmark.pedantic(
        np.argmax,
        args=[A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_argmin_cpu(benchmark, data_cpu):
    logging.info("Running test_argmin_cpu")
    A, _, _, _ = data_cpu
    benchmark.pedantic(
        np.argmin,
        args=[A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )
