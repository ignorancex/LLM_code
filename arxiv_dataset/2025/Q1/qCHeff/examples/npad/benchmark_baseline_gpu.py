import logging

import cupy as cp
import pytest

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


def test_transpose_gpu(benchmark, data_gpu):
    logging.info("Running test_transpose_gpu")
    A, _, _, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[cp.transpose, A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_dagger_gpu(benchmark, data_gpu):
    logging.info("Running test_dagger_gpu")
    A, _, _, _ = data_gpu

    def dagger(x):
        return x.conj().transpose()

    benchmark.pedantic(
        sync_wrapper,
        args=[dagger, A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_matmul_gpu(benchmark, data_gpu):
    logging.info("Running test_matmul_gpu")
    A, B, _, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[cp.matmul, A, B],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_matmul_inplace_gpu(benchmark, data_gpu):
    logging.info("Running test_matmul_inplace_gpu")
    A, B, C, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[cp.matmul, A, B],
        kwargs={"out": C},
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_consec_matmul_gpu(benchmark, data_gpu):
    logging.info("Running test_matmul_cpu")
    A, B, C, _ = data_gpu

    def consec_matmul(a, b, c):
        return a @ b @ c

    benchmark.pedantic(
        sync_wrapper,
        args=[consec_matmul, A, B, C],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_consec_matmul_inplace_gpu(benchmark, data_gpu):
    logging.info("Running test_matmul_inplace_cpu")
    A, B, C, D = data_gpu

    def consec_matmul_ip(A, B, C, out):
        return cp.matmul(cp.matmul(A, B, out=out), C, out=out)

    benchmark.pedantic(
        sync_wrapper,
        args=[consec_matmul_ip, A, B, C],
        kwargs={"out": D},
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_einsum_2_gpu(benchmark, data_gpu):
    logging.info("Running test_einsum_2_gpu")
    A, B, _, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[cp.einsum, "ij,jk->ik", A, B],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_einsum_3_gpu(benchmark, data_gpu):
    logging.info("Running test_einsum_3_gpu")
    A, B, C, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[cp.einsum, "ij,jk,kl->il", A, B, C],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_einsum_4_gpu(benchmark, data_gpu):
    logging.info("Running test_einsum_4_gpu")
    A, B, C, D = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[cp.einsum, "ij,jk,kl,lm->im", A, B, C, D],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_argmax_gpu(benchmark, data_gpu):
    logging.info("Running test_argmax_gpu")
    A, _, _, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[cp.argmax, A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )


def test_argmin_gpu(benchmark, data_gpu):
    logging.info("Running test_argmin_gpu")
    A, _, _, _ = data_gpu
    benchmark.pedantic(
        sync_wrapper,
        args=[cp.argmin, A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )
