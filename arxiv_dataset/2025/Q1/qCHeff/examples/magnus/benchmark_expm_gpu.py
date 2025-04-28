import logging
from itertools import product

import cupy as cp
import pytest

from qcheff.magnus.utils_gpu import expm_taylor

test_dim_list = [10, 100, 1000]
test_batch_size_list = [1, 2, 5, 10, 100, 1000]

test_params_list = [(k, N, N) for k, N in product(test_batch_size_list, test_dim_list)]

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
@pytest.fixture(params=test_params_list, ids=lambda x: f"k={x[0]}, N={x[1]}")
def data_gpu(request):
    rng = cp.random.default_rng(0)
    A = rng.random(size=request.param, dtype=cp.cfloat)

    cp.cuda.Device().synchronize()
    return A


def test_expm_gpu(benchmark, data_gpu):
    A = data_gpu
    result = benchmark.pedantic(
        sync_wrapper,
        args=[expm_taylor, A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )
    assert result.shape == A.shape
