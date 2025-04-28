import logging
from itertools import product

import numpy as np
import pytest
import scipy.linalg

test_dim_list = [10, 100, 1000]
test_batch_size_list = [1, 2, 5, 10, 100, 1000]

test_params_list = [(k, N, N) for k, N in product(test_batch_size_list, test_dim_list)]

test_rounds = 5
test_iter = 1
test_warmup_rounds = 0

logging.basicConfig(level=logging.INFO)


@pytest.fixture(params=test_params_list, ids=lambda x: f"k={x[0]}, N={x[1]}")
def data_cpu(request):
    rng = np.random.default_rng(0)
    A = rng.random(size=request.param) + 1j * rng.random(size=request.param)
    return A


def test_expm_cpu(benchmark, data_cpu):
    A = data_cpu

    result = benchmark.pedantic(
        scipy.linalg.expm,
        args=[A],
        rounds=test_rounds,
        iterations=test_iter,
        warmup_rounds=test_warmup_rounds,
    )
    assert result.shape == A.shape
