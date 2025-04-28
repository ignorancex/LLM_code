import itertools

import numpy as np
import numpy.testing as npt
import pytest
import qutip as qt

from qcheff import qcheff_config, temp_config
from qcheff.operators import create, destroy, eye
from qcheff.operators.utils import commutator, embed_operator, eye_like, tensor, tensor2

test_ns = [2, 5, 10]
test_dtypes = [np.float32, np.float64, np.complex64, np.complex128]
test_sparsity = ["sparse", "dense"]
test_backend = ["cpu", "gpu"]
test_configs = [
    {"backend": b, "sparse": True if s == "sparse" else False, "default_dtype": d}
    for b, s, d in itertools.product(test_backend, test_sparsity, test_dtypes)
]


def config_id(config):
    return f"{config['backend']}-{config['sparse']}-{config['default_dtype'].__name__}"


@pytest.mark.parametrize("config", test_configs, ids=config_id)
@pytest.mark.parametrize("n", test_ns)
def test_commutator(config, n):
    with temp_config(**config):
        A = create(n)
        B = destroy(n)

        qt_A = qt.create(n)
        qt_B = qt.destroy(n)

        result = commutator(B, A)
        expected = qt.commutator(qt_B, qt_A)[:]
        if config["backend"] == "gpu":
            result = result.get()
        if config["sparse"]:
            result = result.toarray()
        else:
            npt.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("config", test_configs, ids=config_id)
@pytest.mark.parametrize("n", test_ns)
def test_anticommutator(config, n):
    with temp_config(**config):
        A = create(n)
        B = destroy(n)

        qt_A = qt.create(n)
        qt_B = qt.destroy(n)

        result = commutator(B, A, kind="anti")
        expected = qt.commutator(qt_B, qt_A, kind="anti").full()
        if config["backend"] == "gpu":
            result = result.get()
        if config["sparse"]:
            result = result.toarray()
        npt.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("config", test_configs, ids=config_id)
@pytest.mark.parametrize("n", test_ns)
def test_tensor2(config, n):
    with temp_config(**config):
        A = create(n)
        B = destroy(n)

        qt_A = qt.create(n)
        qt_B = qt.destroy(n)
        result = tensor2(A, B)
        expected = qt.tensor((qt_A, qt_B)).full()
        if config["backend"] == "gpu":
            result = result.get()
        if config["sparse"]:
            result = result.toarray()
        npt.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("config", test_configs, ids=config_id)
@pytest.mark.parametrize("n", test_ns)
def test_tensor(config, n):
    with temp_config(**config):
        A = create(n)
        B = destroy(n)
        C = eye(n)

        qt_A = qt.create(n)
        qt_B = qt.destroy(n)
        qt_C = qt.qeye(n)

        result = tensor(A, B, C)
        expected = qt.tensor((qt_A, qt_B, qt_C)).full()
        if config["backend"] == "gpu":
            result = result.get()
        if config["sparse"]:
            result = result.toarray()
        npt.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("config", test_configs, ids=config_id)
@pytest.mark.parametrize("n", test_ns)
def test_eye_like(config, n):
    with temp_config(**config):
        A = create(n)
        expected = eye(n)
        result = eye_like(A)

        if config["backend"] == "gpu":
            result = result.get()
            expected = expected.get()
        if config["sparse"]:
            result = result.toarray()
            expected = expected.toarray()
        npt.assert_allclose(result, expected, atol=1e-6)
