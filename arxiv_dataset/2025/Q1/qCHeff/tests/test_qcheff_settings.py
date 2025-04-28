import importlib
import logging

import numpy as np
import pytest

# We only import this class to test its methods
from qcheff import _QCHeffConfig, qcheff_config, temp_config

dtypes = [np.float32, np.float64, np.complex64, np.complex128]


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_set_backend(backend):
    config = _QCHeffConfig()
    config.set_backend(backend)

    match backend:
        case "cpu":
            intended_xp_module = importlib.import_module("numpy")
            intended_scipy_module = importlib.import_module("scipy")
            intended_linalg_module = (
                importlib.import_module("scipy.sparse.linalg")
                if config.sparse
                else importlib.import_module("scipy.linalg")
            )

        case "gpu":
            intended_xp_module = importlib.import_module("cupy")
            intended_scipy_module = importlib.import_module("cupyx.scipy")
            intended_linalg_module = (
                importlib.import_module("cupyx.scipy.sparse.linalg")
                if config.sparse
                else importlib.import_module("cupy.linalg")
            )
    assert (
        config.backend == backend
        and config.device_xp_backend == intended_xp_module
        and config.device_scipy_backend == intended_scipy_module
        and config.default_dtype == config.device_xp_backend.complex128
        and config.device_linalg_backend == intended_linalg_module
    )


def test_set_backend_invalid():
    config = _QCHeffConfig()
    with pytest.raises(ValueError):
        config.set_backend("invalid")


def test_set_default_dtype():
    config = _QCHeffConfig()
    config.set(default_dtype=config._device_xp_backend.float64)
    assert config.default_dtype == config._device_xp_backend.float64


def test_set_default_dtype_invalid():
    config = _QCHeffConfig()
    with pytest.raises(ValueError):
        config.set(default_dtype=int)


def test_set_invalid_key():
    config = _QCHeffConfig()
    with pytest.raises(KeyError):
        config.set(invalid_key="value")


def test_list_options():
    """
    Test that calling list_options() on the _QCHeffConfig object doesn't raise an error.
    """

    config = _QCHeffConfig()
    config.list_options()  # This should print the options without raising an error


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("sparsity", ["sparse", "dense"])
@pytest.mark.parametrize("dtype", dtypes)
def test_temp_config(backend, sparsity, dtype):
    """
    Test that the context manager correctly sets and resets the configuration
    of the qcheff_config object.

    The test sets the backend, sparse, and default_dtype options, and checks
    that they are correctly set and reset both inside and outside the context
    manager.
    """
    sparsity = True if sparsity == "sparse" else False
    original_backend = qcheff_config.backend
    original_sparse = qcheff_config.sparse
    original_dtype = qcheff_config.default_dtype

    with temp_config(backend=backend, sparse=sparsity, default_dtype=dtype):
        assert qcheff_config.backend == backend
        assert qcheff_config.sparse == sparsity
        assert qcheff_config.default_dtype == dtype

    assert qcheff_config.backend == original_backend
    assert qcheff_config.sparse == original_sparse
    assert qcheff_config.default_dtype == original_dtype


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("dtype", dtypes)
def test_temp_config_restore_on_exception(backend, sparse, dtype):
    """
    Test that the context manager restores the original values
    of the qcheff_config object even if an exception is raised
    inside the context.
    """
    original_backend = qcheff_config.backend
    original_sparse = qcheff_config.sparse
    original_dtype = qcheff_config.default_dtype

    try:
        with temp_config(backend=backend, sparse=sparse, default_dtype=dtype):
            assert qcheff_config.backend == backend
            assert qcheff_config.sparse == sparse
            assert qcheff_config.default_dtype == dtype
            msg = "Test exception"
            raise Exception(msg)
    except Exception as e:
        logging.exception(e)

    assert qcheff_config.backend == original_backend
    assert qcheff_config.sparse == original_sparse
    assert qcheff_config.default_dtype == original_dtype


def test_temp_config_invalid_setting():
    with pytest.raises(AttributeError):
        with temp_config(invalid_setting="value"):
            pass
