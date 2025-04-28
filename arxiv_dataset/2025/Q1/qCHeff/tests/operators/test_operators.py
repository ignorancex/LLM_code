import itertools

import cupy as cp
import numpy as np
import numpy.testing as npt
import pytest
import qutip as qt

import qcheff.operators.operators as qops
from qcheff import qcheff_config, temp_config

test_n = 5

# List of functions to test
# 3-tuples containing the function to test, the equivalent QuTiP function,
# and the arguments to pass to the function.
# both function names are passed as strings.

test_params = [
    ("eye", "qeye", (test_n,)),
    ("create", "create", (test_n,)),
    ("destroy", "destroy", (test_n,)),
    ("number", "num", (test_n,)),
    ("identity", "qeye", (test_n,)),
    ("basis", "basis", (test_n, 2)),
    ("projector", "projection", (test_n, 2, 3)),
    ("charge", "charge", (test_n,)),
    ("position", "position", (test_n,)),
    ("momentum", "momentum", (test_n,)),
    ("sigmax", "sigmax", ()),
    ("sigmay", "sigmay", ()),
    ("sigmaz", "sigmaz", ()),
    ("sigmap", "sigmap", ()),
    ("sigmam", "sigmam", ()),
]

test_dtypes = [np.complex64, np.complex128, None]
test_sparsity = ["sparse", "dense"]
test_backend = ["cpu", "gpu"]
test_configs = [
    {"backend": b, "sparse": True if s == "sparse" else False, "default_dtype": d}
    for b, s, d in itertools.product(test_backend, test_sparsity, test_dtypes)
]


def config_id(config):
    if config["default_dtype"] is None:
        _config_id = f"{config['backend']}-{config['sparse']}"
    else:
        _config_id = (
            f"{config['backend']}-{config['sparse']}-{config['default_dtype'].__name__}"
        )

    return _config_id


qutip_comparison_test_ids = [f"test_{f}_same_as_qutip_{fq}" for f, fq, _ in test_params]


@pytest.mark.parametrize("test_params", test_params, ids=qutip_comparison_test_ids)
@pytest.mark.parametrize("config", test_configs, ids=config_id)
def test_function_qutip_comparison(config, test_params):
    with temp_config(**config):
        function_name, qutip_function_name, args = test_params
        function = getattr(qops, function_name)
        qutip_function = getattr(qt, qutip_function_name)
        result = function(*args, dtype=config["default_dtype"])
        expected = qutip_function(*args).full()
        if config["backend"] == "gpu":
            result = result.get()
        if config["sparse"]:
            result = result.toarray()

        npt.assert_allclose(result, expected)


sparsity_test_ids = [f"test_{f}(sparse=True)_is_sparse" for f, fq, _ in test_params]


@pytest.mark.parametrize("test_params", test_params, ids=sparsity_test_ids)
@pytest.mark.parametrize("config", test_configs, ids=config_id)
def test_sparse_operator_check_sparsity(config, test_params):
    config["sparse"] = True if config["sparse"] == "sparse" else False
    with temp_config(**config):
        function_name, _, args = test_params
        function = getattr(qops, function_name)
        result = function(*args, dtype=config["default_dtype"])
        assert (
            qcheff_config.device_scipy_backend.sparse.issparse(result)
            is config["sparse"]
        )


correct_library_test_ids = [f"test_{f}_uses_correct_library" for f, _, _ in test_params]


@pytest.mark.parametrize("test_params", test_params, ids=correct_library_test_ids)
@pytest.mark.parametrize("config", test_configs, ids=config_id)
def test_function_uses_correct_library(config, test_params):
    with temp_config(**config):
        function_name, _, args = test_params
        function = getattr(qops, function_name)
        if config["backend"] == "cpu":
            assert (
                cp.get_array_module(function(*args, dtype=config["default_dtype"]))
                == np
            )
        else:
            assert (
                cp.get_array_module(function(*args, dtype=config["default_dtype"]))
                == cp
            )
