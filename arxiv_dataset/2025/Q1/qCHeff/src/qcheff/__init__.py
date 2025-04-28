import importlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Type

__all__ = ["qcheff_config"]


@dataclass(kw_only=True)
class _QCHeffConfig:
    """
    Default configuration for qcheff package.
    Not meant to be instantiated by the user.

    Instead, use the `qcheff_config` instance provided in this module.
    """

    backend: str = "cpu"
    sparse: bool = True
    debug: bool = False
    _device_xp_backend: ModuleType = field(init=False, repr=False)
    _device_scipy_backend: ModuleType = field(init=False, repr=False)
    _device_linalg_backend: ModuleType = field(init=False, repr=False)
    default_dtype: type | None = field(default=None)

    def __post_init__(self):
        self.set_backend(self.backend)

    @property
    def device_xp_backend(self):
        return self._device_xp_backend

    @property
    def device_scipy_backend(self):
        return self._device_scipy_backend

    @property
    def device_linalg_backend(self):
        return self._device_linalg_backend

    def set_backend(self, backend):
        self.backend = backend
        if backend == "cpu":
            self._device_xp_backend = importlib.import_module("numpy")
            self._device_scipy_backend = importlib.import_module("scipy")
            self._device_linalg_backend = (
                importlib.import_module("scipy.sparse.linalg")
                if self.sparse
                else importlib.import_module("scipy.linalg")
            )

        elif backend == "gpu":
            self._device_xp_backend = importlib.import_module("cupy")
            self._device_scipy_backend = importlib.import_module("cupyx.scipy")
            self._device_linalg_backend = (
                importlib.import_module("cupyx.scipy.sparse.linalg")
                if self.sparse
                else importlib.import_module("cupy.linalg")
            )
        else:
            msg = f"{backend} is not a valid backend. Choose 'cpu' or 'gpu'."
            raise ValueError(msg)
        if self.default_dtype is None:
            self.default_dtype = self._device_xp_backend.complex128
        elif not self._device_xp_backend.issubdtype(
            self.default_dtype, self._device_xp_backend.inexact
        ):
            self.default_dtype = self._device_xp_backend.complex128

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key == "default_dtype" and not self._device_xp_backend.issubdtype(
                    value, self._device_xp_backend.inexact
                ):
                    msg = """default_dtype must be a type defined in either 
                    device_xp_backend or device_scipy_backend."""
                    raise ValueError(msg)
                setattr(self, key, value)
                self.__post_init__()  # Reinitialize to update dependent fields
            else:
                msg = f"{key} is not a valid configuration key."
                raise KeyError(msg)

    def list_options(self):
        options = ["backend", "sparse", "debug", "default_dtype"]
        for option in options:
            print(f"{option}: {getattr(self, option)}")


qcheff_config = _QCHeffConfig()


@contextmanager
def temp_config(**kwargs):
    """
    Temporarily change settings on the qcheff_config object.
    """
    backup_values = {key: getattr(qcheff_config, key) for key in kwargs}
    qcheff_config.set(**kwargs)
    try:
        yield
    finally:
        qcheff_config.set(**backup_values)
