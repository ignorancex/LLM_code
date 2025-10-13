import unittest

from jax import Array
from jax import numpy as np


class TestCase(unittest.TestCase):
    def assertEqualJax(self, a: Array | float, b: Array | float, msg: str | None = None):
        if not np.all(a == b):
            self.fail(msg or f"Arrays are not equal:\n{a}\n !=\n{b}")

    def assertShape(self, arr: Array, shape: tuple[int, ...]):
        if arr.shape != shape:
            self.fail(f"Array shape is {arr.shape}, but expected {shape}.")

    def assertAlmostEqualJax(self, a: Array | float, b: Array | float, delta: float = 1e-3, msg: str | None = None):
        if not np.all(np.abs(a - b) < delta):
            max_delta = np.max(np.abs(a - b))
            self.fail(f"Arrays are not equal. Allowed delta: {delta}, maximum delta: {max_delta}.\n{a}\n != (within delta)\n{b}")

    def assertDtypesEqualJax(self, a: Array, b: Array):
        if not a.dtype == b.dtype:
            self.fail(f"Array dtypes are not equal. {a.dtype} != {b.dtype}")
