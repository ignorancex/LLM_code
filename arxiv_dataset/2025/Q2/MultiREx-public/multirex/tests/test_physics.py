import numpy as np
import pytest
from multirex import spectra

def test_wavenumber_grid():
    # Verify that a sorted array is generated
    wn = spectra.Physics.wavenumber_grid(1, 10, 100)
    assert isinstance(wn, np.ndarray)
    assert np.all(wn[:-1] <= wn[1:])
    # Check that the values are in the expected approximate range
    expected_min = 10000 / 10  # For wl_max = 10
    expected_max = 10000 / 1   # For wl_min = 1
    assert wn[0] >= expected_min
    assert wn[-1] <= expected_max

def test_generate_value():
    # Fixed value case
    assert spectra.Physics.generate_value(5) == 5
    # Range case (if both ends are equal, the returned value is that value)
    value = spectra.Physics.generate_value((10, 10))
    assert value == 10
    # List case: the returned value should be one of the list elements
    value = spectra.Physics.generate_value([1, 2, 3])
    assert value in [1, 2, 3]
    # None case
    assert spectra.Physics.generate_value(None) is None
