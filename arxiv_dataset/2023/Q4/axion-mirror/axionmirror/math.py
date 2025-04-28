"""Math functions."""

import numpy as np


def left_geom_trapz(func, start, end, n_points=1000, offset=1):
    """Fine grained near start. Assumes start < end, vectorized func."""
    if not start < end:
        raise ValueError('Must have start < end')
    x = np.geomspace(offset, end-start+offset, n_points) - offset + start
    y = func(x)
    if np.isnan(x).any():
        raise ValueError('Ill-defined x.')
    return np.trapz(y, x)

def right_geom_trapz(func, start, end, n_points=1000, offset=1):
    """Fine grained near end. Assumes start < end, vectorized func."""
    if not start < end:
        raise ValueError('Must have start < end')
    x = end + offset - np.geomspace(offset, end-start+offset, n_points)[::-1]
    y = func(x)
    if np.isnan(x).any():
        raise ValueError('Ill-defined x.')
    return np.trapz(y, x)