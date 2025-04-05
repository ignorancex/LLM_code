import numpy as np


def change_grid(small_grid, increase_factor):
    """
    Changes from the small grid to a big grid. The increase is given by the increase factor.

    For example, if we have a [200,200] image and use an increase_factor of 3, an array with the shape [600,600]
    would be returned.
    Parameters
    ----------
    small_grid:
        Base array
    increase_factor:
        How much we should increase the array

    Returns
    -------

    """

    tiled = np.repeat(small_grid, increase_factor, axis=0)
    tiled = np.repeat(tiled, increase_factor, axis=1)

    return tiled
