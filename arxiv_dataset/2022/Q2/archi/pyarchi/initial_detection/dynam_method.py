import numpy as np
import cv2

from pyarchi.data_objects import Star
from pyarchi.utils import shape_analysis


def initial_dynam_centers(img, bg_grid, **kwargs):
    """

    Uses the same method as the one in dynamical_centers to detect the original position of each star.
    Afterwards, the positions are ordered by closeness to the center which allows us to keep the same
    order as in the "fits" initial position method.
    Parameters
    ----------
    im:
        First image in the data set
    bg_grid:
        Size of the background grid
    Returns
    -------

    """

    _, centers, distances = shape_analysis(img, bg_grid, kwargs['repeat_removal'])
    stars = []

    for dist in sorted(distances):
        positions = centers[distances.index(dist)]
        stars.append(Star(kwargs["CDPP_type"], positions, dist))
    
    return stars


if __name__ == "__main__":
    pass

