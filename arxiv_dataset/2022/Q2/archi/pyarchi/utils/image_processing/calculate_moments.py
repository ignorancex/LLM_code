import numpy as np
from cv2 import moments


def calculate_moments(contours, bg_grid):
    """
    Apply image moments, as described in `OpenCv docs <https://docs.opencv.org/3.4.2/dd/d49/tutorial_py_contour_features.html>`_

       
    Parameters
    ----------
    contours : list
        List with calculated masks
    
    bg_grid : int
        Size of the background grid

    Returns
    -------
    centers
        Position of each detected star
    distances
        Distance of each star to the image center -> position [100,100]
    """

    scaling_factor = bg_grid / 200 if bg_grid else 1

    center = [100, 100]
    if bg_grid != 0:
        center = np.multiply(center, scaling_factor) + np.floor(scaling_factor / 2)

    centers = []
    distances = []
    for j in range(len(contours)):
        M = moments(contours[j])
        cY = M["m10"] / M["m00"]
        cX = M["m01"] / M["m00"]

        centers.append([cX, cY])
        distances.append(np.sqrt((cX - center[0]) ** 2 + (cY - center[1]) ** 2))

    return centers, distances
