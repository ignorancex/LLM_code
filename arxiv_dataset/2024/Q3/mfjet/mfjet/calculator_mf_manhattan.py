"""

"""
import math

import numpy as np
import shapely
import shapely.ops

from . import minkowski_funcs

class MFManhattanCalculator:
    """
    Minkowski functional calculator for the persistent analysis with 
    Steiner-type formula in Manhattan geometry. 

    """
    def __init__(self):
        pass

    def calc_mfs(self, coords, r):
        """
        Compute MFs given points dilated by a square with half-width r.

        Parameters
        ----------
        coords    : array_like with shape (N_pt, 2)
        r         : float or array_like
            Specifies the half-width of a square in the Minkowski sum.

        Returns
        -------
        np.array with shape (3,) or (r.shape, 3)
            array of Minkowski functionals given r. 
            The last index k is labeling $k$-th Minkiwski functionals
               k=0: Euler characteristic
               k=1: Boundary length
               k=2: Area

        """
        if np.ndim(r) == 0:
            if r == 0:
                npt = coords.shape[0]
                return np.array([npt,0.,0.])
            else:
                geom = self.dilate_points_by_square(coords, r)
                return minkowski_funcs.calc_mfs(geom)
        else:
            return np.stack(
                [
                    self.calc_mfs(coords, this_r)
                    for this_r in r
                ],
                axis=0
            )

    def dilate_points_by_square(self, coords, r):
        """
        Dilate given points by a square with half-width r.
        This dilation function is for Steiner-type formula for Euclidean distance.

        Parameters
        ----------
        coords    : array_like with shape (N_pt, 2)
        r         : float
            Specifies the half-width of a square in the Minkowski sum.

        Returns
        -------
        shapely.Polygon or shapely.MultiPolygon
            geometry object representing dilated points

        """
        list_dilated_points  = [
            shapely.geometry.Point(*coord).buffer(r, cap_style="square", join_style="mitre", mitre_limit=math.inf)
            for coord in coords
        ]
        geom_dilated_points = shapely.ops.unary_union(list_dilated_points)
        return geom_dilated_points

