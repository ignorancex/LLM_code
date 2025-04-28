""" A module containing MF calculators for Euclidean geometry

"""

import numpy as np
import shapely
import shapely.ops

from . import minkowski_funcs

class MFEuclideanCalculator:
    """
    Minkowski functional calculator for the persistent analysis with 
    Steiner-type formula in Euclidean geometry.

    Parameters
    ----------
    quad_segs : int, default 8
        The default number of linear segments in a quarter circle 
        in the approximation of circular arcs. 
        This number will be used as quad_segs param in shapely.buffer method 
        if quad_segs is not provided to member funtions.


    """
    def __init__(self, quad_segs=8):
        self.quad_segs=quad_segs

    def calc_mfs(self, coords, r, quad_segs=None):
        """
        Compute MFs given points dilated by a disk with radius r.

        Parameters
        ----------
        coords    : array_like with shape (N_pt, 2)
        r         : float or array_like
            The circle radius in the Minkowski sum.
        quad_segs : int, default 8
            The number of linear segments in a quarter circle in 
            the approximation of circular arcs.

        Returns
        -------
        np.array with shape (3,) or (r.shape, 3)
            array of Minkowski functionals given r. 
            The last index k is labeling $k$-th Minkiwski functionals
                - k=0: Euler characteristic
                - k=1: Boundary length
                - k=2: Area

        """
        if np.ndim(r) == 0:
            if r == 0:
                npt = coords.shape[0]
                return np.array([npt,0.,0.])
            else:
                geom = self.dilate_points_by_disk(coords, r, quad_segs)
                return minkowski_funcs.calc_mfs(geom)
        else:
            return np.stack(
                [
                    self.calc_mfs(coords, this_r, quad_segs)
                    for this_r in r
                ],
                axis=0
            )

    def dilate_points_by_disk(self, coords, r, quad_segs=None):
        """
        Dilate given points by a disk with radius r.
        This dilation function is for Steiner-type formula for Euclidean distance.

        Parameters
        ----------
        coords    : array_like with shape (N_pt, 2)
        r         : float
            The circle radius in the Minkowski sum.
        quad_segs : int, default 8
            The number of linear segments in a quarter circle in 
            the approximation of circular arcs.

        Returns
        -------
        shapely.Polygon or shapely.MultiPolygon
            geometry object representing dilated points

        """
        list_dilated_points  = [
            shapely.geometry.Point(
                *coord
            ).buffer(
                r,
                cap_style='round',
                quad_segs=self.quad_segs if quad_segs is None else quad_segs
            )
            for coord in coords
        ]
        geom_dilated_points = shapely.ops.unary_union(list_dilated_points)
        return geom_dilated_points
