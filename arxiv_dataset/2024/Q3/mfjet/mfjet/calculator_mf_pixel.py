
import numpy as np
import scipy.sparse
import scipy.signal

from .calculator_mf_manhattan import MFManhattanCalculator
from . import pixel_utils

class MFPixelCalculator:
    """
    Minkowski functional calculator for the persistent analysis with 
    Steiner-type formula in Manhattan geometry. 
    This module is for pixelerated binary image analysis.
    This module is using shapely, and faster than marching square algorithm.

    Parameters
    ----------
    bin_width : float, default 1.
        pixel width
    eps : float, default 1e-6
        epsilon parameter for determining points on the closed boundary of given pixel.
        For a pixel [x0,x1) x [y0,y1), the tolerance of covering points on the closed boundary will be 
            (x-x0) >= -(bin_width) * (eps), (y-y0) >= -(bin_width) * (eps).
    mode : str, default is "center"
        If mode is "center", pixel's center will be used as its representative coordinate.
        For example, the origin (0,0) will be covered by a pixel [-bin_width/2, bin_width/2)x[-bin_width/2, bin_width/2)

        If mode is "corner", pixel's left bottom corner will be used as its representative coordinate.
        For example, the origin (0,0) will be covered by a pixel [0, bin_width)x[0, bin_width/2)

    diagonal_connected : bool, default False:
        If true, diagonally connected pixels will be considered as a connected piece. 
        This flag will affect only calculation of Euler characteristics.

    """

    def __init__(self, bin_width=1., eps=1e-6, mode="center", diagonal_connected=False):
        self.bin_width = bin_width
        self.mode = mode

        if mode == "center":
            self.offset = -bin_width * 0.5
        elif mode == "corner":
            self.offset = 0
        else:
            raise ValueError(f"unknown mode: {mode}")
        self.eps = eps

        if diagonal_connected:
            raise NotImplementedError("connected diagonal is not implemented yet. Please use MFPixelCalculatorMarchingSquare")

        self.calc = MFManhattanCalculator()

    def calc_mfs(self, coords, r):
        """
        Compute MFs given points dilated by a square with half-width r.
        This function automatically takes care of pixelation of coords and discretization of dialation scale.
        The dilation scale r will be discretized as follows:
            r -> Ceil[r / (bin_width*0.5)] * (bin_width*0.5)

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
        buf_coords = pixel_utils.coords_to_binned_coords(coords, bin_width=self.bin_width, eps=self.eps, mode=self.mode)
        half_width = self.bin_width * 0.5
        buf_r = np.ceil((r + half_width * self.eps)/ half_width) * half_width
        return self.calc.calc_mfs(buf_coords, buf_r)

class MFPixelCalculatorMarchingSquare:
    """
    Minkowski functional calculator for the persistent analysis with
    Steiner-type formula in Manhattan geometry.
    This module is for pixelerated binary image analysis.
    This module uses marching square algorithm instead of shapely.

    Parameters
    ----------
    bin_width : float, default 1.
        pixel width
    eps : float, default 1e-6
        epsilon parameter for determining points on the closed boundary of given pixel.
        For a pixel [x0,x1) x [y0,y1), the tolerance of covering points on the closed boundary will be
            (x-x0) >= -(bin_width) * (eps), (y-y0) >= -(bin_width) * (eps).
    mode : str, default is "center"
        If mode is "center", pixel's center will be used as its representative coordinate.
        For example, the origin (0,0) will be covered by a pixel [-bin_width/2, bin_width/2)x[-bin_width/2, bin_width/2)

        If mode is "corner", pixel's left bottom corner will be used as its representative coordinate.
        For example, the origin (0,0) will be covered by a pixel [0, bin_width)x[0, bin_width/2)

    diagonal_connected : bool, default False:
        If true, diagonally connected pixels will be considered as a connected piece.
        This flag will affect only calculation of Euler characteristics.

    """

    def __init__(self, bin_width=1., eps=1e-6, mode="center", diagonal_connected=False):
        self.bin_width = bin_width

        if mode == "center":
            self.offset = -bin_width * 0.5
        elif mode == "corner":
            self.offset = 0
        else:
            raise ValueError(f"unknown mode: {mode}")
        self.eps = eps


        self.filter_binary = np.array([[1,2],[4,8]], dtype=int)

        # lookup table for (local MF * 4)
        self.lookup_mf_local = np.array([
            [0,0,0],
            [1,4,1],
            [1,4,1],
            [2,4,0],
            [1,4,1],
            [2,4,0],
            [2,8,-2] if diagonal_connected else [2,8,2],
            [3,4,-1],
            [1,4,1],
            [2,8,-2] if diagonal_connected else [2,8,2],
            [2,4,0],
            [3,4,-1],
            [2,4,0],
            [3,4,-1],
            [3,4,-1],
            [4,0,0]
        ], dtype=int)[:,::-1]

    def calc_mfs(
            self,
            coords=None, r=None,
            img=None,
    ):
        """
        Compute MFs given points dilated by a square with half-width r.
        This function automatically takes care of pixelation of coords and discretization of dialation scale.
        The dilation scale r will be discretized as follows:
            r -> Ceil[r / (bin_width*0.5)] * (bin_width*0.5)

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
        if img is None:
            img = self.coord_to_img(coords)
        if np.ndim(r) == 0:
            if r == 0:
                #npt = coords.shape[0]
                #return np.array([npt,0.,0.])
                return self.calc_mfs_from_img(img)
            else:
                square_width = np.ceil((r + self.bin_width*self.eps) / (self.bin_width * 0.5)).astype(int)
                if square_width == 0:
                    return self.calc_mfs_from_img(img)
                else:
                    img_dilated = self.dilate_img_by_square(img, square_width)
                    return self.calc_mfs_from_img(img_dilated)
        else:
            return np.stack(
                [
                    self.calc_mfs(img=img, r=this_r)
                    for this_r in r
                ],
                axis=0
            )

    def coord_to_img(self, coord):
        bin_idx = np.floor((coord - self.offset + self.bin_width * self.eps) / self.bin_width).astype(int)
        bin_idx_min = bin_idx.min(axis=0)
        img = scipy.sparse.coo_array(
            (
                np.ones(coord.shape[0], dtype=int),
                (bin_idx - bin_idx_min).T
            )
        ).astype(bool).astype(int).toarray()
        return img

    def dilate_img_by_square(self, img, square_width):
        filter_square = np.ones((square_width, square_width), dtype=int)
        img_dilated = scipy.signal.convolve(img, filter_square).astype(bool).astype(int)
        return img_dilated

    def calc_mfs_from_img(self, img):
        subimage_binary_encoding = scipy.signal.convolve(
            img,
            self.filter_binary
        )
        img_mfs_local = self.lookup_mf_local[subimage_binary_encoding]
        arr_mfs = img_mfs_local.sum(axis=-2).sum(axis=-2) // 4
        arr_mfs = arr_mfs * np.array([1., self.bin_width, self.bin_width**2])# count bin width
        return arr_mfs

