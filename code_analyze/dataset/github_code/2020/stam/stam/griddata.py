import numpy as np
import scipy.spatial.qhull as qhull

# based on
# https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids


def triangulate(xy):
    """
    triangulate(xy)

    Compute the D-D Delaunay triangulation of a grid.

    Parameters
    ----------
    xy : 2-D ndarray of floats with shape (n, D), or length D tuple of 1-D ndarrays with shape (n,).
        Data point coordinates.

    Returns
    -------
    tri : Delaunay object
        2D Delaunay triangulation of `xy`.
    """

    tri = qhull.Delaunay(xy)
    return tri


def interpolate(values, tri, uv, fill_value=np.nan, d=2):
    """
    interpolate(values, tri, uv, fill_value=np.nan, d=2)

    Interpolate unstructured D-D data.

    Parameters
    ----------
    values : ndarray of float or complex, shape (n,)
        Data values.
    tri : Delaunay object
        2D Delaunay triangulation of the grid.
    uv : 2-D ndarray of floats with shape (m, D), or length D tuple of ndarrays broadcastable to the same shape.
        Points at which to interpolate data.
    fill_value : float, optional
        Value used to fill in for requested points outside of the convex hull of the input points (default: `np.nan`).
    d : int, optional
        Number of dimensions (default: 2).

    Returns
    -------
    ret: ndarray
        Array of interpolated values.
    """

    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    ret = np.einsum('nj,nj->n', np.take(values, vertices), weights)
    ret[np.any(weights < 0, axis=1)] = fill_value

    return ret


def triangulate_tracks(tracks, xparam="color", yparam="absmag"):
    """
    triangulate_tracks(tracks, color="bp_rp", mag="mg")

    Compute the 2D Delaunay triangulation of the stellar track color-magnitude coordinates.

    Parameters
    ----------
    tracks : Table
        Stellar-track grid, as retrieved by `stam.tracks.get_isomasses` or `stam.tracks.get_combined_isomasses`.
    xparam : str, optional
        x-axis parameter (usually the color of the stars, e.g. Gaia's Gbp-Grp; default: "color").
    yparam : str, optional
        y-axis parameter (usually the absolute magnitude of the stars, e.g. Gaia's M_G; same size as `x`; default: "absmag").

    Returns
    -------
    tri : Delaunay object
        2D Delaunay triangulation of the stellar track grid.
    """

    track_points = np.array([tracks[xparam], tracks[yparam]]).T
    tri = triangulate(track_points)
    return tri


def interpolate_tracks(tri, color, mag, tracks, param="mass"):
    """
    interpolate_tracks(tri, color, mag, tracks, param="mass")

    Interpolate color-magnitude location over the stellar track grid.

    Parameters
    ----------
    tri : Delaunay object
        2D Delaunay triangulation of the stellar track grid, as retrieved by `stam.interp.triangulate_tracks`.
    color : array_like
        Color of the stars (usually Gaia's Gbp-Grp).
    mag : array_like
        Absolute magnitude of the stars (usually Gaia's M_G; same size as `color`).
    tracks : Table
        Stellar-track grid, as retrieved by `stam.tracks.get_isomasses` or `stam.tracks.get_combined_isomasses`.
    param : str, optional
        The parameter to evaluate (options: "mass", "age", "mh"; default: "mass").

    Returns
    -------
    x : array_like
        The evaluated parameter.
    nan_idx : array_like
        NaN-value indices of `x`.
    """

    obs_points = np.array([color, mag]).T
    x = interpolate(tracks[param], tri, obs_points)
    nan_idx = np.isnan(x)

    return x, nan_idx
