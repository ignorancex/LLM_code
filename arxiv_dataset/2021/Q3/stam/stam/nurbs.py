import numpy as np
from geomdl import BSpline, knotvector


def tracks2surf(tracks, param, delta=0.01, xparam="color", yparam="absmag", xstep=0.1, ystep=0.1):
    """
    tracks2surf(tracks, param, delta=0.01, xparam="color", yparam="absmag", xstep=0.1, ystep=0.1)

    Compute a NURBS surface from a set of stellar evolution tracks, using the `geomdl` package.

    Parameters
    ----------
    tracks : Table
        Stellar-track grid, as retrieved by `stam.gentracks.get_isomasses` or `stam.gentracks.get_combined_isomasses`.
    param : str
        The parameter to evaluate (options: "mass", "age", "mh").
    delta : float, optional
        `geomdl` surface evaluation step size.
    xparam : str, optional
        x-axis parameter (default: "color").
    yparam : str, optional
        y-axis parameter (default: "absmag").
    xstep : float, optional
        x-axis grid step (default: 0.1).
    ystep : float, optional
        y-axis grid step (default: 0.1).

    Returns
    -------
    surf : `geomdl.NURBS.Surface` object
        2D NURBS surface based on `tracks`.
    """

    xmin = np.min(np.around(tracks[xparam], -int(np.round(np.log10(xstep)))))
    xmax = np.max(np.around(tracks[xparam], -int(np.round(np.log10(xstep)))))
    ymin = np.min(np.around(tracks[yparam], -int(np.round(np.log10(ystep)))))
    ymax = np.max(np.around(tracks[yparam], -int(np.round(np.log10(ystep)))))
    x, y = np.meshgrid(np.arange(xmin, xmax, xstep), np.arange(ymin, ymax, ystep))
    z = np.zeros_like(x)*np.nan
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            dist = np.sqrt((x[i, j] - tracks[xparam])**2 + (y[i, j] - tracks[yparam])**2)
            min_idx = np.argmin(dist)
            z[i, j] = tracks[param][min_idx]

    ctrlpts = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    ctrlpts = ctrlpts.reshape((x.shape[0], x.shape[1], 3))
    ctrlpts = ctrlpts.tolist()

    surf = BSpline.Surface()
    surf.degree_u = 3
    surf.degree_v = 3
    surf.ctrlpts2d = ctrlpts

    surf.knotvector_u = knotvector.generate(surf.degree_u, surf.ctrlpts_size_u)
    surf.knotvector_v = knotvector.generate(surf.degree_v, surf.ctrlpts_size_v)

    surf.delta = delta

    return surf


def evaluate(surf, x, y):
    """
    evaluate(surf, x, y)

    Evaluate NURBS surface at a specific set of coordinates.

    Parameters
    ----------
    surf : `geomdl.NURBS.Surface` object
        2D NURBS surface based on `tracks`.
    x : array_like
        x-axis evaluation points.
    y : array_like
        y-axis evaluation points (same size as `x`).

    Returns
    -------
    z: array_like
        Array of evaluated values (same size as `x`).
    """

    z = np.nan*np.zeros_like(x)  # initialize

    # calculate (u, v) coordinates on surface
    v = (x - surf.bbox[0][0]) / (surf.bbox[1][0] - surf.bbox[0][0])
    u = (y - surf.bbox[0][1]) / (surf.bbox[1][1] - surf.bbox[0][1])
    uv = np.array([u, v]).T

    # evaluate only points inside surface (points outside surface can't be evaluated)
    idx = ((0 <= v) & (v <= 1)) & ((0 <= u) & (u <= 1))
    if np.any(idx):
        z[idx] = np.array(surf.evaluate_list(uv[idx, :].tolist()))[:, -1]

    return z
