import time
from tqdm import tqdm
import numpy as np
from . import rbf, griddata, nurbs


def assign_param(x, xerror, y, yerror, tracks, n_realizations=10,
                 xparam="color", yparam="absmag", param="mass", interp_fun="rbf",
                 binary_polygon=None, show_progress_bar=True,
                 return_realizations=False, **kwargs):
    """
    assign_param(x, xerror, y, yerror, tracks, n_realizations=10, param="mass", interp_fun="rbf",
                 binary_polygon=None, **kwargs)

    Assign a specific parameter (mass or metallicity) to a star, based on its color and magnitude.

    Parameters
    ----------
    x : array_like
        x-axis parameter (usually the color of the stars, e.g. Gaia's Gbp-Grp).
    xerror : array_like
        `color` uncertainty (same size as `color`).
    y : array_like
        Absolute magnitude of the stars (usually Gaia's M_G; same size as `color`).
    yerror : array_like
        `mag` uncertainty (same size as `color`).
    tracks : Table
        Stellar-track grid, as retrieved by `stam.tracks.get_isomasses` or `stam.tracks.get_combined_isomasses`.
    n_realizations : int, optional
        Number of realizations to draw for each star, from a 2D-Gaussian distribution around the color-magnitude
        position of the star (default: 10).
    xparam : str, optional
        x-axis parameter (default: "color").
    yparam : str, optional
        y-axis parameter (default: "absmag").
    param : str, optional
        Which parameter to estimate (options: "mass", "mh"; default: "mass").
    interp_fun : str, optional
        Which interpolation method to use (options: "rbf", "griddata", "nurbs"; default: "rbf").
    binary_polygon : Path object, optional
        The polygon defining the equal-mass binary region on the HR-diagram (default: None).
    show_progress_bar : bool, optional
        If true, show progress bar (default: True).
    return_realizations : bool, optional
        If true, also returns the individual realizations of `param` (default: `False`).

    Returns
    -------
    param_mean : array_like
        Estimated parameter mean for each star (same size as `x`).
    param_error : array_like
        Estimated parameter standard deviation for each star (same size as `x`).
    binary_param_mean : array_like, returned only if `binary_polygon` is not None
        Estimated parameter mean for each star inside `binary_polygon`, assuming an equal-mass binary (same size as `x`).
    binary_param_error : array_like, returned only if `binary_polygon` is not None
        Estimated parameter standard deviation for each star inside binary_polygon, assuming an equal-mass binary (same size as `x`).
    weight : array_like, returned only if `binary_polygon` is not None
        Single-star probability: the fraction of realizations in which the star was *outside* of binary_polygon (same size as `x`).
    realizations : array_like, returned only if `return_realizations` is `True`
        Individual realizations of `param` (shape `(len(x), n_realizations)`).
    """

    if interp_fun not in ["rbf", "griddata", "nurbs"]:
        interp_fun = "rbf"
        print(f"{interp_fun} unknown, using rbf instead.")

    print(f"Using {interp_fun} interpolation...")
    if interp_fun == "rbf":
        rbfi = rbf.rbfi_tracks(tracks, xparam=xparam, yparam=yparam, param=param, **kwargs)
    elif interp_fun == "griddata":
        tri = griddata.triangulate_tracks(tracks, xparam=xparam, yparam=yparam)
    elif interp_fun == "nurbs":
        surf = nurbs.tracks2surf(tracks, param, xparam=xparam, yparam=yparam)

    param_mean = np.zeros(len(x))*np.nan
    param_error = np.zeros(len(x))*np.nan

    if return_realizations:
        realizations = np.zeros((len(x), n_realizations))*np.nan

    if binary_polygon is not None:
        binary_param_mean = np.zeros(len(x))*np.nan
        binary_param_error = np.zeros(len(x))*np.nan
        weight = np.ones(len(x))

    if show_progress_bar:
        iterations = tqdm(range(len(x)))
    else:
        iterations = range(len(x))

    t = time.time()
    for i in iterations:  # for each gaia source
        try:
            mean = [x[i], y[i]]
            cov = [[xerror[i]**2, 0], [0, yerror[i]**2]]
    
            points = np.random.multivariate_normal(mean, cov, size=n_realizations)
    
            if binary_polygon is not None:
                # check which realizations fall inside the binary sequence
                binary_idx = binary_polygon.contains_points(points)
                if np.any(binary_idx):
                    # some of the points fall inside the binary sequence, take more
                    points_extra = np.random.multivariate_normal(mean, cov, size=n_realizations)
                    points = np.concatenate((points, points_extra))
                    binary_idx = binary_polygon.contains_points(points)
    
                    # calculate relative single/binary weight
                    weight[i] = np.count_nonzero(~binary_idx)/len(binary_idx)  # single-star probability
    
                    # shift the points that fall inside the binary sequence to 2.5log(2)-fainter magnitudes
                    points[binary_idx, 1] = points[binary_idx, 1] + 2.5*np.log10(2)
    
            if interp_fun == "rbf":
                curr_param = rbf.interpolate_tracks(rbfi, points[:, 0], points[:, 1])
            elif interp_fun == "griddata":
                curr_param, nan_idx = griddata.interpolate_tracks(tri, points[:, 0], points[:, 1], tracks, param=param)
            elif interp_fun == "nurbs":
                curr_param = nurbs.evaluate(surf, points[:, 0], points[:, 1])
    
            if binary_polygon is None:
                # don't take binary sequence into account
                param_mean[i] = np.nanmean(curr_param)
                param_error[i] = np.nanstd(curr_param)
            else:
                # single-star parameter estimation
                param_mean[i] = np.nanmean(curr_param[~binary_idx])
                param_error[i] = np.nanstd(curr_param[~binary_idx])
    
                # binary twin parameter estimation
                binary_param_mean[i] = np.nanmean(curr_param[binary_idx])
                binary_param_error[i] = np.nanstd(curr_param[binary_idx])
    
            if return_realizations:
                realizations[i, :] = curr_param
        except ValueError as e:
            print(f"ValueError occurred for index {i}: {e}")

    print(f"Calculating mean took {time.time() - t:.1f} sec.")

    if binary_polygon is None:
        output = [param_mean, param_error]
    else:
        # take binary sequence into account
        output = [param_mean, param_error, binary_param_mean, binary_param_error, weight]

    if return_realizations:
        output.append(realizations)

    return output


def assign_score_based_on_cmd_position(color, color_error, mag, mag_error, polygon, n_realizations=10, show_progress_bar=True):
    """
    assign_score_based_on_cmd_position(color, color_error, mag, mag_error, polygon, n_realizations=10)

    Assign a star to a polygon based on its color and magnitude (taking uncertainties into account).

    Parameters
    ----------
    color : array_like
        Color of the stars (usually Gaia's Gbp-Grp).
    color_error : array_like
        `color` uncertainty (same size as `color`).
    mag : array_like
        Absolute magnitude of the stars (usually Gaia's M_G; same size as `color`).
    mag_error : array_like
        `mag` uncertainty (same size as `color`).
    polygon : Path object
        A polygon defining a location on the CMD.
    n_realizations : int, optional
        Number of realizations to draw for each star, from a 2D-Gaussian distribution around the color-magnitude
        position of the star (default: 10).
    show_progress_bar : bool, optional
        If true, show progress bar (default: True).

    Returns
    -------
    score : array_like

    """
    score = np.zeros(len(color))

    if show_progress_bar:
        iterations = tqdm(range(len(color)))
    else:
        iterations = range(len(color))
    for i in iterations:  # for each gaia source
        # skip if NaN
        if np.any(np.isnan([color[i], mag[i]])):
            score[i] = np.nan
            continue
        mean = [color[i], mag[i]]
        cov = [[color_error[i], 0], [0, mag_error[i]]]

        x = np.random.multivariate_normal(mean, cov, size=n_realizations)
        idx = polygon.contains_points(np.array([x[:, 0], x[:, 1]]).T)
        score[i] = np.count_nonzero(idx)/n_realizations

    return score


def assign_to_polygon(color, color_error, mag, mag_error, polygon, n_realizations=10, thresh=0.5, show_progress_bar=True):
    """
    assign_to_polygon(color, color_error, mag, mag_error, polygon, n_realizations=10, thresh=0.5)

    Assign a star to a polygon based on its color and magnitude (taking uncertainties into account).

    Parameters
    ----------
    color : array_like
        Color of the stars (usually Gaia's Gbp-Grp).
    color_error : array_like
        `color` uncertainty (same size as `color`).
    mag : array_like
        Absolute magnitude of the stars (usually Gaia's M_G; same size as `color`).
    mag_error : array_like
        `mag` uncertainty (same size as `color`).
    polygon : Path object
        A polygon defining a location on the CMD.
    n_realizations : int, optional
        Number of realizations to draw for each star, from a 2D-Gaussian distribution around the color-magnitude
        position of the star (default: 10).
    thresh : float, optional
        Polygon assignment minimal threshold score (default: 0.5).
    show_progress_bar : bool, optional
        If true, show progress bar (default: True).

    Returns
    -------
    inside_idx : array_like
        Indices of stars inside the polygon.
    """

    score = assign_score_based_on_cmd_position(color, color_error, mag, mag_error, polygon,
                                               n_realizations=n_realizations, show_progress_bar=show_progress_bar)
    inside_idx = score >= thresh

    return inside_idx
