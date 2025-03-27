import time
from tqdm import tqdm
import numpy as np
from . import rbf, griddata, nurbs


def assign_param(color, color_error, mag, mag_error, tracks, n_realizations=10, param="mass", interp_fun="rbf",
                 binary_polygon=None, **kwargs):
    """
    assign_param(color, color_error, mag, mag_error, tracks, n_realizations=10, param="mass", interp_fun="rbf",
                 binary_polygon=None, **kwargs)

    Assign a specific parameter (mass or metallicity) to a star, based on its color and magnitude.

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
    tracks : Table
        Stellar-track grid, as retrieved by `stam.tracks.get_isomasses` or `stam.tracks.get_combined_isomasses`.
    n_realizations : int, optional
        Number of realizations to draw for each star, from a 2D-Gaussian distribution around the color-magnitude
        position of the star (default: 10).
    param : str, optional
        Which parameter to estimate (options: "mass", "mh"; default: "mass").
    interp_fun : str, optional
        Which interpolation method to use (options: "rbf", "griddata", "nurbs"; default: "rbf").
    binary_polygon : Path object, optional
        The polygon defining the equal-mass binary region on the HR-diagram (default: None).

    Returns
    -------
    param_mean : array_like
        Estimated parameter mean for each star (same size as `color`).
    param_error : array_like
        Estimated parameter standard deviation for each star (same size as `color`).
    binary_param_mean : array_like, returned only if binary_polygon is not None
        Estimated parameter mean for each star inside binary_polygon, assuming an equal-mass binary (same size as `color`).
    binary_param_error : array_like, returned only if binary_polygon is not None
        Estimated parameter standard deviation for each star inside binary_polygon, assuming an equal-mass binary (same size as `color`).
    weight : array_like, returned only if binary_polygon is not None
        Single-star probability: the fraction of realizations in which the star was *outside* of binary_polygon (same size as `color`).
    """

    if interp_fun not in ["rbf", "griddata", "nurbs"]:
        interp_fun = "rbf"
        print(f"{interp_fun} unknown, using rbf instead.")

    print(f"Using {interp_fun} interpolation...")
    if interp_fun == "rbf":
        rbfi = rbf.rbfi_tracks(tracks, param=param, **kwargs)
    elif interp_fun == "griddata":
        tri = griddata.triangulate_tracks(tracks)
    elif interp_fun == "nurbs":
        surf = nurbs.tracks2surf(tracks, param)

    param_mean = np.zeros(len(color))*np.nan
    param_error = np.zeros(len(color))*np.nan

    if binary_polygon is not None:
        binary_param_mean = np.zeros(len(color))*np.nan
        binary_param_error = np.zeros(len(color))*np.nan
        weight = np.ones(len(color))

    t = time.time()
    for i in tqdm(range(len(color))):  # for each gaia source
        mean = [color[i], mag[i]]
        cov = [[color_error[i], 0], [0, mag_error[i]]]

        x = np.random.multivariate_normal(mean, cov, size=n_realizations)

        if binary_polygon is not None:
            # check which realizations fall inside the binary sequence
            binary_idx = binary_polygon.contains_points(x)
            if np.any(binary_idx):
                # some of the points fall inside the binary sequence, take more
                x_extra = np.random.multivariate_normal(mean, cov, size=n_realizations)
                x = np.concatenate((x, x_extra))
                binary_idx = binary_polygon.contains_points(x)

                # calculate relative single/binary weight
                weight[i] = np.count_nonzero(~binary_idx)/len(binary_idx)  # single-star probability

                # shift the points that fall inside the binary sequence to 2.5log(2)-fainter magnitudes
                x[binary_idx, 1] = x[binary_idx, 1] + 2.5*np.log10(2)

        if interp_fun == "rbf":
            curr_param = rbf.interpolate_tracks(rbfi, x[:, 0], x[:, 1])
        elif interp_fun == "griddata":
            curr_param, nan_idx = griddata.interpolate_tracks(tri, x[:, 0], x[:, 1], tracks, param=param)
        elif interp_fun == "nurbs":
            curr_param = nurbs.evaluate(surf, x[:, 0], x[:, 1])

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

    print(f"Calculating mean took {time.time() - t:.1} sec.")

    if binary_polygon is None:
        return param_mean, param_error
    else:
        # take binary sequence into account
        return param_mean, param_error, binary_param_mean, binary_param_error, weight
