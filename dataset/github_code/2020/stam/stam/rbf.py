from scipy.interpolate import Rbf


def rbfi_tracks(tracks, xparam="color", yparam="absmag", param="mass", **kwargs):
    """
    rbfi_tracks(tracks, xparam="color", yparam="absmag", param="mass")

    Compute the radial basis function interpolator instance of the stellar tracks.

    Parameters
    ----------
    tracks : Table
        Stellar-track grid, as retrieved by `stam.gentracks.get_isomasses`, `stam.gentracks.get_combined_isomasses`, or `stam.get_isotrack`.
    xparam : str, optional
        x-axis parameter (usually the color of the stars, e.g. Gaia's Gbp-Grp; default: "color").
    yparam : str, optional
        y-axis parameter (usually the absolute magnitude of the stars, e.g. Gaia's M_G; same size as `x`; default: "absmag").
    param : str, optional
        The parameter to evaluate (options: "mass", "age", "mh", "color", "absmag"; default: "mass").

    Returns
    -------
    rbfi : Rbf interpolator instance
        Radial basis function interpolator instance based on the stellar tracks.
    """

    rbfi = Rbf(tracks[xparam], tracks[yparam], tracks[param], **kwargs)

    return rbfi


def interpolate_tracks(rbfi, x, y):
    """
    interpolate_tracks(tri, color, mag, tracks, param="mass")

    Interpolate color-magnitude location over the stellar track grid.

    Parameters
    ----------
    rbfi : Rbf interpolator instance
        Radial basis function interpolator instance, as retrieved by `stam.rbf.rbfi_tracks`.
    x : str, optional
        x-axis parameter (usually the color of the stars, e.g. Gaia's Gbp-Grp; default: "color").
    y : str, optional
        y-axis parameter (usually the absolute magnitude of the stars, e.g. Gaia's M_G; same size as `x`; default: "absmag").

    Returns
    -------
    z : array_like
        The evaluated parameter.
    """

    z = rbfi(x, y)

    return z
