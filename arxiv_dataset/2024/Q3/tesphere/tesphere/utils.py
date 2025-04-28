import numpy as np
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq


def integrate_los(f, rprj, rmax_sph, epsrel=1e-2):
    """Calculate column density

    Parameters
    ----------
    f : function
        The function that returns the volume density at a given spherical
        radius.
    rprj : float
        Projected radius at which the line-of-sight integration is performed.
    rmax_sph : float
        The maximum spherical radius to integrate out.

    Returns
    -------
    dcol : float
        Column density.
    """
    def func(z, rprj):
        r = np.sqrt(rprj**2 + z**2)
        return f(r)
    if isinstance(rprj, np.ndarray):
        dcol = []
        for R in rprj:
            zmax = np.sqrt(rmax_sph**2 - R**2)
            res, _ = quad(func, 0, zmax, args=(R,), epsrel=epsrel, limit=200)
            dcol.append(2*res)
        dcol = np.array(dcol)
    else:
        zmax = np.sqrt(rmax_sph**2 - rprj**2)
        res, _ = quad(func, 0, zmax, args=(rprj,), epsrel=epsrel, limit=200)
        dcol = 2*res
    return dcol


def integrate_2d_projected(f, rmax_prj, rmax_sph, epsrel=1e-2):
    """Calculate 2D integral over the projected area

    f is a function of the spherical radius r. This function integrates
    f over the quasi-cylindrical volume extended along z direction and
    bracketed within [0, rmax_pj] along R direction. The maximum extension
    along z direction depends on R such that z_max = sqrt(rmax_sph^2 - R^2).

    Parameters
    ----------
    f : function
        The function that returns the volume density at a given spherical
        radius.
    rmax_prj : float
        Maximum radial extent in the projected plane.
    rmax_sph : float
        The maximum spherical radius to integrate out.

    Returns
    -------
    dcol : float
        Column density.
    """
    def zmax(rprj):
        return np.sqrt(rmax_sph**2 - rprj**2)

    def func(z, rprj):
        r = np.sqrt(rprj**2 + z**2)
        return f(r)*2*np.pi*rprj
    if isinstance(rmax_prj, np.ndarray):
        res = []
        for R in rmax_prj:
            y1, _ = dblquad(func, 0, R, lambda x: 0, zmax, epsrel=epsrel)
            res.append(y1)
        res = np.array(res)
    else:
        res, _ = dblquad(func, 0, rmax_prj, lambda x: 0, zmax, epsrel=epsrel)
    return 2*res


def fwhm(f, rmax, which='volume', epsrel=1e-2):
    """Calculate the FWHM of the column density profile

    Caution: This does not necessarily find the "first" root, such that
    it might overestimate fwhm if there are multiple crossings. It is
    user's responsibility to provide appropriate rmax.

    Parameters
    ----------
    f : function
        The function that returns the volume/column density at a given
        spherical radius.
        Either rho(r) or Sigma(r) depending on `which`.
    rmax : float
        If which='volume', the maximum spherical radius to integrate out.
        If which='column', the maximum radius to bracket the root.
    which : str, optional
        If the input is volume density, `volume`
        If the input is column density, `column`

    Returns
    -------
    fwhm : float
        The FWHM of the column density profile.
    """
    if which == 'volume':
        dcol0 = integrate_los(f, 0, rmax, epsrel=epsrel)
        fwhm = 2*brentq(lambda x: integrate_los(f, x, rmax, epsrel=epsrel)
                        - 0.5*dcol0, 0, rmax)
    elif which == 'column':
        dcol0 = f(0)
        fwhm = 2*brentq(lambda x: f(x) - 0.5*dcol0, 0, rmax)
    else:
        raise ValueError("which must be either volume or column")
    return fwhm


def fwhm_bgrsub(f, rmax_sph, method='iterative', rbgr=np.nan, epsrel=1e-2):
    """Calculate background-subtracted FWHM

    Iterate until dcol = 0 at r=R_FWHM.

    Parameters
    ----------
    f : function
        The function that returns the volume density at a given
        spherical radius.
    rmax_sph : float
        The maximum spherical radius to integrate out.

    Returns
    -------
    fwhm : float
        The FWHM of the column density profile.
    """

    def dcol(r):
        """Return column density at radius r"""
        res = integrate_los(f, r, rmax_sph, epsrel=epsrel/2)
        return res

    match method:
        case 'radius':
            if np.isnan(rbgr):
                raise ValueError("radius based background subtraction requires"
                                 " `rbgr` parameter")
            dcol_bgr = dcol(rbgr)
            rfwhm = fwhm(lambda x: dcol(x) - dcol_bgr, rmax_sph,
                         which='column', epsrel=epsrel/2)
        case 'iterative':
            rfwhm = fwhm(f, rmax_sph, which='volume', epsrel=epsrel/2)
            rfwhm0 = 1e100

            while np.abs((rfwhm - rfwhm0)/rfwhm0) > epsrel:
                rfwhm0 = rfwhm
                dcol_bgr = dcol(rfwhm)
                rfwhm = fwhm(lambda x: dcol(x) - dcol_bgr, rmax_sph,
                             which='column', epsrel=epsrel/2)
        case _:
            raise ValueError(f"method {method} is not supported")

    return rfwhm
