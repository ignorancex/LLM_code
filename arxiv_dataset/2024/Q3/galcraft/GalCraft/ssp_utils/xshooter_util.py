import glob, re

import numpy as np
from scipy import interpolate
from astropy.io import fits


def age_metal(filename):
    """
    Extract age (T) and metallicity (Z or MH) values from the given filename.
    Supports formats:
    - 'XSL_SSP_logT<logT>_MH<MH>_Kroupa_PC.fits'
    - 'XSL_SSP_T<age>_Z<Z>_Kroupa_P00.fits'
    """
    pattern1 = r'XSL_SSP_logT(?P<logT>[-+]?[0-9]*\.?[0-9]+)_MH(?P<MH>[-+]?[0-9]*\.?[0-9]+)_.+\.fits'
    pattern2 = r'XSL_SSP_T(?P<T>[-+]?[0-9]*\.?[0-9]+e?[+-]?[0-9]*)_Z(?P<Z>[-+]?[0-9]*\.?[0-9]+)_.+\.fits'

    match1 = re.search(pattern1, filename)
    match2 = re.search(pattern2, filename)

    if match1:
        return 10 ** float(match1.group('logT')) / 1e9, float(match1.group('MH'))
    elif match2:
        return float(match2.group('T')) / 1e9, float(match2.group('Z'))
    else:
        return None

class xshooter:

    def __init__(self, pathname, FWHM_tem='lsf_xshooter', age_range=None,
                 metal_range=None):

        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname

        all = [age_metal(f) for f in files]
        all_ages, all_metals = np.array(all).T
        ages, metals = np.unique(all_ages), np.unique(all_metals)
        n_ages, n_metal = len(ages), len(metals)

        assert set(all) == set([(a, b) for a in ages for b in metals]), \
            'Ages and Metals do not form a Cartesian grid'

        if '_KU_' in files[0]:
            flux_factor = 5567946.09
        elif '_SA_' in files[0]:
            flux_factor = 9799552.50
        else:
            raise ValueError('initial-mass function is unknown, can not calculate flux unit factor')

        hdu = fits.open(files[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam = 10 ** (h2['CRVAL1'] + np.arange(h2['NAXIS1']) * h2['CDELT1']) * 10

        templates = np.empty((ssp.size, n_ages, n_metal))
        age_grid = np.empty((n_ages, n_metal))
        metal_grid = np.empty((n_ages, n_metal))
        flux = np.ones((n_ages, n_metal))

        for j, age in enumerate(ages):
            for k, met in enumerate(metals):
                p = all.index((age, met))
                hdu = fits.open(files[p])
                ssp = hdu[0].data
                templates[:, j, k] = ssp
                age_grid[j, k] = age
                metal_grid[j, k] = met

        new_wave = np.arange(lam[0], lam[-1], np.min(np.diff(lam)))
        new_templates = np.zeros([len(new_wave), templates.shape[1], templates.shape[2]])
        for i in range(templates.shape[1]):
            for j in range(templates.shape[2]):
                interpolator = interpolate.interp1d(lam, templates[:, i, j])
                new_templates[:, i, j] = interpolator(new_wave)

        new_templates = new_templates * flux_factor

        lam = new_wave
        templates = new_templates
        new_wave = None
        new_templates = None

        if age_range is not None:
            w = (age_range[0] <= age_grid[:, 0]) & (age_grid[:, 0] <= age_range[1])
            templates = templates[:, w, :]
            age_grid = age_grid[w, :]
            metal_grid = metal_grid[w, :]
            flux = flux[w, :]
            n_ages, n_metal = age_grid.shape

        if metal_range is not None:
            w = (metal_range[0] <= metal_grid[0, :]) & (metal_grid[0, :] <= metal_range[1])
            templates = templates[:, :, w]
            age_grid = age_grid[:, w]
            metal_grid = metal_grid[:, w]
            flux = flux[:, w]
            n_ages, n_metal = age_grid.shape

        self.templates = templates
        self.lam_temp = lam
        self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.n_ages = n_ages
        self.n_metal = n_metal
        self.flux = flux
        self.FWHM_tem = FWHM_tem
