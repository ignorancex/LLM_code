import glob, re
import numpy as np
from astropy.io import fits




def age_metal(filename):
    """
    Extract the age and metallicity from the name of a file of
    the MILES library of Single Stellar Population models as
    downloaded from http://miles.iac.es/ as of 2022

    This function relies on the MILES file containing a substring of the
    precise form like Zm0.40T00.0794, specifying the metallicity and age.

    :param filename: string possibly including full path
        (e.g. 'miles_library/Eun1.30Zm0.40T00.0794.fits')
    :return: age (Gyr), [M/H]

    """
    # s = re.findall(r'Z[m|p][0-9]\.[0-9]{2}T[0-9]{2}\.[0-9]{4}', filename)[0]
    s = re.findall(r'Z[m|p]\d+\.\d+T\d+\.\d+', filename)[0]
    split_result = s.split("T")
    metal = split_result[0]
    age = float(split_result[1])
    if "Zm" in metal:
        metal = -float(metal[2:])
    elif "Zp" in metal:
        metal = float(metal[2:])

    return age, metal




class miles:


    def __init__(self, pathname, FWHM_tem=2.51, age_range=None,
                 metal_range=None):

        files = glob.glob(pathname)
        assert len(files) > 0, "Files not found %s" % pathname

        all = [age_metal(f) for f in files]
        all_ages, all_metals = np.array(all).T
        ages, metals = np.unique(all_ages), np.unique(all_metals)
        n_ages, n_metal = len(ages), len(metals)

        assert set(all) == set([(a, b) for a in ages for b in metals]), \
            'Ages and Metals do not form a Cartesian grid'

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the galaxy spectrum, to determine the
        # size needed for the array which will contain the template spectra.
        hdu = fits.open(files[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam = h2['CRVAL1'] + np.arange(h2['NAXIS1']) * h2['CDELT1']
        # lam_range_temp = lam[[0, -1]]
        # ssp_new, ln_lam_temp = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[:2]
        # ssp_new = ssp

        # if norm_range is not None:
        #     norm_range = np.log(norm_range)
        #     band = (norm_range[0] <= lam) & (lam <= norm_range[1])

        templates = np.empty((ssp.size, n_ages, n_metal))
        age_grid, metal_grid, flux = np.empty((3, n_ages, n_metal))

        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the galaxy and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels Vazdekis --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        # if FWHM_gal is not None:
        #     FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
        #     sigma = FWHM_dif / 2.355 / h2['CDELT1']  # Sigma difference in pixels

        # Here we make sure the spectra are sorted in both [M/H] and Age
        # along the two axes of the rectangular grid of templates.
        for j, age in enumerate(ages):
            for k, met in enumerate(metals):
                p = all.index((age, met))
                hdu = fits.open(files[p])
                ssp = hdu[0].data
                # if FWHM_gal is not None:
                #     if np.isscalar(FWHM_gal):
                #         if sigma > 0.1:  # Skip convolution for nearly zero sigma
                #             ssp = ndimage.gaussian_filter1d(ssp, sigma)
                #     else:
                #         ssp = util.gaussian_filter1d(ssp, sigma)  # convolution with variable sigma
                # ssp_new = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[0]
                # ssp_new = ssp
                # if norm_range is not None:
                #     flux[j, k] = np.mean(ssp_new[band])
                #     ssp_new /= flux[j, k]  # Normalize every spectrum
                templates[:, j, k] = ssp
                age_grid[j, k] = age
                metal_grid[j, k] = met

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

        # The code below is the cause of the flux scale issue!!!!!
        # if norm_range is None:
        #     flux = np.median(templates[templates > 0])
        #     templates /= flux  # Normalize by a scalar


        self.templates = templates
        self.lam_temp = lam
        if 'loginterp' in pathname:
            self.age_grid = 10 ** (age_grid - 9)
        else:
            self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.n_ages = n_ages
        self.n_metal = n_metal
        self.flux = flux
        self.FWHM_tem = FWHM_tem
