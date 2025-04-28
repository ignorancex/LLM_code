import glob
import numpy as np


def age_metal(filename, age_grid, metal_grid):
    '''
    extract the age and metallicity from the name of file
    modified from miles_util.py of PPXF
    '''
    s = filename[:-4].split('_')
    age = age_grid[int(s[-2])]
    metal = metal_grid[int(s[-1])]
    # print(filename, age, metal)
    return age, metal


class conroy:
    '''
    This is a class which is written for uploading the pegase-hr SSP models.
    The SSP models have been interpolated in a better grid, following the steps
    in Kacharov et al. 2018 using ulyss.
    The new age and metallicity grid is similar as the MILES SSP models.
    This file is modified from miles_util.py of PPXF
    '''
    def __init__(self, pathname, FWHM_tem=2.51,
                 age_range=None, metal_range=None):

        files = glob.glob(pathname + 'conroy_ssp_mist_v2.3_*.npy')
        age_grid = np.load(pathname + 'logage_grid.npy')
        age_grid = 10**(age_grid)
        metal_grid = np.load(pathname + 'metal_grid.npy')
        all = [age_metal(f, age_grid, metal_grid) for f in files]
        all_ages, all_metals = np.array(all).T
        ages, metals = np.unique(all_ages), np.unique(all_metals)
        n_ages, n_metal = len(ages), len(metals)

        # assert set(all) == set([(a, b) for a in ages for b in metals]), \
            # 'Ages and Metals do not form a Cartesian grid'

        # Extract the wavelength range and logarithmically rebin one spectrum
        # to the same velocity scale of the galaxy spectrum, to determine the
        # size needed for the array which will contain the template spectra.

        ssp = np.load(files[0])
        lam = np.load(pathname + 'wave.npy')
        # lam_range_temp = lam[[0, -1]]
        # ssp_new, ln_lam_temp = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[:2]
        # ssp_new = ssp

        # if norm_range is not None:
        #     norm_range = np.log(norm_range)
        #     band = (norm_range[0] <= lam) & (lam <= norm_range[1])

        templates = np.empty((ssp.size, n_ages, n_metal))
        age_grid = np.empty((n_ages, n_metal))
        metal_grid = np.empty((n_ages, n_metal))
        flux = np.ones((n_ages, n_metal))

        # Convolve the whole Vazdekis library of spectral templates
        # with the quadratic difference between the galaxy and the
        # Vazdekis instrumental resolution. Logarithmically rebin
        # and store each template as a column in the array TEMPLATES.

        # Quadratic sigma difference in pixels PEGASE-HR --> galaxy
        # The formula below is rigorously valid if the shapes of the
        # instrumental spectral profiles are well approximated by Gaussians.
        # if FWHM_gal is not None:
        #     FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
        #     sigma = FWHM_dif/2.355/0.2   # Sigma difference in pixels

        # Here we make sure the spectra are sorted in both [M/H] and Age
        # along the two axes of the rectangular grid of templates.
        for j, age in enumerate(ages):
            for k, met in enumerate(metals):
                p = all.index((age, met))
                ssp = np.load(files[p])
                # if FWHM_gal is not None:
                #     if np.isscalar(FWHM_gal):
                #         if sigma > 0.1:   # Skip convolution for nearly zero sigma
                #             ssp = ndimage.gaussian_filter1d(ssp, sigma)
                #     else:
                #         ssp = util.gaussian_filter1d(ssp, sigma)  # convolution with variable sigma
                # ssp_new = util.log_rebin(lam_range_temp, ssp, velscale=velscale)[0]
                # ssp_new = ssp
                # if norm_range is not None:
                #     flux[j, k] = np.mean(ssp_new[band])
                #     ssp_new /= flux[j, k]
                templates[:, j, k] = ssp
                age_grid[j, k] = age
                metal_grid[j, k] = met

        if age_range is not None:
            w = (age_range[0] <= age_grid[:, 0]) & (age_grid[:, 0] <= age_range[1])
            templates = templates[:, w, :]
            age_grid = age_grid[w, :]
            metal_grid = metal_grid[w, :]
            n_ages, n_metal = age_grid.shape

        if metal_range is not None:
            w = (metal_range[0] <= metal_grid[0, :]) & (metal_grid[0, :] <= metal_range[1])
            templates = templates[:, :, w]
            age_grid = age_grid[:, w]
            metal_grid = metal_grid[:, w]
            n_ages, n_metal = age_grid.shape

        # if norm_range is None:
        #     templates /= np.median(templates)  # Normalize by a scalar

        self.templates = templates
        self.lam_temp = lam
        self.age_grid = age_grid
        self.metal_grid = metal_grid
        self.n_ages = n_ages
        self.n_metal = n_metal
        self.flux = flux
        self.FWHM_tem = FWHM_tem
