import os
import numpy as np
from multiprocessing import Pool
from time import perf_counter as clock

from scipy.stats import binned_statistic_2d

import matplotlib.colors as colors
from matplotlib import pyplot as plt


##################################################################################
# Initialization functions
def initialize_Dirs(CommandOptions):
    '''
    Initialize directory paths
    :param CommandOptions: Command parameters
    :return: Paths for each directory
    '''
    if os.path.isfile(CommandOptions.defaultDir) == True:
        for line in open(CommandOptions.defaultDir, "r"):
            if not line.startswith('#'):
                line = line.split('=')
                line = [x.strip() for x in line]
                if os.path.isdir(line[1]) == True:
                    if line[0] == 'configDir':
                        configDir = line[1]
                    elif line[0] == 'outputDir':
                        outputDir = line[1]
                    elif line[0] == 'modelDir':
                        modelDir = line[1]
                    elif line[0] == 'templateDir':
                        templateDir = line[1]
                else:
                    print("WARNING! "+line[1]+" specified as default "+line[0]+" is not a directory!")
    else:
        print("WARNING! "+CommandOptions.defaultDir+" is not a file!")

    return configDir, modelDir, templateDir, outputDir


##################################################################################
# Doppler shift functions

def doppler_shift(wavelength, flux, dv):
    '''
    This is the function from The Payne
    dv is in km/s
    We use the convention where a positive dv means the object is moving away.

    This linear interpolation is actually not that accurate, but is fine if you
    only care about accuracy to the level of a few tenths of a km/s. If you care
    about better accuracy, you can do better with spline interpolation.
    '''
    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux



##################################################################################
# Reddening function
def reddening_cal00(lam, ebv):
    """
    Reddening curve of `Calzetti et al. (2000)
    <http://ui.adsabs.harvard.edu/abs/2000ApJ...533..682C>`_
    This is reliable between 0.12 and 2.2 micrometres.
    - LAMBDA is the restframe wavelength in Angstrom of each pixel in the
      input galaxy spectrum (1 Angstrom = 1e-4 micrometres)
    - EBV is the assumed E(B-V) colour excess to redden the spectrum.
      In outputs the vector FRAC gives the fraction by which the flux at each
      wavelength has to be multiplied, to model the dust reddening effect.

    """
    ilam = 1e4/lam  # Convert Angstrom to micrometres and take 1/lambda
    rv = 4.05  # C+00 equation (5)

    # C+00 equation (3) but extrapolate for lam > 2.2
    # C+00 equation (4) (into Horner form) but extrapolate for lam < 0.12
    k1 = rv + np.where(lam >= 6300, 2.76536*ilam - 4.93776,
                       ilam*((0.029249*ilam - 0.526482)*ilam + 4.01243) - 5.7328)
    fact = 10**(-0.4*ebv*k1.clip(0))  # Calzetti+00 equation (2) with opposite sign

    return fact # The model spectrum has to be multiplied by this vector





##################################################################################
# Plotting functions

def plot_binned_grids_color(x, y, values, statistic, x_edges, y_edges, xlabel, ylabel,
                            plot_cb=True, cmap='hot', cblabel=None, color_Lognorm=False,
                            invert_x=True, **kwargs):
    '''

    :param x:
    :param y:
    :param values:
    :param statistic:
    :param x_edges:
    :param y_edges:
    :param cmap:
    :param xlabel:
    :param ylabel:
    :param cblabel:
    :param percentile:
    :param norm:
    :return:
    '''

    binned_statistic = binned_statistic_2d(x, y, values, statistic, bins=[x_edges, y_edges]).statistic

    if statistic == 'count':
        binned_statistic[binned_statistic==0] = np.nan

    vmin = np.nanpercentile(binned_statistic[binned_statistic!=0], 0.5)
    vmax = np.nanpercentile(binned_statistic[binned_statistic!=0], 99.5)

    if color_Lognorm == True:
        color_norm = colors.LogNorm(vmin, vmax)
    else:
        color_norm = colors.Normalize(vmin, vmax)

    ax = plt.gca()
    im = ax.pcolormesh(x_edges, y_edges, binned_statistic.T, cmap=cmap, norm=color_norm, **kwargs)
    if plot_cb == True:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cblabel)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if invert_x:
        ax.invert_xaxis()
    # fig.tight_layout()

    return im, ax, binned_statistic, x_edges, y_edges





def plot_parameter_maps(mass_fraction_pixel_bin, age_grid_2d, metal_grid_2d, alpha_grid, reg_dim,
                        x_edges, y_edges, xlabel, ylabel, cmap, **kwargs):
    '''

    :param mass_fraction_pixel_bin:
    :param age_grid_2d:
    :param metal_grid_2d:
    :param reg_dim:
    :param x_edges:
    :param y_edges:
    :param xlabel:
    :param ylabel:
    :param cmap:
    :param kwargs:
    :return:
    '''

    x_edges = np.flip(x_edges)

    mass_weighted_age = np.zeros(mass_fraction_pixel_bin.shape[3:])
    mass_weighted_metal = np.zeros(mass_fraction_pixel_bin.shape[3:])
    mass_weighted_alpha = np.zeros(mass_fraction_pixel_bin.shape[3:])

    for i in range(mass_fraction_pixel_bin.shape[-2]):
        for j in range(mass_fraction_pixel_bin.shape[-1]):
            mass_fraction_pixel = mass_fraction_pixel_bin[:, :, :, i, j]
            if np.sum(mass_fraction_pixel) == 0:
                mass_weighted_age[i, j] = np.nan
                mass_weighted_metal[i, j] = np.nan
                mass_weighted_alpha[i, j] = np.nan
            else:
                age, metal, alpha = mean_age_metal_alpha(age_grid_2d, metal_grid_2d, alpha_grid, mass_fraction_pixel, reg_dim, True)
                mass_weighted_age[i, j] = age
                mass_weighted_metal[i, j] = metal
                mass_weighted_alpha[i, j] = alpha

    plt.figure(figsize=[18, 4])

    plt.subplot(131)
    ax1 = plt.gca()
    vmin = np.nanpercentile(mass_weighted_age, 0.5)
    vmax = np.nanpercentile(mass_weighted_age, 99.5)
    im1 = ax1.pcolormesh(x_edges, y_edges, mass_weighted_age, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('log(Age)')
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)

    plt.subplot(132)
    ax2 = plt.gca()
    vmin = np.nanpercentile(mass_weighted_metal, 0.5)
    vmax = np.nanpercentile(mass_weighted_metal, 99.5)
    im2 = ax2.pcolormesh(x_edges, y_edges, mass_weighted_metal, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('[M/H]')
    ax2.set_ylabel(ylabel)
    ax2.set_xlabel(xlabel)

    plt.subplot(133)
    ax3 = plt.gca()
    vmin = np.nanpercentile(mass_weighted_alpha, 0.5)
    vmax = np.nanpercentile(mass_weighted_alpha, 99.5)
    im3 = ax3.pcolormesh(x_edges, y_edges, mass_weighted_alpha, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('[Alpha/Fe]')
    ax3.set_ylabel(ylabel)
    ax3.set_xlabel(xlabel)

    ax1.invert_xaxis()
    ax2.invert_xaxis()
    ax3.invert_xaxis()

    plt.tight_layout()

##################################################################################





##################################################################################
# Spectral Degrading functions


def cal_degrade_sig(FWHM_gal, FWHM_tem, dlam):
    FWHM_dif = np.sqrt(FWHM_gal ** 2 - FWHM_tem ** 2)
    sigma = FWHM_dif / 2.355 / dlam

    # if np.all((sigma==0)):
    #     sigma = sigma - 1 # this is the case when FWHM_gal == FWHM_tem

    return sigma



def degrade_spec_ppxf(spec, spec_err=None, sig=0, gau_npix=None):
    '''
    Modified from PPXF v8.1.0
    :param spec:
    :param spec_err:
    :param sig:
    :param gau_npix:
    :return:
    '''
    # This function can now skip the err if doesn't have
    if np.all(sig == 0):
        return spec, spec_err
    elif np.isscalar(sig):
        sig = np.zeros(spec.shape) + sig
    sig = sig.clip(1e-10)  # forces zero sigmas to have 1e-10 pixels

    if gau_npix == None:
        p = int(np.ceil(np.max(3*sig)))
    else:
        p = gau_npix
    m = 2 * p + 1  # kernel sizes
    x2 = np.linspace(-p, p, m) ** 2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]

    gau = np.exp(-x2[:, None] / (2 * sig ** 2))
    gau /= np.sum(gau, 0)[None, :] # Normalize kernel

    conv_spectrum = np.einsum('ij,ij->j', a, gau)

    if np.all(spec_err != None):
        a_e = np.zeros((m, n))
        for j in range(m):
            a_e[j, p:-p] = spec_err[j:n - m + j + 1]
        conv_spectrum_err = np.sqrt((a_e**2 * gau**2).sum(0))
    else:
        conv_spectrum_err = None

    return conv_spectrum, conv_spectrum_err



def process_degrade_cube(spec, spec_err, i, j, sigma, gau_npix=None):
    spec_ij_degraded, spec_err_ij_degraded = degrade_spec_ppxf(spec, spec_err, sigma, gau_npix)
    return spec_ij_degraded, spec_err_ij_degraded, i, j



def degrade_spec_cube(cube_flux, cube_err, FWHM_gal, FWHM_tem, ncpu, dlam, gau_npix=None):
    # If not using cube_err, set it tobe np.zeros(cube_flux.shape)
    t = clock()

    sigma = cal_degrade_sig(FWHM_gal, FWHM_tem, dlam)

    cube_flux_degraded = np.zeros(cube_flux.shape)
    cube_err_degraded = np.zeros(cube_err.shape)

    pool = Pool(processes=ncpu)
    results = []
    for i in range(cube_flux.shape[1]):
        for j in range(cube_flux.shape[2]):
            results.append(pool.apply_async(process_degrade_cube, (cube_flux[:, i, j], cube_err[:, i, j], i, j, sigma, gau_npix)))
    pool.close()
    pool.join()

    for result in results:
        spec_ij_degraded, spec_err_ij_degraded, i, j = result.get()
        cube_flux_degraded[:, i, j] = spec_ij_degraded
        cube_err_degraded[:, i, j] = spec_err_ij_degraded

    print('Elapsed time in generating spectra: %.2f s' % (clock() - t))

    return cube_flux_degraded, cube_err_degraded





##################################################################################
# Other functions

def mean_age_metal_alpha(age_grid_2d, metal_grid_2d, alpha_grid, weights, reg_dim, quiet=False):
    '''
    This is modified from PPXF, compatible with single alpha case
    :param age_grid_2d:
    :param metal_grid_2d:
    :param alpha_grid:
    :param weights:
    :param reg_dim:
    :param quiet:
    :return:
    '''

    log_age_grids = np.log10(age_grid_2d) + 9
    metal_grids = metal_grid_2d

    weights_sum = np.sum(weights, axis=2)
    mean_log_age = np.sum(weights_sum * log_age_grids) / np.sum(weights_sum)
    mean_metal = np.sum(weights_sum * metal_grids) / np.sum(weights_sum)
    mean_alpha = np.sum(np.sum(weights * alpha_grid, axis=2)) / np.sum(weights_sum)

    if not quiet:
        print('Weighted <logAge> [yr]: %#.3g' % mean_log_age)
        print('Weighted <[M/H]>: %#.3g' % mean_metal)
        print('Weighted <[Alpha/Fe]>: %#.3g' % mean_alpha)

    return mean_log_age, mean_metal, mean_alpha

