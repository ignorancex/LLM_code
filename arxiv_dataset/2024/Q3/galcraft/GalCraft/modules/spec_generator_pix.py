import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from . import utils
import numpy as np
import collections
from spectres import spectres

from scipy.stats import binned_statistic_dd
from scipy.interpolate import RegularGridInterpolator
from time import perf_counter as clock


SpecResult = collections.namedtuple('SpecResult', ['mass_fraction_pixel', 'galaxy_rebin', 'i', 'j'])

class SpecGeneratorPix():

    def __init__(self, d_t, xb, yb, zb, logage_grid, metal_grid, alpha_grid, x_edges, y_edges, use_losvd, use_extinc,
                 extinc_factor, single_alpha, templatesOversampled_shape, new_wave, wave_oversampled,
                 sig_oversampled, factor, logger, templatesTransposed, interpolator_method):

        self.d_t = d_t
        self.xb = xb
        self.yb = yb
        self.zb = zb
        self.logage_grid = logage_grid.copy()
        self.metal_grid = metal_grid.copy()
        self.alpha_grid = alpha_grid.copy()
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.use_losvd = use_losvd
        self.use_extinc = use_extinc
        self.extinc_factor = extinc_factor
        self.single_alpha = single_alpha
        self.new_wave = new_wave.copy()
        self.wave_oversampled = wave_oversampled.copy()
        self.sig_oversampled = sig_oversampled.copy()
        self.templatesOversampled_shape = templatesOversampled_shape
        self.factor = factor
        self.logger = logger
        self.templatesTransposed = templatesTransposed.copy()
        self.interpolator_method = interpolator_method

        # Set up the spectra interpolator
        # mask_logage = self.logage_grid
        # mask_metal =
        # mask_alpha =
        # self.logage_grid_pix = self.logage_grid[mask_logage]
        # self.metal_grid_pix = self.metal_grid[mask_metal]
        # self.alpha_grid_pix = self.alpha_grid[mask_alpha]
        # self.templatesTransposed = self.templatesTransposed[:, mask_logage, mask_metal, mask_alpha]
        self.interpolator = self.setup_interpolator()



    def __call__(self, i, j):

        t = clock()

        self.d_t_pixel = self.d_t[(self.d_t['bin_x'] == self.x_edges.shape[0] - 1 - i) & (self.d_t['bin_y'] == j + 1)]
        # delattr(self, 'd_t')
        vel = np.array(self.d_t_pixel['vr'])
        mass = np.array(self.d_t_pixel['mass'])

        # Set up the spectra interpolator
        # mask_logage = self.logage_grid
        # mask_metal =
        # mask_alpha =
        # self.logage_grid_pix = self.logage_grid[mask_logage]
        # self.metal_grid_pix = self.metal_grid[mask_metal]
        # self.alpha_grid_pix = self.alpha_grid[mask_alpha]
        # self.templatesTransposed = self.templatesTransposed[:, mask_logage, mask_metal, mask_alpha]
        # self.interpolator = self.setup_interpolator()


        if len(self.d_t_pixel) > 0:


            # Calculate the weights
            if self.single_alpha:

                sample = np.transpose(np.vstack([self.d_t_pixel['cube_logage'], self.d_t_pixel['cube_m_h']]), (1, 0))

                assert self.interpolator_method != "interp_on_alpha", "For single alpha SSP templates, there is no option to choose 'interp_on_alpha' as the interpolation method."

                if self.interpolator_method == "linear":
                    statistic_mass, edges, bin_indexes = binned_statistic_dd(sample, mass, statistic='count',
                                                                             bins=[self.logage_grid, self.metal_grid],
                                                                             expand_binnumbers=True)
                    # Calculate the mass_fraction distribution
                    mass_fraction_pixel = np.zeros(self.templatesOversampled_shape[1:])
                    for k in range(len(sample)):
                        particle = sample[k]
                        weight_logage_left = 1 - (particle[0] - self.logage_grid[bin_indexes[0, k] - 1]) / (self.logage_grid[bin_indexes[0, k]] - self.logage_grid[bin_indexes[0, k] - 1])
                        weight_logage_right = 1 - weight_logage_left
                        weight_metal_left = 1 - (particle[1] - self.metal_grid[bin_indexes[1, k] - 1]) / (self.metal_grid[bin_indexes[1, k]] - self.metal_grid[bin_indexes[1, k] - 1])
                        weight_metal_right = 1 - weight_metal_left
                        weights_grid = np.meshgrid([weight_logage_left, weight_logage_right],
                                                   [weight_metal_left, weight_metal_right])
                        weights = (weights_grid[0] * weights_grid[1]).reshape(-1)
                        grids_index = np.meshgrid([bin_indexes[0, k] - 1, bin_indexes[0, k]],
                                                  [bin_indexes[1, k] - 1, bin_indexes[1, k]])
                        logage_index = grids_index[0].reshape(-1)
                        metal_index = grids_index[1].reshape(-1)
                        mass_fraction_pixel[logage_index, metal_index, 0] += weights

                elif self.interpolator_method == "nearest":
                    mass_fraction_pixel = binned_statistic_dd(sample, mass, statistic='sum', bins=[self.xb, self.yb]).statistic
                    mass_fraction_pixel = np.stack([mass_fraction_pixel], axis=-1)

                else:
                    raise ValueError("Calculate weights failed, the interpolation method has to be either 'nearest', 'linear', or 'interp_on_alpha'.")


            else:


                sample = np.transpose(np.vstack([self.d_t_pixel['cube_logage'], self.d_t_pixel['cube_m_h'], self.d_t_pixel['cube_alpha_fe']]), (1, 0))

                if self.interpolator_method == "linear":
                    statistic_mass, edges, bin_indexes = binned_statistic_dd(sample, mass, statistic='count',
                                                                             bins=[self.logage_grid, self.metal_grid, self.alpha_grid],
                                                                             expand_binnumbers=True)
                    # Calculate the mass_fraction distribution
                    mass_fraction_pixel = np.zeros(self.templatesOversampled_shape[1:])

                    for k in range(len(sample)):
                        particle = sample[k]

                        weight_logage_left = 1 - (particle[0] - self.logage_grid[bin_indexes[0, k] - 1]) / (self.logage_grid[bin_indexes[0, k]] - self.logage_grid[bin_indexes[0, k] - 1])
                        weight_logage_right = 1 - weight_logage_left
                        weight_metal_left = 1 - (particle[1] - self.metal_grid[bin_indexes[1, k] - 1]) / (self.metal_grid[bin_indexes[1, k]] - self.metal_grid[bin_indexes[1, k] - 1])
                        weight_metal_right = 1 - weight_metal_left
                        weight_alpha_left = 1 - (particle[2] - self.alpha_grid[bin_indexes[2, k] - 1]) / (self.alpha_grid[bin_indexes[2, k]] - self.alpha_grid[bin_indexes[2, k] - 1])
                        weight_alpha_right = 1 - weight_alpha_left
                        weights_grid = np.meshgrid([weight_logage_left, weight_logage_right],
                                                   [weight_metal_left, weight_metal_right],
                                                   [weight_alpha_left, weight_alpha_right])
                        weights = (weights_grid[0] * weights_grid[1] * weights_grid[2]).reshape(-1)
                        grids_index = np.meshgrid([bin_indexes[0, k] - 1, bin_indexes[0, k]],
                                                  [bin_indexes[1, k] - 1, bin_indexes[1, k]],
                                                  [bin_indexes[2, k] - 1, bin_indexes[2, k]])
                        logage_index = grids_index[0].reshape(-1)
                        metal_index = grids_index[1].reshape(-1)
                        alpha_index = grids_index[2].reshape(-1)

                        mass_fraction_pixel[logage_index, metal_index, alpha_index] += weights


                elif self.interpolator_method == "nearest":
                    mass_fraction_pixel = binned_statistic_dd(sample, mass, statistic='sum', bins=[self.xb, self.yb, self.zb]).statistic

                elif self.interpolator_method == "interp_on_alpha":
                    sample = sample[:, 0:2]
                    fraction = 2.5 * self.d_t_pixel['cube_alpha_fe']
                    statistic_mass_alpha00 = binned_statistic_dd(sample, values=np.array(mass*(1-fraction)),
                                                                 statistic='sum', bins=[self.xb, self.yb]).statistic
                    statistic_mass_alpha04 = binned_statistic_dd(sample, values=np.array(mass*fraction),
                                                                 statistic='sum', bins=[self.xb, self.yb]).statistic
                    mass_fraction_pixel = np.stack([statistic_mass_alpha00, statistic_mass_alpha04], axis=-1)

                else:
                    raise ValueError(
                        "Stacking spectra failed, the interpolation method has to be either 'nearest', 'linear', or 'interp_on_alpha'.")



            # Calculate the stacked spectrum
            result_galaxy = np.zeros(self.templatesOversampled_shape[0])

            for k in range(len(sample)):
                particle = sample[k]

                if self.interpolator_method == "nearest" or self.interpolator_method == "linear":
                    result = self.interpolator(particle)[0]

                elif self.interpolator_method == "interp_on_alpha":
                    result_alpha00 = self.interpolator[0](particle)[0]
                    result_alpha04 = self.interpolator[1](particle)[0]
                    result = result_alpha00 * (1 - fraction[k]) + result_alpha04 * fraction[k]

                result = result * mass[k] / (1e-20 * 4 * np.pi * (self.d_t_pixel[k]['dist']*3.08567758e21)**2 / (3.826e33))

                if self.use_losvd == True:
                    result = utils.doppler_shift(self.wave_oversampled, result, vel[k])

                if self.use_extinc == True:
                    result = result * utils.reddening_cal00(self.wave_oversampled, self.d_t_pixel[k]['exbv'] / self.extinc_factor)

                result_galaxy += result


            # Degrade and Rebin the spectra
            galaxy_degraded = utils.degrade_spec_ppxf(result_galaxy, None, self.sig_oversampled, gau_npix=None)[0]
            galaxy_rebin = spectres(self.new_wave, self.wave_oversampled, galaxy_degraded, fill=np.nan, verbose=False)



        else:

            mass_fraction_pixel = np.zeros(self.templatesOversampled_shape[1:])
            galaxy_rebin = np.zeros(int(self.templatesOversampled_shape[0] / self.factor))


        self.logger.info('Generated a spectrum for spatial pixel %s using %s particles, time elapsed: %.2f s' % ([int(i), int(j)], len(self.d_t_pixel), clock() - t))
        return SpecResult(mass_fraction_pixel=mass_fraction_pixel, galaxy_rebin=galaxy_rebin, i=i, j=j)




    def setup_interpolator(self):
        # Set up the spectra interpolator

        if self.single_alpha:
            assert self.interpolator_method != "interp_on_alpha", "'interp_on_alpha' is not avaliable for single alpha case."
            SSP_grids = (self.logage_grid, self.metal_grid)
            interpolator = RegularGridInterpolator(SSP_grids, self.templatesTransposed[:, :, 0, :], method=self.interpolator_method)
            self.interpolator = interpolator

        else:
            if self.interpolator_method == "nearest" or self.interpolator_method == "linear":
                SSP_grids = (self.logage_grid, self.metal_grid, self.alpha_grid)
                interpolator = RegularGridInterpolator(SSP_grids, self.templatesTransposed, method=self.interpolator_method)
                self.interpolator = interpolator
            elif self.interpolator_method == "interp_on_alpha":
                SSP_grids = (self.logage_grid, self.metal_grid)
                interpolator_alpha00 = RegularGridInterpolator(SSP_grids, self.templatesTransposed[:, :, 0, :], method="nearest")
                interpolator_alpha04 = RegularGridInterpolator(SSP_grids, self.templatesTransposed[:, :, 1, :], method="nearest")
                self.interpolator = [interpolator_alpha00, interpolator_alpha04]
            else:
                raise ValueError(
                    "Initializing the interpolator ailed, the method has to be either 'nearest', 'linear', or 'interp_on_alpha'.")

        return interpolator
