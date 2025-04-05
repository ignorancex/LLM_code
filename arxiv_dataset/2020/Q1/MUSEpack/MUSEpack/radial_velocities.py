#!/usr/bin/env python


__version__ = '1.1'

__revision__ = '20210128'

import sys
import os
import shutil
import warnings
import pyspeckit
import pickle
import numpy as np
from astropy.io import ascii
from astropy.table import Table
from astropy.table import Column
from astropy import units as u
from astropy import constants as const
from astropy import log
from astropy.stats import median_absolute_deviation as MAD
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from multiprocessing import cpu_count
from functools import partial
from scipy.special import wofz
from ppxf import ppxf
from ppxf import ppxf_util
import dask
import time
import logging
from scipy.ndimage.filters import gaussian_filter

''' internal modules'''
from MUSEpack.ppxf_MC import ppxf_MC
from MUSEpack.utils import *
from MUSEpack.line_fitter import *


logging.basicConfig(format='%(asctime)s [%(levelname)8s]:'\
+ ' %(message)s [%(funcName)s]', datefmt="%Y-%m-%d %H:%M:%S")


class RV_spectrum:
    """
    This class contains the 1D spectrum, as well as all fitting parameters
    and output values including the radial velocity catalog.

    Args:
        spec_id : :obj:`str`
            unique identifier of the spectrum

        spec_f : :func:`numpy.array`
            flux array of the spectrum in erg/s/cm:math:`^2`/Angstrom

        spec_err : :func:`numpy.array`
            flux uncertainty array of the spectrum

        spec_lambda : :func:`numpy.array`
            wavelength_array of the spectrum in Angstrom

    Kwargs:
        loglevel : :obj:`str` (optional, default: ``INFO``)
            ``DEBUG``: all functions run on a single core to obtain proper
            output in the correct order for proper debugging.

        templatebins : :obj:`float` (optional, default: 100000)
            Number of wavelength bins for the oversampled spectrum used to fit
            the spectral lines.

        specbinsize : :obj:`float` (optional, default: 1.25)
            The spectral bin size. The default is set to fit the MUSE dataset

        dispersion : :obj:`float` (optional, default: 2.4)
            The dispersion of the spectrograph. The default is set to the
            nominal MUSE instrument dispersion

        linetype: :obj:`str` (optional, default: absorption)
            absorption: All spectral lines are absorption lines

            emission: All spectral lines are emission lines

            both : spectral lines can be absorption or emission lines. **This**
            **mode has not been tested yet !!!**

        rv_sys : :obj:`float` (optional, default: 0)
            systematic RV shift in km/s
            Should be provided for large velocity offsets or redshifted objects

    """

    def __init__(self, spec_id, spec_f, spec_err, spec_lambda,\
    loglevel="INFO", templatebins=100000, specbinsize=1.25, dispersion=2.4,\
    linetype='absorption', rv_sys=0.):

        self.spec_id = spec_id          # ID of input spectrum
        self.spec_f = spec_f            #input spectrum
        self.spec_err = spec_err        #1s uncertainty of input spectrum
        self.spec_SNR = spec_f / spec_err #SNR of the spectrum
        self.spec_lambda = spec_lambda  #lambda array for input spectrum

        self.fit_residuals = np.zeros_like(spec_f) * np.nan  #plot residuals
                                                           # from the specplot
        self.template_f = np.zeros_like(spec_f)      # artificially created
                                                     #spectrum after fit with
                                                     #catalog wavelengths.
        self.fit_f = np.zeros_like(spec_f)      # fitted spectrum.
        self.continuum = np.zeros_like(spec_f) # the continuum of
                                               #artificially created spectrum
        self.linetype = linetype #linetype: absorption, emission, both

        self.highres_samples = templatebins #samplerate for the template
        self.spec_lambda_highres = np.linspace(spec_lambda[0],\
                                   spec_lambda[-1], templatebins)
        self.template_f_highres\
        = np.zeros_like(self.spec_lambda_highres, dtype=float)
        self.fit_f_highres\
        = np.zeros_like(self.spec_lambda_highres, dtype=float)
        self.continuum_highres\
        = np.zeros_like(self.spec_lambda_highres, dtype=float)

        self.specbinsize = specbinsize
        self.dispersion = dispersion

        self.rv_sys = rv_sys

        self.cat = None # pandas dataframe for all the line fit parameters
        self.rv = None      #radial velocity of the star
        self.erv = None     #1sigma radial velocity uncertainty
        self.n_lines = None  # number of lines used to fit the RVs

        self.rv_peak = None      #radial velocity of the star
        self.erv_peak = None     #1sigma radial velocity uncertainty
        self.n_lines_peak = None  # number of lines used to fit the peak RVs

        self.loglevel = loglevel

        self.logger = logging.getLogger(self.spec_id)

        if self.loglevel == "INFO":
            self.logger.setLevel(logging.INFO)
        if self.loglevel == "DEBUG":
            self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(str(self.spec_id) + '.log', mode='w')

        fh.setLevel(self.loglevel)
        formatter = logging.Formatter('%(asctime)s [%(levelname)8s]: '\
        + '%(message)s [%(funcName)s]', datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.info('RV_spectrum vers. ' + str(__version__)\
        + ' rev. ' + str(__revision__))
        self.logger.info('Initiate instance of RV_spectrum for: '\
        + str(self.spec_id))
        self.logger.debug('DEBUG mode => all modules run on one core')
        self.logger.info('Systematic RV shift: '\
        + str('{:.2f}'.format(self.rv_sys)) + ' km/s')

    def clean(self):
        """
        This modules resets all of the output. This **must** be executed before
        repeating the spectral fit without re-initiating the class.

        """

        self.logger.info('The fit output is cleaned:')
        self.fit_residuals = np.zeros_like(self.spec_f)
        self.template_f = np.zeros_like(self.spec_f)
        self.continuum = np.zeros_like(self.spec_f)

    def catalog(self, initcat=None, save=False, load=None, printcat=False):
        """
        This module handles the catalog that holds the fit results
        of the primary lines.

        Kwargs:
            initcat : :obj:`ascii` (optional, default: :obj:`None`)
                :obj:`ascii` file containing the primary lines
                format: name, lambda, start, end

            save : :obj:`bool` (optional, default: :obj:`False`)
                :obj:`True`: The catalog is written to a file.

            load : :obj:`str` (optional, default: :obj:`None`)
                Loads a catalog from catalog file.

            printcat : :obj:`bool` (optional, default: :obj:`False`)
                :obj:`True`: The catalog is printed to the terminal

        """

        if load:
            self.logger.info('Load catalog from file: ' + str(load))
            if initcat:
                initcat = None
                self.logger.warning('Loading a catalog: initcat = None')
            self.cat.read_csv(path_or_buf=load)

        if save:
            self.logger.info('Save catalog to file: ' + str(load))
            self.cat.to_csv(path_or_buf=self.spec_id + '.cat')

        if initcat:
            self.logger.info('Initiating the catalog')
            temp = ascii.read(initcat)
            self.logger.info('Adding lines: ' + str(temp['name'].data))

            l_lab = temp['lambda']
            l_start = temp['start']
            l_end = temp['end']

            d = {'l_lab': l_lab,\
                 'l_start': l_start,\
                 'l_end': l_end,\
                 'l_fit': np.empty_like(temp['lambda']),\
                 'a_fit': np.empty_like(temp['lambda']),\
                 'sg_fit': np.empty_like(temp['lambda']),\
                 'sl_fit': np.empty_like(temp['lambda']),\
                 'cont_order': temp['contorder'],\
                 'RV': np.empty_like(temp['lambda']),\
                 'eRV': np.empty_like(temp['lambda']),\
                 'used': np.empty_like(temp['name']),\
                 'significance': np.empty_like(temp['lambda'])}

            self.cat = pd.DataFrame(d, index=temp['name'].data)

        if printcat:
            print(self.cat)

    def plot(self, oversampled=False):
        """
        Plots the spectrum and the regions of the primary lines to a file
        including the spectral fit and the template.

        Kwargs:
            oversampled : :obj:`bool` (optional, default: :obj:`False`)
                :obj:`True`: The oversampled spectral fit and the template are
                used in the plot.

        """

        self.logger.info('Initiate plotting')

        n_lines = len(self.cat.index)
        nrows = int(len(self.cat.index) / 3.) + 1
        if len(self.cat.index) % 3 > 0:
            nrows += 1
        plthight = 2 * nrows + 1

        spec_plot = plt.figure(self.spec_id, figsize=(8, plthight))
        plotgrid = gridspec.GridSpec(nrows, 3)
        ax_spec = plt.subplot(plotgrid[0,:])

        non_inf_cont = ~np.isinf(self.template_f / self.continuum)
        non_inf_cont_highres\
        = ~np.isinf(self.template_f_highres / self.continuum_highres)

        # to better plot larger numbers
        exponent = int(np.log10(np.nanmedian(self.spec_f[non_inf_cont])) - 1.)
        factor = float(10 ** (exponent * (-1)))
        ax_spec.plot(self.spec_lambda[non_inf_cont],\
        self.spec_f[non_inf_cont] * factor, c='black',\
        label='data', linewidth=2)

        if not oversampled:
            ax_spec.plot(self.spec_lambda[non_inf_cont],\
            self.template_f[non_inf_cont] * factor, c='red',\
            label='template', linewidth=2)

            ax_spec.plot(self.spec_lambda[non_inf_cont],\
            self.fit_residuals[non_inf_cont] * factor, c='green',\
            label='residual', linewidth=2)

        if oversampled:
            ax_spec.plot(self.spec_lambda_highres[non_inf_cont_highres],\
            self.template_f_highres[non_inf_cont_highres] * factor, c='red',\
            label='template', linewidth=2)

            ax_spec.plot(self.spec_lambda[non_inf_cont],\
            self.fit_residuals[non_inf_cont] * factor, c='green',\
            label='residual', linewidth=2)

        ax_spec.legend(fontsize=12)
        ax_spec.set_ylabel(r'flux [$10^{' + str(exponent)\
        + r'}$ erg/s/cm$^2$/${\rm \AA}$]', fontsize=12)
        ax_spec.set_xlabel(r'wavelength $[{\rm \AA}]$', fontsize=12)
        ax_spec.tick_params(axis='both', which='major', labelsize=12, width=2)
        ax_spec.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        for n in np.arange(len(self.cat.index)):
            nrow = int(n / 3) + 1
            ncol = n % 3
            el = self.cat.index[n]
            ax_norm = plt.subplot(plotgrid[nrow, ncol])

            ax_norm.plot(self.spec_lambda, self.spec_f / self.continuum,\
            c='black', label='data', linewidth=2)

            if not oversampled:
                ax_norm.plot(self.spec_lambda,\
                self.template_f / self.continuum, c='red',\
                label='template', linewidth=1.5)

                ax_norm.plot(self.spec_lambda,\
                self.fit_f / self.continuum, '--', c='lightblue',\
                label='fit', linewidth=1.5)

                ax_norm.plot(self.spec_lambda,\
                self.fit_residuals / self.continuum, c='green',\
                label='residual', linewidth=1.5)

            if oversampled:
                ax_norm.plot(self.spec_lambda_highres,\
                self.template_f_highres / self.continuum_highres,\
                c='red', label='template', linewidth=1.5)

                ax_norm.plot(self.spec_lambda_highres,\
                self.fit_f_highres / self.continuum_highres, '--',\
                c='lightblue', label='fit', linewidth=1.5)

                ax_norm.plot(self.spec_lambda,\
                self.fit_residuals / self.continuum,\
                c='green', label='residual', linewidth=1.5)

            ax_norm.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax_norm.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax_norm.xaxis.set_major_locator(plt.MaxNLocator(2))

            ax_norm.set_xlabel(r'wavelength $[{\rm \AA}]$', fontsize=12)
            ax_norm.tick_params(axis='both', which='major', labelsize=12,\
            width=2)
            ax_norm.tick_params(axis='y', which='major', labelsize=12,\
            width=2, direction='in', left=True, right=True)

            if ncol == 0:
                ax_norm.set_ylabel(r'norm. flux', fontsize=12)
            if ncol != 0:
                ax_norm.yaxis.set_ticklabels([])

            ax_norm.set_xlim(self.cat.loc[el, 'l_start'],\
            self.cat.loc[el, 'l_end'])
            ax_norm.annotate(el, (0.1, 0.5), xycoords='axes fraction',\
            fontsize=12, fontweight='bold')

        plt.tight_layout(w_pad=-0.25, h_pad=-1.8)
        plt.savefig(self.spec_id + '_plot.png', dpi=600)
        plt.close()

    def line_fitting(self, input_cat, line_idxs, niter=5, n_CPU=-1,\
    resid_level=None, max_contorder=2, max_ladjust=4,\
    adjust_preference='contorder', input_continuum_deviation=0.05,\
    llimits=[-2., 2.], max_exclusion_level=0.3, blends=None, autoadjust=False,\
    fwhm_block=False):

        """
        Initializing the line fitting by calling :mod:`line_fitter`

        Args:
            input_cat: :func:`numpy.array`
                input spectral line catalog with wavelengths in Angstrom

            line_idxs: :func:`numpy.array`
                :obj:`str` :func:`numpy.array`: of the line names for which RV
                fits should be performed, must be identical to "name" in
                init_cat

        Kwargs:
            niter : :obj:`int` (optional, default: 5)
                The maximum number of iteration if convergence is not reached

            n_CPU : :obj:`float` (optional, default: -1)
                Setting the number of CPUs used for the parallelization. If set
                to -1 all available system resources are used. Maximum number
                of CPUs is the number of spectral lines the fit is performed
                on.

            resid_level: :obj:`float` or :obj:`None` (optional, default: None)
                The maximum MAD for the fit residuals for a succesfull fit

            max_contorder : :obj:`int` (optional, default: 2)
                The maximum polynominal order of the continuum

            max_ladjust : :obj:`int` (optional, default: 4)
                Ahe maximum number of wavelength range adjustments in
                steps of 5 Angstrom

            adjust_preference: :obj:`str` (optional, default: contorder)
                ``contorder``: continuum order is adjusted first

                ``wavelength``: wavelength range is adjusted first

            input_continuum_deviation :obj:`float` (optional, default: 0.05)
                Fraction by how much the continuum is allowed to deviate from a
                running median estimate. This is set to prevent lines mimicking
                a continuum

            llimits: :obj:`list` (optional, default: [-2., 2.])
                the limits for the wavelength fit as set in `ppxf`_

            max_exclusion_level :obj:`float` (optional, default: 0.3)
                The exclusion level for lines to be excluded from the next
                baseline estimate as set in `pyspeckit`_

            blends: :obj:`ascii` or :obj:`None` (optional, default: None)
                A file with primary lines that contain blends to provide a
                maximum amplitude ratio of the primary and the blend to prevent
                that the blend becomes the dominant line in the fit.

            autoadjust :obj:`bool` (optional, default: :obj:`False`)
                :obj:`True`: the wavelength limits ``llimit`` will be adjusted
                to the fit of the previous iteration. All other wavelength
                range are adjusted accordingly taking into account the proper
                velocity corrected shift :math:`\Delta \lambda/\lambda`. This
                is especially important to detect hyper-velocity stars.

            fwhm_block :obj:`bool` (optional, default: :obj:`False`)
                :obj:`True`: The minimum fwhm of the voigt profiles of the
                fitted lines ais the instrument's dispersion. This prevents
                unphysical lines.

                :obj:`False`: The minimum fwhm of the voigt profiles of the
                fitted lines is zero.

        """

        self.logger.info('Starting the line fitting')
        start_time = time.time()

        if blends:
            self.logger.info('A file with blends was provided')
        if autoadjust:
            self.logger.info('Automatic adjustments of wavelength limits')
        if fwhm_block:
            self.logger.info('FWHM is at least the instruments dispersion')
        if self.loglevel == "DEBUG":
            n_CPU = 1

        warnings.filterwarnings('ignore', category=RuntimeWarning,\
        message='divide by zero encountered in true_divide')
        warnings.filterwarnings('ignore', category=RuntimeWarning,\
        message='invalid value encountered in true_divide')
        warnings.filterwarnings('ignore', category=RuntimeWarning,\
        message='invalid value encountered in greater_equal')

        if isinstance(input_continuum_deviation, float):
            input_continuum_deviation\
            = {line_idxs[i]: input_continuum_deviation\
            for i in range(len(line_idxs))}
        assert len(input_continuum_deviation) == len(line_idxs),\
        'len(input_continuum_deviation) differs from number of lines'

        if n_CPU == -1:
            n_CPU = cpu_count()
        self.logger.info('Max number of cores: ' + str(n_CPU))

        result = [dask.delayed(line_fitter)(self, input_cat, ldx, niter,\
                  resid_level, max_contorder, max_ladjust, adjust_preference,\
                  input_continuum_deviation, llimits, max_exclusion_level,\
                  blends, autoadjust, fwhm_block)\
                  for ldx in np.array(line_idxs)]

        if self.loglevel == "DEBUG":
            results = dask.compute(*result, num_workers=1,\
                      scheduler='single-threaded')
        if not self.loglevel == "DEBUG":
            results = dask.compute(*result, num_workers=n_CPU,\
                      scheduler='processes')

        for result in results:

            if not result[13]:
                self.template_f[result[5]] = result[6][result[5]]
                self.fit_f[result[5]] = result[11][result[5]]
                self.cat.loc[result[0], 'l_fit'] = result[1]
                self.cat.loc[result[0], 'a_fit'] = result[2]
                self.cat.loc[result[0], 'sl_fit'] = result[3]
                self.cat.loc[result[0], 'sg_fit'] = result[4]

                self.continuum[result[5]] = result[7]
                self.template_f[result[5]] += result[7]
                self.fit_f[result[5]] += result[7]
                self.fit_residuals[result[5]]\
                = self.spec_f[result[5]] - self.fit_f[result[5]]

                self.template_f_highres[result[15]] = result[16][result[15]]
                self.fit_f_highres[result[15]] = result[14][result[15]]

                self.continuum_highres[result[15]] = result[17]
                self.template_f_highres[result[15]] += result[17]
                self.fit_f_highres[result[15]] += result[17]

                self.cat.loc[result[0], 'l_start'] = result[8]
                self.cat.loc[result[0], 'l_end'] = result[9]
                self.cat.loc[result[0], 'cont_order'] = result[10]

                self.cat.loc[result[0], 'significance'] = result[12]

            if result[13]:
                self.logger.warning(result[0]\
                + ' failed and will be marked in the catalog')
                self.cat.loc[result[0], 'used'] = 'f'

        elapsed_time = time.time() - start_time
        self.logger.info('Finished line fitting')
        self.logger.info('Elapsed time: '\
        + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    def rv_fit_peak(self, line_sigma=3, line_significants=5):
        '''
        This module determines the radial velocity solely on the fitted peaks
        of the spectral lines

        Kwargs:
            line_sigma : :obj:`int` (optional, default: 3)
                The sigma-level for the sigma clipping before determining peak
                radial velocity measurement

            line_significants: :obj:`int` (optional, default: 5)
                The sigma-level for the spectral line to be above the continuum
                in order to be considered *valid*
        '''

        self.logger.info('Calculating RVs based on peaks only')
        v = np.array((self.cat.loc[:, 'l_fit'].values\
        / self.cat.loc[:, 'l_lab'] - 1.) * const.c.to('km/s').value)

        for i, vi in enumerate(v):
            self.logger.info('Line ' + self.cat.index[i]\
            + ': RV=' + str('{:4.2f}'.format(vi)) + ' km/s')

        remaining_lines\
        = line_clipping(self, v, line_significants, sigma=line_sigma)
        self.n_lines_peak = len(v[~remaining_lines.mask])

        if remaining_lines.mask.all():
            self.logger.error('NO USABLE LINE FOUND WITH SET PARAMETER !!')
        else:
            self.rv_peak = np.median(v[~remaining_lines.mask])
            self.erv_peak = MAD(v[~remaining_lines.mask])

            self.logger.info('Finished Calculating RVs based on peaks only: '\
            + 'RV=(' + str('{:4.2f}'.format(self.rv_peak)) + '+-'\
            + str('{:4.3f}'.format(self.erv_peak)) + ')km/s based on '\
            + str(self.n_lines_peak)\
            + '/' + str(len(v)) + ' lines')

    def rv_fit(self, guesses, niter=10000, line_sigma=3,\
    n_CPU=-1, line_significants=5, RV_guess_var=0.):

        """
        The module runs the radial velocity fit using `ppxf`_ and the
        :ref:`Monte Carlo`.

        Args:
            guesses : :func:`numpy.array`
                The initial guesses for the the radial velocity fit guesses in
                the form [RV,sepctral_dispersion]

        Kwargs:
            niter : :obj:`int` (optional, default: 10000)
                number of iterations to bootstrap the spectrum

            line_sigma: :obj:`int` (optional, default: 3):
                sigma for the RV clipping for the individual lines

            n_CPU : :obj:`float` (optional, default: -1)
                Setting the number of CPUs used for the parallelization. If set
                to -1 all available system resources are used. Maximum number
                of CPUs is the number of spectral lines the fit is performed
                to.

            line_significants: :obj:`int` (optional, default: 5)
                The sigma-level for the spectral line to be above the continuum
                in order to be considered *valid*

            RV_guess_var : :obj:`float` (optional, default: 0)
                The maximum variation the RV guess will be varied using a
                uniform distribution.

        """

        self.logger.info('Starting the RV fitting')
        self.logger.info('Settings: niter=' + str(niter)\
        + '; sigma RV for excluding lines: ' + str(line_sigma))

        start_time = time.time()

        if self.loglevel == "DEBUG":
            n_CPU = 1
        if n_CPU == -1:
            n_CPU = cpu_count()
        self.logger.info('Max number of cores: '\
        + str(n_CPU))

        ### transfer the spectrum to log-space
        log_spec_f, logspec_lambda, velscale_spec\
        = ppxf_util.log_rebin([self.spec_lambda[0],\
        self.spec_lambda[-1]], self.spec_f)

        log_template_f, log_template_lambda, velscale_template\
        = ppxf_util.log_rebin([self.spec_lambda[0],\
        self.spec_lambda[-1]], self.template_f)

        log_spec_err, log_spec_err_lambda, velscale_spec_err\
        = ppxf_util.log_rebin([self.spec_lambda[0],\
        self.spec_lambda[-1]], self.spec_err)

        log_spec_err[~np.isfinite(log_spec_err)] = 1.
        # This happens if AO was used and there is a gap in the spec

        exponent = int(np.log10(np.nanmedian(log_spec_f)) - 4.)
        # Obtain larger flux numbers to make it numerically stable

        factor = float(10 ** (exponent * (-1)))

        log_spec_f = log_spec_f * factor
        log_template_f = log_template_f * factor
        log_spec_err = log_spec_err * factor

        l_fit = self.cat.loc[:, 'l_fit'].values.astype(np.float64)
        l_lab = self.cat.loc[:, 'l_lab'].values.astype(np.float64)
        line_name = self.cat.index

        fwhm_g, fwhm_l, fwhm_v\
        = voigt_FWHM(self.cat.loc[:, 'sg_fit'].values.astype(np.float64),\
        self.cat.loc[:, 'sl_fit'].values.astype(np.float64))

        used = np.where(self.cat.loc[:, 'used'] == 'f')
        l_fit = np.delete(l_fit, used)
        l_lab = np.delete(l_lab, used)
        line_name = np.delete(line_name, used)
        fwhm_g = np.delete(fwhm_g, used)
        fwhm_l = np.delete(fwhm_l, used)
        fwhm_v = np.delete(fwhm_v, used)

        v = np.zeros_like(l_lab)
        ev = np.zeros_like(l_lab)

        for i, line in enumerate(l_lab):

            self.logger.info('Started line ' + line_name[i])
            if self.cat.loc[line_name[i], 'significance'] > line_significants:
                mask = np.zeros(len(log_template_f), dtype=bool)
                ind_min = np.argmin(np.abs(log_template_lambda\
                - np.log(line - 0.5 * fwhm_v[i]))) - 1

                ind_max = np.argmin(np.abs(log_template_lambda\
                - np.log(line + 0.5 * fwhm_v[i]))) + 1

                ind_center = np.argmin(np.abs(log_template_lambda\
                - np.log(line)))
                mask[ind_min:ind_max + 1] = True

                log_spec_f[~np.isfinite(log_spec_f)] = 0.

                pp_outliers_init = ppxf.ppxf(log_template_f, log_spec_f,\
                log_spec_err, velscale_spec, guesses,\
                mask=mask, degree=-1, clean=False, quiet=True,\
                plot=False, fixed=[0, 1])

                self.logger.info('RV guess variation line ' + line_name[i]\
                + ': ' + str('{:4.2f}'.format(RV_guess_var)) + 'km/s')

                v[i], ev[i] = ppxf_MC(log_template_f, log_spec_f,\
                log_spec_err, velscale_spec, guesses, nrand=0.5 * niter,\
                goodpixels=pp_outliers_init.goodpixels, degree=-1,\
                moments=2, RV_guess_var=RV_guess_var, n_CPU=n_CPU)

                self.logger.info('Finished line ' + line_name[i]\
                + ': RV=(' + str('{:4.2f}'.format(v[i])) + '+-'\
                + str('{:4.3f}'.format(ev[i])) + ')km/s')

            else:
                self.logger.info(line_name[i]\
                + ': Line not significant [level ='\
                + str(line_significants) + ']')
                ev[i] = np.nan
                v[i] = np.nan

            self.cat.loc[line_name[i], 'RV'] = v[i]
            self.cat.loc[line_name[i], 'eRV'] = ev[i]

        remaining_lines = line_clipping(self, v, line_significants,\
        sigma=line_sigma)

        if remaining_lines.mask.all():
            self.logger.error('NO USABLE LINE FOUND WITH SET PARAMETER !!')
        else:
            self.cat.loc[line_name[~remaining_lines.mask], 'used'] = 'x'

            l_fit = l_fit[~remaining_lines.mask]
            fwhm_v = fwhm_v[~remaining_lines.mask]

            rv_var_lines = MAD(self.cat.loc[line_name[~remaining_lines.mask],\
            'RV'])
            RV_guess_var_min = RV_guess_var
            if rv_var_lines > RV_guess_var:
                RV_guess_var = rv_var_lines

            self.logger.info('RV guess variation: min = '\
            + str(RV_guess_var_min) + 'km/s; used = '\
            + str('{:4.2f}'.format(RV_guess_var)) + 'km/s')

            mask = np.zeros(len(self.spec_lambda), dtype=bool)

            for i, line in enumerate(l_fit):
                ind_min = np.argmin(np.abs(logspec_lambda\
                - np.log(line - 0.5 * fwhm_v[i]))) - 1
                ind_max = np.argmin(np.abs(logspec_lambda\
                - np.log(line + 0.5 * fwhm_v[i]))) + 1
                ind_center = np.argmin(np.abs(logspec_lambda - np.log(line)))
                mask[ind_min:ind_max + 1] = True

            # pp_final_plot = plt.figure(str(self.spec_id)\
            # + '_ppxf_fit_final', figsize=(10, 3))
            if len(l_lab) > 1:
                pp_final_init = ppxf.ppxf(log_template_f, log_spec_f,\
                log_spec_err, velscale_spec, guesses, degree=-1, clean=False,\
                mask=mask, quiet=True, fixed=[0, 1])

            if len(l_lab) == 1:
                pp_final_init = pp_outliers_init
            # pp_final_init.plot()
            # # plt.tight_layout()
            # plt.savefig(self.spec_id + '_ppxf_fit_final.png', dpi=600)
            # plt.close()

            self.logger.info('Started final RV fit')
            self.rv, self.erv = ppxf_MC(log_template_f, log_spec_f,\
            log_spec_err, velscale_spec, guesses,\
            nrand=niter, goodpixels=pp_final_init.goodpixels, degree=-1,\
            moments=2, spec_id=self.spec_id, RV_guess_var=RV_guess_var,\
            n_CPU=n_CPU)

            elapsed_time = time.time() - start_time
            self.logger.info('Used lines for RV fit: '\
            + ', '.join(line_name[~remaining_lines.mask]))

            self.logger.info('Finished RV fitting: RV=('\
            + str('{:4.2f}'.format(self.rv)) + '+-'\
            + str('{:4.3f}'.format(self.erv)) + ')km/s based on '\
            + str(len(l_fit)) + '/' + str(len(line_name)) + ' lines')

            self.logger.info('Elapsed time: '\
            + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
