__version__ = '1.0.1'

__revision__ = '20210128'

import sys
import os
import shutil
import warnings
import pyspeckit
import numpy as np
from astropy import units as u
from astropy import log
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from scipy.special import wofz
import time
import logging
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from MUSEpack.utils import *


def line_fitter(self, linecat, line_idx, niter,\
input_resid_level, max_contorder, max_ladjust, adjust_preference,\
input_continuum_deviation, llimits, max_exclusion_level, blends,\
autoadjust, fwhm_block):

    '''
    This is module fits the spectral lines using pyspeckit. It automatically
    determines the goodness of fit and decides the best solution for the line
    and continuum fit. It handles blended lines in a way a maximum lines ratio
    can not be exceeded to not make the weaker blend the dominant lines.
    Additionally, the limits in wavelength range in iteration :math:`n` can be
    adjusted automatically based on iteration :math:`n` to accomodate for
    larger wavelength shifts.

    In this way it is possible to fit spectral lines using a line catalog as 
    only input.

    Args:
        linecat : :func:`numpy.array`
            Array with the spectral lines and their wavelengths

        line_idx : :obj:`str`
            Name of the primary line

        niter : :obj:`int`
            Number of iterations

        input_resid_level : :obj:`float`
            The maximum MAD for the fit residuals for a succesfull fit

        max_contorder : :obj:`int`
            The maximum polynominal order of the continuum to have

        max_ladjust : :obj:`str`
            The maximum number of wavelength range adjustments in steps of 5
            Angstrom

        adjust_preference : :obj:`str`
            contorder: continuum order is adjusted first

            wavelength: wavelength range is adjusted first

        input_continuum_deviation : :obj:`float`
            by how much the continuum is allowed to deviate from a running
            median estimate. This is set to prevent lines mimicking a continuum

        llimits : :obj:`list`
            the limits for the wavelength fit as set in ``ppxf``

        max_exclusion_level : :obj:`float`
            The exclusion level for lines to be excluded from the next baseline
            estimate as set in ``pyspeckit``

        blends : :obj:`ascii`-file or :obj:`None`
            A file with primary lines that contain blends to provide a maximum
            amplitude ratio of the primary and the blend to prevent that the
            blend becomes the dominant line in the fit.

        autoadjust : :obj:`bool`
            :obj:`True`: the wavelength limits ``llimit`` will be adjusted to
            the fit of the previous iteration. All other wavelength range are
            adjusted accordingly taking into account the proper velocity
            corrected shift :math:`\Delta \lambda/\lambda`. This is especially
            important to detect hyper-velocity stars.

            :obj:`False`: no adjustment to the limits done

        fwhm_block : :obj:`bool:obj:`
            :obj:`True`: The minimum fwhm of the voigt profiles of the fitted
            lines is the instrument's dispersion

            :obj:`False`: The minimum fwhm of the voigt profiles of the fitted
            lines is zero

    Returns:
        line_idx : :obj:`str`
            Name of the primary line

        temp_l : :obj:`float`
            fitted wavelength of the primary line

        temp_a : :obj:`float`
            fitted amplitude of the primary line

        temp_sl : :obj:`float`
            fitted Lorentzian gamma of the primary line

        temp_sg : :obj:`float`
            fitted Gaussian sigma of the primary line

        spec_select_idx : :func:`numpy.array`
            indices of the used part of the spectrum

        template_f : :func:`numpy.array`
            template spectrum shifted to the rest-frame

        continuum : :func:`numpy.array`
            the fitted continuum

        lstart : :obj:`float`
            first used wavelength bin (might differ from input if it was
            adjusted during fitting)

        lend : :obj:`float`
            last used wavelength bin (might differ from input if it was
            adjusted during fitting)

        contorder : :obj:`int`
            The order of the polynominal used for the continuum

        fit_f : array
            the fitted spectrum

        significance : :obj:`float`
            the line strength over the continuum

        fit_failed : :obj:`bool`
            :obj:`True`: if the fit failed for some reason. This line will be
            excluded from further analyses

        fit_f_highres : :obj:`float`
            the fitted spectrum and the oversampling resolution

        spec_select_idx_highres : :func:`numpy.array`
            indices of the used part of the over sampled spectrum

        template_f_highres : :func:`numpy.array`
            over sampled template spectrum shifted to the rest-frame

        continuum_highres : :func:`numpy.array`
            the fitted over sampled continuum

    '''

    self.logger.info('Started line ' + line_idx)

    resid_level = 5.
    std_resid = resid_level + 1.
    continuum_dev = input_continuum_deviation[line_idx] + 1.

    lstart = float(self.cat.loc[line_idx, 'l_start'])
    lend = float(self.cat.loc[line_idx, 'l_end'])
    contorder = self.cat.loc[line_idx, 'cont_order']

    n_ladjust = 0
    max_cont_reached = False
    max_n_ladjust_reached = False
    significant = True
    fit_failed = False

    niter += 1 # Iteration 1 are setting the intial guesses

    while (std_resid > resid_level or np.isnan(std_resid)\
    or continuum_dev > input_continuum_deviation[line_idx]\
    or np.isnan(continuum_dev)) and not fit_failed:

        lines_select = linecat[np.where((linecat >= lstart)\
        & (linecat < lend))]

        spec_select_idx\
        = np.where((self.spec_lambda >= lstart) & (self.spec_lambda < lend))

        spec_select_idx_highres\
        = np.where((self.spec_lambda_highres >= lstart) &\
        (self.spec_lambda_highres < lend))

        linefit_guess, linefit_limits, linefit_limited =\
        initial_guesses(self, lines_select, blends, llimits=llimits)

        exponent\
        = int(np.log10(np.nanmedian(self.spec_f[spec_select_idx])) - 4.)
        factor = float(10 ** (exponent * (-1)))

        template_f = np.zeros_like(self.spec_lambda, dtype=float)
        fit_f = np.zeros_like(self.spec_lambda, dtype=float)
        temp_f = np.array(self.spec_f[spec_select_idx] * factor, dtype=float)
        temp_err = np.array(self.spec_err[spec_select_idx] * factor,\
        dtype=float)
        temp_lambda = np.array(self.spec_lambda[spec_select_idx], dtype=float)
        continuum = np.zeros_like(self.spec_lambda, dtype=float)

        template_f_highres\
        = np.zeros_like(self.spec_lambda_highres, dtype=float)

        fit_f_highres = np.zeros_like(self.spec_lambda_highres, dtype=float)
        temp_lambda_highres\
        = np.array(self.spec_lambda_highres[spec_select_idx_highres],\
        dtype=float)

        continuum_highres\
        = np.zeros_like(self.spec_lambda_highres, dtype=float)

        if input_resid_level == None:
            resid_level = np.std(temp_err / np.max(temp_err))
        else:
            resid_level = input_resid_level

        sp = pyspeckit.Spectrum(data=temp_f, error=temp_err, xarr=temp_lambda,\
                                unit=r'flux [$10^{' + str(exponent)\
                                + '}$ erg$^{-1}$ s$^{-1}$ cm$^{-2}$ '\
                                + '$\AA^{-1}$]', header={})
        sp.xarr.set_unit(u.AA)
        sp.xarr.xtype = 'wavelength'

        if self.loglevel == "DEBUG":
            plt.ion()
            sp.plotter(linewidth=2, title=line_idx)

        exclusion_level = 0.01
        iterations = 0
        pat = []

        while iterations < niter:
            if iterations == 1:
                newparinfo = update_parinfo(self, linefit_guess, \
                llimits, line_idx, blends, sp.specfit.parinfo, False, False)

            if iterations > 1:
                newparinfo = update_parinfo(self, linefit_guess,\
                llimits, line_idx, blends, sp.specfit.parinfo,\
                autoadjust, fwhm_block)

            with log.log_to_list() as log_list:
                sp.baseline(order=int(contorder), plot_baseline=False,\
                subtract=False, annotate=False, highlight_fitregion=False,\
                excludefit=True, save=True, exclude=None,\
                fit_plotted_area=False, exclusionlevel=float(exclusion_level))

                if iterations == 0:
                    sp.specfit.multifit(fittype='voigt', renormalize='auto',\
                    annotate=False, show_components=False, verbose=False,\
                    color=None, guesses=list(linefit_guess), parinfo=None,\
                    reset_fitspec=True, plot=False, limits=linefit_limits,\
                    limited=linefit_limited)

                if iterations > 0:
                    sp.specfit.multifit(fittype='voigt', renormalize='auto',\
                    annotate=False, show_components=False, verbose=False,\
                    color=None, parinfo=newparinfo,\
                    reset_fitspec=True, plot=False)

                if exclusion_level >= max_exclusion_level:
                    self.logger.error(line_idx + ': LINE FIT FAILED !!!!!: \
                    Exclusion level reached max of '\
                    + str(max_exclusion_level))
                    fit_failed = True
                    break

                if len(log_list) > 0 and log_list[0].message[:8] == 'gnorm=0.'\
                and exclusion_level < max_exclusion_level:
                    exclusion_level += 0.01
                    self.logger.info(line_idx + ': Adjusting exclusion level'\
                    + ' to: ' + str('{:2.2f}'.format(exclusion_level)))
                    iterations = 0
                    sp = pyspeckit.Spectrum(data=temp_f, error=temp_err,\
                    xarr=temp_lambda, unit=r'flux [$10^{' + str(exponent)\
                    + '}$ erg$^{-1}$ s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]',\
                    header={})
                    sp.xarr.set_unit(u.AA)
                    sp.xarr.xtype = 'wavelength'

                    if self.loglevel == "DEBUG":
                        sp.plotter()

                if len(log_list) == 0 or log_list[0].message[:8] != 'gnorm=0.':
                    if iterations > 0:
                        chi2_change =\
                        (chi2 - sp.specfit.optimal_chi2(reduced=True,\
                        threshold='auto'))\
                        / sp.specfit.optimal_chi2(reduced=True,\
                        threshold='auto')

                        if sp.specfit.optimal_chi2(reduced=True,\
                        threshold='auto') == chi2\
                        or np.abs(chi2_change) < 0.05:
                            break

                        self.logger.debug(line_idx + ': Iteration: '\
                        + str(iterations) + ' delta Chi2: '\
                        + str('{:2.3f}'.format(chi2_change)))

                    chi2 = sp.specfit.optimal_chi2(reduced=True,\
                    threshold='auto')
                    self.logger.debug(line_idx + ': Iteration: '\
                    + str(iterations) + ' Chi2: '\
                    + str('{:3.3f}'.format(chi2)))

                    if self.loglevel == "DEBUG":
                        sp.specfit.plot_components(axis=sp.plotter.axis,\
                        add_baseline=True,\
                        component_fit_color=plotcolor(iterations))
                        ax = sp.plotter.axis
                        pat.append(mlines.Line2D([], [],\
                        color=plotcolor(iterations),\
                        label='# iter: ' + str(iterations)))
                        ax.legend(handles=pat)

                    iterations += 1

                if self.loglevel == "DEBUG":
                    input("Press ENTER to continue")

        if self.loglevel == "DEBUG":
            sp.specfit.plotresiduals(axis=sp.plotter.axis, clear=False,\
            yoffset=0.9 * np.min(temp_f), label=False, linewidth=2, color='g')
            sp.specfit.plot_fit(annotate=False, lw=2, composite_fit_color='r')
            sp.baseline.plot_baseline(annotate=False,\
            baseline_fit_color='orange', linewidth=2)

            print('Parinfo:')
            print(sp.specfit.parinfo)
            print(' ')
            input("Press ENTER to close the plot window")

        if not fit_failed:
            if iterations < niter:
                self.logger.info(line_idx + ': Converged after '\
                + str(iterations) + ' iterations; Chi2: '\
                + str('{:3.3f}'.format(chi2)) + ' delta Chi2 [%]: '\
                + str('{:2.3f}'.format(chi2_change)))

            if iterations == niter:
                self.logger.info(line_idx\
            + ': Maximum number of iterations (' + str(iterations)\
            + ') reached; Chi2: ' + str(chi2))

            par_extract_idx = []
            for lab_line_idx, lab_lines\
            in enumerate(np.array([self.cat.loc[line_idx, 'l_lab']])):
                par_extract_idx = np.concatenate((par_extract_idx,\
                np.where(lines_select == lab_lines)[0]))

            for i in range(len(lines_select)):
                xcen = lines_select[i]
                gamma = sp.specfit.parinfo[int(4 * i + 3)]['value']
                sigma = sp.specfit.parinfo[int(4 * i + 2)]['value']
                amp = sp.specfit.parinfo[int(4 * i + 0)]['value']

                highres_f = voigt_funct(self.spec_lambda_highres, xcen, 1,\
                sigma, gamma)
                lowres_f = spec_res_downgrade(self.spec_lambda_highres,\
                highres_f, self.spec_lambda)

                lineflux = amp * lowres_f
                lineflux_highres = amp * highres_f

                template_f += lineflux / factor
                template_f_highres += lineflux_highres / factor

                highres_f = voigt_funct(self.spec_lambda_highres,\
                sp.specfit.parinfo[int(4 * i + 1)]['value'], 1, sigma, gamma)
                lowres_f = spec_res_downgrade(self.spec_lambda_highres,\
                highres_f, self.spec_lambda)

                lineflux_fit = amp * lowres_f
                lineflux_fit_highres = amp * highres_f

                fit_f += lineflux_fit / factor
                fit_f_highres += lineflux_fit_highres / factor

                if (i == par_extract_idx).any():

                    temp_a = sp.specfit.parinfo[int(4 * i + 0)]['value']\
                    / factor
                    temp_l = sp.specfit.parinfo[int(4 * i + 1)]['value']
                    temp_sg = sp.specfit.parinfo[int(4 * i + 2)]['value']
                    temp_sl = sp.specfit.parinfo[int(4 * i + 3)]['value']

            baseline_temp = baseline_extract(self, sp, contorder)
            baseline_sub_spec =\
            sp.data / baseline_temp(temp_lambda - temp_lambda[0])
            continuum = baseline_temp(temp_lambda - temp_lambda[0]) / factor
            continuum_highres = baseline_temp(temp_lambda_highres\
            - temp_lambda_highres[0]) / factor

            resid = \
            (self.spec_f[spec_select_idx] - fit_f[spec_select_idx]) / continuum

            std_resid = np.std(resid)

            continuum_dev = continuum_deviation(self, temp_lambda,\
            temp_f, continuum, contorder)

            if std_resid > resid_level or np.isnan(std_resid)\
            or continuum_dev > input_continuum_deviation[line_idx]\
            or np.isnan(continuum_dev):

                if std_resid > resid_level or np.isnan(std_resid):
                    self.logger.info(line_idx\
                    + ': STD of residuals: '\
                    + str('{:2.4f}'.format(std_resid))\
                    + ' targeted: ' + str('{:2.4f}'.format(resid_level)))

                if continuum_dev > input_continuum_deviation[line_idx]:
                    self.logger.info(line_idx + ': Continuum deviation: '\
                    + str('{:2.4f}'.format(continuum_dev)) + ' targeted: '\
                    + str('{:2.4f}'.format(input_continuum_deviation[line_idx])))

                if contorder == max_contorder:
                    max_cont_reached = True
                if n_ladjust == max_ladjust:
                    max_n_ladjust_reached = True

                if max_cont_reached and max_n_ladjust_reached:
                    self.logger.warning(line_idx\
                    + ": Maximum adjustments reached: Check the output")
                    break

                if adjust_preference == 'contorder':
                    if not max_cont_reached:
                        contorder += 1
                        self.logger.info(line_idx\
                        + ': Adjusting continuum order to: ' + str(contorder))
                    if max_cont_reached and not max_n_ladjust_reached:
                        lstart -= 5.
                        lend += 5.
                        self.logger.info(line_idx\
                        + ': maximum continuum order reached => Adjusting'\
                        + ' wavelength range to: [' + str(lstart)\
                        + ',' + str(lend) + ']')
                        n_ladjust += 1
                        max_cont_reached = False
                        contorder = self.cat.loc[line_idx, 'cont_order']

                if adjust_preference != 'contorder':
                    contorder = self.cat.loc[line_idx, 'cont_order']
                    if not max_n_ladjust_reached:
                        if adjust_preference == 'min_lambda'\
                        or adjust_preference == 'minmax_lambda':
                            lstart -= 5.
                            self.logger.info(line_idx\
                            + ': Adjusting lower wavelength range to: '\
                            + str(lstart))

                        if adjust_preference == 'max_lambda'\
                        or adjust_preference == 'minmax_lambda':
                            lend += 5.
                            self.logger.info(line_idx + ': Adjusting upper \
                            wavelength range to: ' + str(lend))
                        n_ladjust += 1
                    if max_n_ladjust_reached and not max_cont_reached:
                        contorder += 1
                        self.logger.info(line_idx\
                        + ': maximum wavelength adjustment reached =>'\
                        + ' Adjusting continuum order to: ' + str(contorder))

                        lstart = self.cat.loc[line_idx, 'l_start']
                        lend = self.cat.loc[line_idx, 'l_end']

    if fit_failed:
        temp_l = 0.
        temp_a = 0.
        temp_sl = 0.
        temp_sg = 0.

    significance = np.abs(temp_a) / np.median(temp_err / factor)

    if not fit_failed:
        self.logger.info(line_idx + ': STD of residuals: '\
        + str('{:2.4f}'.format(std_resid)) + ' targeted: '\
        + str('{:2.4f}'.format(resid_level)))
        self.logger.info(line_idx + ': Continuum deviation: '\
        + str('{:2.4f}'.format(continuum_dev)) + ' targeted: '\
        + str('{:2.4f}'.format(input_continuum_deviation[line_idx])))
        self.logger.info(line_idx + ' Line significance: '\
        + str('{:3.2f}'.format(significance)))

        if (temp_l - self.cat.loc[line_idx, 'l_lab'] < 0.8 * llimits[0]\
        or temp_l - self.cat.loc[line_idx, 'l_lab'] > 0.8 * llimits[1])\
        and not autoadjust:
            self.logger.warning(line_idx\
            + ' exceeds 0.8 of lambda limits; lfit: ' + str(temp_l)\
            + ' llab: ' + str(self.cat.loc[line_idx, 'l_lab'])\
            + ' llimits = [' + str(llimits[0]) + ' ' + str(llimits[1]) + ']')

    self.logger.info('Finished line ' + line_idx)

    return line_idx, temp_l, temp_a, temp_sl, temp_sg, spec_select_idx,\
    template_f, continuum, lstart, lend, contorder, fit_f, significance,\
    fit_failed, fit_f_highres, spec_select_idx_highres, template_f_highres,\
    continuum_highres
