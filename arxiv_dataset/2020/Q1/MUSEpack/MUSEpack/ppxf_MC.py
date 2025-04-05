#!/usr/bin/env python

__version__ = '0.1.1'

__revision__ = '20190731'

import numpy as np
from astropy.stats import sigma_clip
from lmfit import Model
from ppxf import ppxf
from multiprocessing import cpu_count
from functools import partial
import matplotlib.pyplot as plt
from astropy.stats import median_absolute_deviation as MAD
import logging
import dask


def ppxf_MC(log_template_f, log_spec_f, log_spec_err, velscale, guesses,\
            nrand=100, degree=4, goodpixels=None, moments=4, vsyst=0, sigma=5,\
            spec_id=None, RV_guess_var=0., n_CPU=-1):
    '''
    This module runs the Monte Carlo `ppxf`_ runs, which is needed for the RV
    measurements. Most of the input parameters are similar to the standard
    ppxf parameters (see `Cappellari and Emsellem 2004`_) for a more detailed
    explanation).

    Args:
        log_template_f : :func:`numpy.array`
            The logarithmically binned template spectrum

        log_spec_f : :func:`numpy.array`
            The logarithmically binned source spectrum

        log_spec_err : :func:`numpy.array`
            The logarithmically source spectrum uncertainties

        velscale: :obj:`float`
            The velocity scale of the source spectrum

        guesses : :func:`numpy.array`
            The initial guesses for the the radial velocity fit guesses in
            the form [RV,sepctral_dispersion]

    Kwargs:
        nrand : :obj:`int` (optional, default: 5)
            The maximum number of iteration if convergence is not reached

        degree : :obj:`int` (optional, default: 4)
            The degree of the additive polynomial to fit offsets in the
            continuum. A low order polynominal may be needed to get better
            results

        goodpixels : :func:`numpy.array` (optional, default: 5)
            A :func:`numpy.array`: of pixels that are used by ppxf to fit
            the template to the source spectrum

        moments : :obj:`int` (optional, default: 4)
            The moments to be fit (v, sigma, h3, h4)

        vsyst : :obj:`float` (optional, default: 0.)
            A systematic velocity. This may be needed if the system move at
            high velocities compared to the rest frame. If the guess of vsyst
            is good, the fit runs faster and more stable.

        sigma : :obj:`int` (optional, default: 5)
            The sigma used to clip outliers in the histogram determination.

        spec_id : :obj:`str` (optional, default: None)
            An ID number of the source spectrum. This becomes handy when
            fitting many individual sources because the output files will be
            named with the ID.
            
        RV_guess_var : :obj:`float` (optional, default: 0)
            The maximum variation the RV guess will be varied using a 
            uniform distribution.

        n_CPU : :obj:`int` (optional, default: -1)
            The number of cores used for the Monte Carlo velocity fitting. If
            n_CPU=-1 than all available cores will be used.

    '''

    noise = np.ones_like(log_spec_f) # Constant error spectrum

    if goodpixels.any() == None:
        goodpixels = np.arange(len(log_spec_f))

    iter_spec = log_spec_f.copy()

    results = [dask.delayed(_ppxf_bootstrap)(log_template_f, log_spec_f,\
    log_spec_err, velscale, degree, goodpixels, guesses, moments, vsyst,\
    iter_spec, noise, RV_guess_var) for n in np.arange(nrand)]

    if n_CPU == 1:
        uncert_ppxf = dask.compute(*results, num_workers=1,\
        scheduler='single-threaded')

    if n_CPU > 0:
        uncert_ppxf = dask.compute(*results, num_workers=n_CPU,\
        scheduler='processes')

    clipped = sigma_clip(np.array(uncert_ppxf)[:, 0], sigma=sigma, stdfunc=MAD)
    clipped = clipped.data[~clipped.mask]

    if not spec_id:
        ret = np.histogram(clipped, bins=int(20), density=True)
        bins = [ret[1][i] + 0.5 * (ret[1][i + 1] - ret[1][i])\
        for i in range(len(ret[0]))]

        result = Model(_gaussian_fit).fit(ret[0], x=bins, a=np.max(ret[0]),\
        b=np.mean(clipped), c=np.std(clipped))

        popt = [result.best_values['a'], result.best_values['b'],\
        result.best_values['c']]

    if spec_id:
        plt.figure(spec_id)
        ret = plt.hist(clipped, bins=int(20), density=True,\
        label='n\ realizations: ' + str(nrand))

        bins = [ret[1][i] + 0.5 * (ret[1][i + 1] - ret[1][i])\
        for i in range(len(ret[0]))]

        binlength = bins[1] - bins[0]

        result = Model(_gaussian_fit).fit(ret[0], x=bins, a=np.max(ret[0]),\
        b=np.mean(clipped), c=np.std(clipped))

        popt = [result.best_values['a'], result.best_values['b'],\
        result.best_values['c']]

        x = ((np.arange(20000 * len(bins)) - 10000 * len(bins))\
        / (100 * len(bins))) + popt[1]

        plt.plot(x, _gaussian_fit(x, *popt), lw=3)
        plt.axvline(x=popt[1], c='r', lw=3,\
        label=r'mean$=$' + str('{:4.2f}'.format(popt[1]))\
        + r'$\,\frac{\rm km}{\rm s}$')

        plt.axvline(x=popt[1] + popt[2], c='r', linestyle='--', lw=3,\
        label=r'$1\sigma = $' + str('{:5.3f}'.format(popt[2]))\
        + r'$\,\frac{\rm km}{\rm s}$')

        plt.axvline(x=popt[1] - popt[2], c='r', linestyle='--', lw=3)
        plt.xlim(popt[1] - 3 * popt[2], popt[1] + 3 * popt[2])
        plt.xlabel(r'velocity [$\frac{\rm km}{\rm s}$]')
        plt.ylabel(r'relative number')
        plt.legend()
        plt.tight_layout()
        plt.savefig(spec_id + '_v_dist.png', dpi=600)
        plt.close()

    return popt[1], popt[2]


def _ppxf_bootstrap(log_template_f, log_spec_f, log_spec_err, velscale,\
    degree, goodpixels, guesses, moments, vsyst, iter_spec, noise,\
    RV_guess_var):

    '''
    This is the bootstrap Monte Carlo module of the ppxf MC code.
    Use negligible BIAS=0 to estimate Bootstrap errors. See Section 3.4 of
    Cappellari & Emsellem (2004).

    Args:
        log_template_f : :func:`numpy.array`
            The logarithmically binned template spectrum

        log_spec_f : :func:`numpy.array`
            The logarithmically binned source spectrum

        log_spec_err : :func:`numpy.array`
            The logarithmically source spectrum uncertainties

        velscale : obj:`float`
            The velocity scale of the source spectrum

        degree : :obj:`int`
            The degree of the additive polynomial to fit offsets in the
            continuum. A low order polynominal may be needed to get better
            results

        goodpixels : :func:`numpy.array`
            A :func:`numpy.array`: of pixels that are used by ppxf to fit
            the template to the source spectrum

        guesses: :func:`numpy.array`
            the guesses for ppxf. For two moments it reflects (RV, sigma)

        moments : :obj:`int`
            The moments to be fit (v, sigma, h3, h4)

        vsyst : :obj:`float`
            A systematic velocity. This may be needed if the system move at
            high velocities compared to the rest frame. If the guess of vsyst
            is good, the fit runs faster and more stable.

        iterspec: :func:`numpy.array`
            a copy of log_template_f that can be manipulated in each step

        noise: :func:`numpy.array`
            constant error spectrum

    '''


    pm = np.random.randint(1, 3, size=len(log_spec_err))
    pm[np.where(pm == 2)] = -1
    pm_log_spec_err = pm * log_spec_err

    np.random.seed()
    scramble = goodpixels[(np.random.rand(len(goodpixels))\
    * len(goodpixels)).astype(int)]

    new_log_spec_err = pm_log_spec_err[scramble]
    iter_spec[goodpixels] = log_spec_f[goodpixels] + new_log_spec_err

    rv_var = np.random.uniform(-1.,1.) * RV_guess_var
    var_guesses =  [sum(x) for x in zip(guesses, [rv_var, 0.])]

    pp1 = ppxf.ppxf(log_template_f, iter_spec, noise, velscale, var_guesses,\
    goodpixels=goodpixels, plot=False, degree=degree, moments=moments,\
    vsyst=vsyst, quiet=1, bias=0, velscale_ratio=1, fixed=[0, 1])

    if moments <= 2:
        uncert = np.array([pp1.sol[0], pp1.sol[1]])
    if moments > 2:
        uncert = np.array([pp1.sol[0], pp1.sol[1], pp1.sol[2], pp1.sol[3]])

    return uncert


def _gaussian_fit(x, a, b, c):

    '''
    A function for a gaussian, which is conform with the fitting

    '''

    g = a * np.exp((-1) * (x - b) ** 2.0 / (2 * c ** 2))
    return g
