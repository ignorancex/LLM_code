#!/usr/bin/env python
  
# Compability with python2.x for x>6
from __future__ import print_function

# plot_many.py
# Martin Kilbinger (2020)


import sys
import os
import re
import subprocess
import copy

from optparse import OptionParser

import numpy as np
import collections

from CorrMatrix_ABC.covest import *


def params_default():
    """Set default parameter values.

    Parameters
    ----------
    None

    Returns
    -------
    p_def: class param
        parameter values
    """

    p_def = param(
        n_D = 750,
        ABC = 'mean_std_ABC',
        MCMC_norm = 'mean_std_fit_norm_deb',
        MCMC_T2 = 'mean_std_fit_SH',
        spar = '1.0 0.0',
        spar_name = 'a_b',
        model = 'affine',
        sig2 = 5.0,
        verbose = True,
        boxwidth = 0.15,
    )

    return p_def


def parse_options(p_def):
    """Parse command line options.

    Parameters
    ----------
    p_def: class param
        parameter values

    Returns
    -------
    options: tuple
        Command line options
    args: string
        Command line string
    """

    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    parser.add_option('-D', '--n_D', dest='n_D', type='int', default=p_def.n_D,
        help='Number of data points, default={}'.format(p_def.n_D))
 
    parser.add_option('-p', '--par', dest='spar', type='string', default=p_def.spar,
        help='True parameter values, default=\'{}\''.format(p_def.spar))
    parser.add_option('-P', '--par_name', dest='spar_name', type='string', default=p_def.spar_name,
        help='Parameter names, default=\'{}\''.format(p_def.spar_name))
    parser.add_option('-M', '--model', dest='model', type='string', default=p_def.model,
        help='Model, one in \'affine\', \'quadratic\', default=\'{}\''.format(p_def.model))
    parser.add_option('-s', '--sig2', dest='sig2', type='float', default=p_def.sig2,
        help='sigma^2, diagonal on covariance matrix, default=\'{}\''.format(p_def.sig2))

    parser.add_option('', '--ABC', dest='ABC', type='string', default=p_def.ABC,
        help='ABC result input file name base, default=\'{}\''.format(p_def.ABC))
    parser.add_option('', '--MCMC_norm', dest='MCMC_norm', type='string', default=p_def.MCMC_norm,
        help='MCMC normal result input file name base, default=\'{}\''.format(p_def.MCMC_norm))
    parser.add_option('', '--MCMC_T2', dest='MCMC_T2', type='string', default=p_def.MCMC_T2,
        help='MCMC T^2 (Hotelling) result input file name base, default=\'{}\''.format(p_def.MCMC_T2))

    parser.add_option('', '--sig_var_noise', dest='sig_var_noise', type='string',
        help='MCMC \'noise\' to be subtracted from sigma(var) plots for fits, default=None')

    parser.add_option('-b', '--boxwidth', dest='boxwidth', type='float', default=p_def.boxwidth,
        help=f'box width for box plot, default={p_def.boxwidth}, determined from n_S array')

    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', help='verbose output')

    options, args = parser.parse_args()

    return options, args


def check_options(options):
    """Check command line options.

    Parameters
    ----------
    options: tuple
        Command line options

    Returns
    -------
    erg: bool
        Result of option check. False if invalid option value.
    """

    return True


def update_param(p_def, options):
    """Return default parameter, updated and complemented according to options.
    
    Parameters
    ----------
    p_def:  class param
        parameter values
    optiosn: tuple
        command line options
    
    Returns
    -------
    param: class param
        updated paramter values
    """

    param = copy.copy(p_def)

    # Update keys in param according to options values
    for key in vars(param):
        if key in vars(options):
            setattr(param, key, getattr(options, key))

    # Add remaining keys from options to param
    for key in vars(options):
        if not key in vars(param):
            setattr(param, key, getattr(options, key))

    param.par_name = my_string_split(options.spar_name, verbose=False, stop=True)

    par = my_string_split(param.spar, num=2, verbose=param.verbose, stop=True)
    param.par = [float(p) for p in par]

    tmp = my_string_split(param.sig_var_noise, num=2, verbose=param.verbose, stop=True)
    if tmp != None:
        param.sig_var_noise = [float(s) for s in tmp]
    else:
        param.sig_var_noise = None

    return param


def read_fit(par_name, fbase, label, fct=None, flab=None, verbose=False):
    """Read fit mean and std and return Result

    Parameters
    ----------
    par_name: string
        parameter names
    fbase: string
        file base
    label: string
        label string
    fct: dictionary, optional, default=None
        functions for plotting of mean, std, std(var) predictions
    flab: dictionary, optional, default=None
        labels for mean, std, std(var) predictions
    verbose: bool, optional, default=False
        verbose output if True

    Returns
    -------
    fit: class Results
        results
    """

    fit = Results(par_name, 0, 0, fbase, yscale=['linear', 'log'], fct=fct)
    n_S_arr = fit.read_mean_std(update=True, verbose=verbose)

    fit.label = label
    fit.n_S_arr = n_S_arr
    fit.flab = flab

    return fit


def plot_box(fits, n_D, par, dy, which, boxwidth=None, xlog=False, ylog=False,
             sig_var_noise=None, model='affine', verbose=False):
    """Create box plot of various fits as function of number of realisations

    Parameters
    ----------
    fits: list of Results
        fit results
    n_D: integer
        dimension of data vector
    par: array of float
        input parameter values
    dy: array of float
        input parameter y range around mean
    boxwidth: float, optional
        box width for box plots, default: None, width is determined from n
    xlog: bool, optional
        logarithmic x-axis, default False
    ylog: bool, optional
        logarithmic y-axis, default False
    model: string, optional, default='affine'
        model, one in 'affine'  or 'quadratic'
    verbose: bool, optional, default=False
        verbose output if True

    Returns
    -------
    None
    """

    marker = ['.', 'D']
    markersize = [6] * len(marker)
    color = ['b', 'g', 'r']
    linestyle = ['-', '--', '-.']
    marker = ['D', 'o', 's']

    fac_xlim = 1.3

    plot_init(n_D, -1, fs=fits[0].fs)

    box_width = boxwidth
    rotation = 'vertical'

    if xlog == True:
        width   = lambda p, box_width: 10**(np.log10(p)+box_width/2.)-10**(np.log10(p)-box_width/2.)
        flinlog = lambda x: np.log10(x)
    else:
        width   = lambda p, box_width: np.zeros(len(n)) + float(box_width)
        flinlog = lambda x: x

    # Collect all n_S and count occurance
    all_n = []
    for fit in fits:
        all_n.append(fit.n_S_arr)
    all_n = np.array(all_n).flatten()
    count = collections.Counter(all_n)

    for i, p in enumerate(fits[0].par_name):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        leg = []
        labels = []
        leg2 = []
        labels2 = []
        x_loc = []
        x_lab = []

        if xlog == True:
            ax.set_xscale('log')
        xmin = 1e10
        xmax = -1

        for k, fit in enumerate(fits):

            # Get x- and y-values
            n = np.array(fit.n_S_arr)
            if which != 'std_var':
                y = np.array(getattr(fit, which)[p])
            else:
                y = fit.get_std_var(p)

            # Compute x shift to account for multiple occurances
            dn = []
            for n_S in n:
                if count[n_S] == 1:
                    dn.append(1)
                else:
                    dn.append((1+2.5*box_width)**(k-0.5*len(fits)))
            dn = np.array(dn)

            # Exclude points to increase visibility
            cnd = (n!=1257) & (n!=2703) & (n!=5811)
            idxl = np.where(cnd)
            rem = n[np.where(np.invert(cnd))]
            n = n[idxl]
            y = y[idxl]
            dn = dn[idxl]

            if i==0 and which=='mean' and verbose:
                print('{}: '.format(fit.label), end='')
                print(n)
            if which=='mean' and i==0 and verbose:
                print('{} removed: '.format(fit.label), end='')
                print(rem)

            if which != 'std_var':
                bplot = ax.boxplot(y.transpose(), 0, '{}.'.format(color[k]), positions=n*dn, widths=width(n, box_width), patch_artist=True)
                for key in bplot:
                    plt.setp(bplot[key], color=color[k], linewidth=1)
                plt.setp(bplot['whiskers'], linestyle=linestyle[k], linewidth=1)
                plt.setp(bplot['boxes'], facecolor=lighten_color(color[k], amount=0.25))
                leg.append(bplot['boxes'][0])
            else:
                pl, = ax.plot(n, y, marker=marker[k], color=color[k], linestyle='None')
                leg.append(pl)
            labels.append(fit.label)

            xmin = min(xmin, n[0])
            xmax = max(xmax, n[-1])

            # xticks
            for n_S in n:
                x_loc.append(n_S)
                lab = '{}'.format(n_S)
                x_lab.append(lab)

            # Curves for predicted values
            if fit.fct is not None and which in fit.fct:
                if n[0] == 2:
                    f0 = 2.0
                else:
                    f0 = 1.0
                f1 = 10.0
                n_fine = np.logspace(np.log10(n[0]/f0), np.log10(n[-1]*f1), 50)
                if k==0:
                    c = 'k'
                    ls = 'dotted'
                else:
                    c = color[k]
                    ls = linestyle[k]
                pl = ax.plot(n_fine, fit.fct[which](n_fine, n_D, par[i]), color=c, linestyle=ls)
                if fit.flab and which in fit.flab:
                    leg2.append(pl[0])
                    labels2.append(fit.flab[which])
                if sig_var_noise:
                    pl = ax.plot(n_fine, fit.fct[which](n_fine, n_D, par[i]) + no_bias(n_fine, n_D, sig_var_noise[i]),
                            color=color[k], linestyle='dotted')
                    leg2.append(pl[0])
                    labels2.append('{} + $\\sigma_{{\\mathrm{{noise}}}}$'.format(fit.flab[which]))


        # Dashed vertical line at n_S = n_D
        ax.plot([n_D, n_D], [-1e2, 1e2], 'k:', linewidth=1)
        ax.plot([n_D, n_D], [1e-5, 1e2], 'k:', linewidth=1)

        # Labels
        ax.set_xlabel('$n_{{\\rm s}}$', fontsize=fit.fs)
        if which == 'mean':
            s = rf'$\bar {p}$'
        elif which == 'std':
            s = rf'SE$({p})$'
        else:
            s = rf'SD$[$Var$({p})]$'
        plt.ylabel(s, fontsize=fit.fs)
        Leg2 = plt.legend(leg2, labels2, frameon=False, loc='upper right')
        plt.gca().add_artist(Leg2)
        ax.legend(leg, labels, loc='upper left', frameon=False)

        # Limits
        if xlog:
            Xmin = xmin / fac_xlim
            Xmax = xmax * fac_xlim
        else:
            fac_xlim   = 1.05
            if xmin > 10:
                Xmin = (xmin - 5) / fac_xlim**5
            else:
                Xmin = (xmin - 0.5) / fac_xlim**5
            Xmax = xmax * fac_xlim
        ax.set_xlim(Xmin, Xmax)

        if not ylog:
            ax.set_yscale('linear')
            miny = par[i] - dy[i]
            if which == 'std':
                miny = max(miny, 0)
            ax.set_ylim(miny, par[i] + dy[i])
        else:
            ax.set_yscale('log')
            ax.set_ylim(par[i] / dy[i], par[i] * dy[i])

        plt.xticks(x_loc, x_lab, rotation=rotation)

        # Second x-axis
        x_loc = fits[0].n_S_arr
        ax2 = plt.twiny()
        x2_loc = []
        x2_lab = []
        for i, n_S in enumerate(x_loc):
            x2_loc.append(flinlog(n_S))
            frac = float(n_D) / float(n_S)
            if frac > 100:
                lab = '{:.0f}'.format(frac)
            else:
                lab = '{:.2g}'.format(frac)
            x2_lab.append(lab)
        ax2.set_xticks(x2_loc)
        ax2.set_xticklabels(x2_lab)
        ax2.set_xlabel('$p / n_{\\rm s}$', size=fit.fs)
        if xlog:
            ax2.set_xlim(flinlog(x_loc[0] / fac_xlim), flinlog(x_loc[-1] * fac_xlim))

        fig.savefig('{}_{}.pdf'.format(which, p), bbox_inches='tight')


def main(argv=None):

    p_def = params_default()

    options, args = parse_options(p_def)

    if check_options(options) is False:
        return 1

    param = update_param(p_def, options)

    # Save calling command
    log_command(argv)
    if options.verbose:
        log_command(argv, name='sys.stderr')

    # Read sampling results
    fit_ABC = read_fit(param.par_name, param.ABC, 'ABC',
                       fct={'mean': no_bias}, flab={'mean': 'true value'}, verbose=param.verbose)
                             # fct={'std': no_bias, 'std_var': std_fish_deb_TJ14},
    fit_MCMC_norm = read_fit(param.par_name, param.MCMC_norm, 'MCMC $N$',
                             fct={'std': no_bias, 'std_var': std_fish_Gupta}, #std_fish_deb},
                             flab={'std': r'$\mathbf{\hat F}$', 'std_var': r'$\mathbf{\hat F}$'},
                             verbose=param.verbose)
                             #flab={'std': r'$\hat{\bm{\mathrm{F}}}$', 'std_var': r'$\hat{\bm{\mathrm{F}}}$'},
    fit_MCMC_T2 = read_fit(param.par_name, param.MCMC_T2, 'MCMC $T^2$',
                           fct={'std': par_fish_SH}, flab={'std': r'$\mathbf{\hat F}_{T^2}$'}, verbose=param.verbose)
                           #fct={'std': par_fish_SH}, flab={'std': r'$\hat{\bm{\mathrm{F}}}_{T^2}$'}, verbose=param.verbose)

    mode = -1
    delta = 200
    dpar_exact, det = Fisher_error_ana(np.zeros(param.n_D), param.sig2, delta, mode=mode)
    print('input par and normal std:          ', end='')
    for i, p in enumerate(param.par):
        print('{:.4f}  +- {:.5f}     '.format(p, dpar_exact[i], mode), end='')
    print('')

    fits = [fit_ABC, fit_MCMC_norm, fit_MCMC_T2]

    xlog = True
    ylog = False

    dy = [0.03, 1]
    plot_box(fits, param.n_D, param.par, dy, 'mean', boxwidth=param.boxwidth, xlog=xlog,
              ylog=False, model=param.model, verbose=param.verbose)

    dy = [0.006, 0.2]
    plot_box(fits, param.n_D, dpar_exact, dy, 'std', boxwidth=param.boxwidth, xlog=xlog,
              ylog=ylog, model=param.model, verbose=param.verbose)

    sig_var_noise = param.sig_var_noise
    dpar2 = dpar_exact ** 2
    dy = [100, 100]
    plot_box(fits, param.n_D, dpar2, dy, 'std_var', boxwidth=param.boxwidth, xlog=xlog,
              ylog=True, sig_var_noise=sig_var_noise, model=param.model, verbose=param.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
