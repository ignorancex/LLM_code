#from astropy.io import ascii
#from astropy.table import Table
from icecream import ic
from matplotlib import pyplot as plt, ticker
import numpy as np
import os
from scipy.optimize import curve_fit
import sys
from time import time

from plottery.plotutils import colorscale, savefig, update_rcParams
update_rcParams()

from hbtpy.hbt_tools import save_plot
from .plot_auxiliaries import (
    binning, definitions, get_axlabel, get_bins, get_label_bincol, logcenters,
    massbins, plot_line)
from .plot_definitions import (
    ccolor, scolor, massnames, units, xbins, binlabel, axlabel)


def plot_massfunction(
        sim, subs, binparam, bins=None, binlabel='', bin_in_log10=True,
        hostmass='M200Mean', overwrite=True, norm=False, show_centrals=False,
        mbins=np.logspace(8,15,21), mubins=np.logspace(-5, 0, 21),
        show_all=False, show_slope=False):
    """Plot subhalo mass function, both as a function of msub and mu

    TO-DO:
        * I still need to store the data in tables
        * Allow calculation of ratios of columns e.g., by passing two
          names in ``binparam``, perhaps with another kwarg whose
          value is the operation (e.g., 'div', 'sum', 'mean', etc)
    """
    print('Plotting mass function with {0}'.format(binparam), end=' ')
    if bin_in_log10:
        print('(log10)')
    else:
        print()
    # we will make one plot each
    names = ('msub', 'mu')
    suff = 'log{0}'.format(binparam) if bin_in_log10 else binparam
    if norm:
        suff += '_norm'
    outputs = ['n{0}_{1}'.format(name, suff) for name in names]
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        definitions(subs, hostmass=hostmass)
    gcen = cen & ~dark
    gsat = sat & ~dark
    fig1, ax1 = plt.subplots(figsize=(8,6))
    fig2, ax2 = plt.subplots(figsize=(8,6))
    figs = [fig1, fig2]
    axes = [ax1, ax2]
    for ax in axes:
        ax.set_yscale('log')
    ax1.set_xlabel(r'log $m_\mathrm{{sub}}/{0}$'.format(Msun))
    # ylabel = r'$N(m_\mathrm{sub})$'
    # if norm:
    #     ylabel = ylabel[:-1] + r'/m_\mathrm{sub}$'
    ylabel = r'$n(m_\mathrm{sub})$' if norm else r'$N(\mathrm{sub})$'
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel(
        r'log $\mu$ $\equiv$ log $(m_\mathrm{sub}/M_\mathrm{200m})$')
    nlabel = r'$n(\mu)$' if norm else r'$N(\mu)$'
    ax2.set_ylabel(nlabel)
    # central bin values for plots (msub and mu; it is assumed that
    # they are binned in log space)
    logm = np.log10(logcenters(mbins))
    logmu = np.log10(logcenters(mubins))
    binvals = subs.catalog[binparam]
    hfig, hax = plt.subplots()
    hquants = [binvals, subs.centrals[binparam], subs.satellites[binparam]]
    if bin_in_log10:
        hquants = [np.log10(i) for i in hquants]
    #hbins = np.linspace(0, 100, 201)
    hbins = 100
    if not bin_in_log10:
        hax.hist(hquants[0], hbins, log=True, label='all')
        hax.hist(hquants[1], hbins, log=True, histtype='step',
                 lw=3, label='centrals')
    hax.hist(hquants[2], hbins, log=True, histtype='step',
             lw=3, label='satellites')
    hax.legend()
    hax.set_xlabel(binlabel)
    save_plot(hfig, 'hist/{0}{1}'.format(binparam, '_log'*bin_in_log10), sim)
    print('binvals =', binvals[sat].min(), binvals[sat].max(),
        np.percentile(binvals[sat], [1,25,50,75,99]))
    print('bins =', bins)
    h = np.histogram(binvals[sat], bins)[0]
    print('hist =', h, h.sum(), sat.sum())
    print('R>60:')
    cols = ['TrackId','HostHaloId','Rank','Nbound',binparam]
    if 'Distance' in binparam or 'Velocity' in binparam:
        dcols = [binparam+i for i in '012']
        cols = cols + dcols
    #
    xbins = np.log10(logcenters(bins)) if bin_in_log10 \
        else (bins[:-1]+bins[1:]) / 2
    # normalization (not used right now)
    nij = 1
    fn = lambda x, a, b, x0=10: a + b*(x-x0)
    fitmasks = ((logm >= 9) & (logm <= 12),
                (logmu >= -4) & (logmu <= -1))
    for fig, ax, x, b, xb, mask, name, output, legloc \
            in zip(figs, axes, (mtot,mtot/mhost), (mbins,mubins),
                   (logm,logmu), fitmasks, names,
                   outputs, ('upper right','upper right')):
        logxo = np.log10(logcenters(b))
        if show_centrals:
            # centrals
            nc = np.histogram(x[cen], b)[0]
            ngc = np.histogram(x[gcen], b)[0]
            n0 = nc.sum() if norm else 1
            ax.plot(logxo[nc > 0], nc[nc > 0]/n0, '--', color=ccolor, lw=4,
                    label='Central subhaloes')
            ax.plot(logxo[ngc > 0], ngc[ngc > 0]/n0, '-', color=ccolor, lw=4,
                    label='Central galaxies')
        # satellites
        ns = np.histogram(x[sat], b)[0]
        ngs = np.histogram(x[gsat], b)[0]
        nfit, ncov = curve_fit(fn, xb[mask], np.log10(ns[mask]), p0=(-2,-1))
        ic(nfit)
        ic(np.diag(ncov)**0.5)
        if show_all:
            n0 = ns.sum() if norm else 1
            ax.plot(logxo[ns > 0], ns[ns > 0]/n0, '--', color=scolor, lw=4,
                    zorder=100, label='All satellite subhaloes')
            ax.plot(logxo[ngs > 0], ngs[ngs > 0]/n0, '-', color=scolor, lw=4,
                    zorder=100, label='All satellite galaxies')
        else:
            ax.plot([], [], '--', color='k', lw=3,
                    label='Satellite subhaloes')
            ax.plot([], [], '-', color='k', lw=3,
                    label='Satellite galaxies')
        ns, xe, ye = np.histogram2d(binvals[sat], x[sat], (bins,b))
        ngs = np.histogram2d(binvals[gsat], x[gsat], (xe,ye))[0]
        if bin_in_log10:
            colors, cmap = colorscale(array=10**xbins, log=True)
        else:
            colors, cmap = colorscale(array=xbins, log=True)
        for i in range(ns.shape[0]):
            j = (ns[i] > 0)
            if j.sum() == 0:
                continue
            n0 = ns[i].sum() if norm else 1
            ax.plot(logxo[j], ns[i][j]/n0, '--', color=colors[i], lw=4,
                    zorder=20-i)
            ax.plot(logxo[j], ngs[i][j]/n0, '-', color=colors[i], lw=4,
                    zorder=20-i)
        cbar = plt.colorbar(cmap, ax=ax)
        cbar.set_label(binlabel)
        if bin_in_log10:
            cbar.ax.set_yscale('log')
            if binparam == 'ComovingMostBoundDistance':
                cbar.ax.yaxis.set_major_formatter(
                    ticker.FormatStrFormatter('%.1f'))
        ax.legend(loc=legloc, ncol=1, fontsize=14)
        # to give the legend some space
        # ylim = ax.get_ylim()
        # ax.set_ylim(ylim[0], 10*ylim[1])
        save_plot(fig, output, sim)
    return


def plot_massfunction_old(sim, subs, hostmass='M200Mean'):
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = \
        definitions(subs, hostmass=hostmass)
    gcen = cen & ~dark
    gsat = sat & ~dark
    # testing
    fig1, ax1 = plt.subplots(figsize=(9,6))
    fig2, ax2 = plt.subplots(figsize=(9,6))
    figs = [fig1, fig2]
    axes = [ax1, ax2]
    for ax in axes:
        ax.set_yscale('log')
    ax1.set_xlabel(r'log $m_\mathrm{{sub}}/{0}$'.format(Msun))
    ax1.set_ylabel(r'$N(m_\mathrm{sub})$')
    ax2.set_xlabel(
        r'log $\mu$ $\equiv$ log $(m_\mathrm{sub}/M_\mathrm{200m})$')
    ax2.set_ylabel(r'$N(\mu)$')
    mbins = np.logspace(8, 15, (15-8)//0.2)
    logm = np.log10(logcenters(mbins))
    mubins = np.logspace(-5, 0, 5//0.2)
    mhbins = np.logspace(10, 14.5, 9)
    logmh = np.log10(logcenters(mhbins))
    # normalization?
    nij = 1
    for ax, x, bins, name \
            in zip(axes, (mtot,mtot/mhost), (mbins,mubins), ('msub','mu')):
        xo = logcenters(bins)
        #logxo = np.log10(xo)
        logxo = logm
        # normalization of y axis. Just keeping the name from below
        #if name == 'mu':
            #nij = 1 / xo
        # all subhalos
        # centrals
        c = np.histogram(x[cen], bins)[0]
        gc = np.histogram(x[gcen], bins)[0]
        ax.plot(logxo[c > 0], (nij*c)[c > 0], '--', color=ccolor, lw=4,
                label='Central subhaloes')
        ax.plot(logxo[gc > 0], (nij*gc)[gc > 0], '-', color=ccolor, lw=4,
                label='Central galaxies')
        # satellites
        s = np.histogram(x[sat], bins)[0]
        gs = np.histogram(x[gsat], bins)[0]
        ax.plot(logxo[s > 0], (nij*s)[s > 0], '--', color=scolor, lw=4,
                label='Satellite subhaloes')
        ax.plot(logxo[gs > 0], (nij*gs)[gs > 0], '-', color=scolor, lw=4,
                label='Satellite galaxies')
        xcolname = 'log{0}'.format(name)
        nm = Table({xcolname: logxo, 'csub': c, 'cgal': gc,
                    'ssub': s, 'sgal': gs})
        nm[xcolname].format = '%.2f'
        ascii.write(nm, os.path.join(sim.data_path, 'n{0}.txt'.format(name)),
                    format='fixed_width', overwrite=True)
        # split by host mass
        s = np.histogram2d(mhost[sat], x[sat], (mhbins,bins))[0]
        gs = np.histogram2d(mhost[gsat], x[gsat], (mhbins,bins))[0]
        for sample, sname in zip((s, gs), ('subhaloes', 'galaxies')):
            ns = Table({'{0:.2f}'.format(mhi): si
                        for mhi, si in zip(logmh, sample)})
            ns[xcolname] = nm[xcolname]
            output = os.path.join(
                sim.data_path, 'n{0}_{1}.txt'.format(name, sname))
            ascii.write(ns, output, format='fixed_width', overwrite=True)
        colors, cmap = colorscale(array=logmh)
        for i in range(s.shape[0]):
            j = (s[i] > 0)
            if j.sum() == 0:
                continue
            #if name == 'mu':
                #nij = 1 / xo[j]
            ax.plot(logxo[j], nij*s[i][j], '--', color=colors[i], lw=2,
                    zorder=20-i)
            ax.plot(logxo[j], nij*gs[i][j], '-', color=colors[i], lw=2,
                    zorder=20-i)
        cbar = plt.colorbar(cmap, ax=ax)
        cbar.set_label(r'log $M_\mathrm{{host}}/{0}$'.format(Msun))
        ax.legend(loc='upper right', ncol=1, fontsize=16)
    for fig, name in zip(figs, ('msub','mu')):
        save_plot(fig, 'n{0}'.format(name), sim)
    return
