from icecream import ic
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm, Normalize
import multiprocessing as mp
import numpy as np
import os
from scipy.stats import gaussian_kde
import seaborn as sns
from time import time
from tqdm import tqdm
import warnings

from plottery.plotutils import colorscale, savefig, update_rcParams
from plottery.statsplots import contour_levels
update_rcParams()

from hbtpy.hbt_tools import load_subhalos, parse_args, save_plot
from hbtpy.helpers.plot_auxiliaries import (
    binning, definitions, format_filename, get_axlabel, get_bins,
    get_label_bincol, logcenters, massbins, plot_line)
from hbtpy.helpers.plot_definitions import (
    ccolor, scolor, massnames, units, xbins, binlabel, events, axlabel, ylims)
from hbtpy.subhalo import Subhalos
from hbtpy.track import Track


warnings.simplefilter('ignore', RuntimeWarning)


def main():
    args = parse_args()
    sim, subs = load_subhalos(args, -1)
    subs = Subhalos(
        subs, sim, -1, exclude_non_FoF=True, logMmin=9,
        logM200Mean_min=13, logMstar_min=9,
        verbose_when_loading=False)
    # first a galaxy whose stellar mass has grown since infall
    sat = subs.satellite_mask
    grown = (subs['history:max_Mstar:z'] < subs['history:first_infall:z']) \
        & (subs['Rank'] > 0) & (subs['Mstar'] > 1e10)
    ic(grown.size, grown.sum())

    rdm = np.random.default_rng(seed=19)
    test = rdm.choice(np.arange(grown.size)[~grown])

    plot_track_orbit(subs, test)
    #plot_mratio(subs, sat)
    return


def plot_track_orbit(subs, trackid, ccol='Mstar', scol='Mbound', clog=True,
                     slog=True, vmin=None, vmax=None, preview=True):
    mass_to_s = lambda m: (m / 5)**2
    track = Track(subs['TrackId'][trackid], subs.sim)
    ic(np.sort(track.colnames))
    xyzcols = [f'ComovingMostBoundPosition{i}' for i in '012']
    xyz = track[xyzcols].to_numpy().T
    host = Track(track.central(-1), subs.sim)
    xyzhost = host[xyzcols].to_numpy().T
    fig, ax = plt.subplots(figsize=(8,6))
    c = track[ccol]
    s = track[scol]
    if slog:
        s = np.log10(s)
    norm = LogNorm(vmin=vmin, vmax=vmax) if clog \
        else Normalize(vmin=vmin, vmax=vmax)
    cm = ax.scatter(
        xyz[0]-xyzhost[0], xyz[1]-xyzhost[1], c=c, s=s, norm=norm)
    plt.colorbar(cm, ax=ax, label=get_label_bincol('Mstar'))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # size illustrations
    x0 = 3 * [xlim[0] + 0.05*(xlim[1]-xlim[0])]
    y0 = ylim[0] + 0.05*(ylim[1]-ylim[0]) * np.arange(1, 4)
    # this is good only for Mbound!
    s0 = [10, 11, 12]
    ax.scatter(x0, y0, c='k', s=s0)
    for (xi, yi, si) in zip(x0, y0, s0):
        ax.text(xi+0.03*(xlim[1]-xlim[0]), yi, str(si),
                ha='left', va='center', fontsize=18)
    ax.set(xlabel='x (Mpc)', ylabel='y (Mpc)')
    ccol = ccol.replace(':', '-').replace('/', '-over-')
    scol = scol.replace(':', '-').replace('/', '-over-')
    output = f'orbit/orbit__{trackid}__{ccol}__{scol}.pdf'
    save_plot(fig, output, subs.sim)

    return


def plot_mratio(subs, sat):
    # stellar mass change since infall
    logxkde = np.linspace(-0.8, 1.7, 1000)
    xkde = 10**logxkde
    x = np.log10(subs['Mstar/history:first_infall:Mstar'][sat])
    _, logbins, = np.histogram(x, 'auto')
    bins = 10**logbins
    xx = subs['history:first_infall:Mstar'][sat]
    xxx = subs['Mstar'][sat]
    ic(xxx.min(), xxx.max(), (xxx==0).sum(), np.isnan(xxx).sum())
    ic(xx.min(), xx.max(), (xx==0).sum(), np.isnan(xx).sum())
    ic(x.min(), x.max(), np.isnan(x).sum())
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    kde = gaussian_kde(x, 0.5)
    ax.plot(xkde, kde(logxkde), 'k-', )
    #ax.hist(10**x, logbins, color='C9', histtype='stepfilled', log=True)
    #ax.hist(x, 'auto', color='C9', histtype='stepfilled', log=True)
    # binned by infall mass
    bincol = 'history:first_infall:Mbound'
    minf_bins = np.logspace(10.5, 12.5, 5)
    log_minfbins = np.log10(minf_bins)
    binlabel = get_label_bincol(bincol)
    ic(binlabel)
    for i in range(1, minf_bins.size):
        binmask = (subs[bincol] > minf_bins[i-1]) & (subs[bincol] <= minf_bins[i])
        ic(x[binmask].shape)
        label = f'({log_minfbins[i-1]:.1f},{log_minfbins[i]:.1f}]'
        kde = gaussian_kde(x[binmask], 0.5)
        ax.plot(xkde, kde(logxkde), f'C{i}-', label=label)
        #ax.hist(x[binmask], 'auto', histtype='step', label=f'Bin {i}') 
        # sns.kdeplot(subs['Mstar/history:first_infall:Mstar'], ax=ax,
        #             color=f'C{i}', label=label, log_scale=10)
    ax.axvline(0, ls='--', color='k')
    ax.legend(fontsize=14)
    ax.set(xscale='log', yscale='log', xlabel='Mstar/Mstar_infall', ylim=(1e-2, 10))
    output = 'hist_Mstar_change'
    save_plot(fig, output, sim)
    return


if __name__ == '__main__':
    main()
