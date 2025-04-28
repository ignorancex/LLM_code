"""Explore the trajectories of particles that no exist outside of R200m"""
from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy.stats import (
    binned_statistic as binstat, binned_statistic_2d as binstat2d)
import seaborn as sns
from time import time
from tqdm import tqdm
import warnings

from plottery.plotutils import colorscale, savefig, update_rcParams
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
    rcol = 'ComovingMostBoundDistance'
    r200col = 'R200MeanComoving'
    args = parse_args()
    sim, subs = load_subhalos(args, -1)
    subs = Subhalos(
        subs, sim, -1, exclude_non_FoF=True, logMmin=9,
        logM200Mean_min=13, logMstar_min=9,
        verbose_when_loading=False)
    sb = (subs[rcol] > subs[r200col])
    ic(sb.size, sb.sum())
    # plot radial distribution of satellites - how far out do they reach?
    # Does this depend on cluster or satellite mass?
    # Are galaxies with r>r200 infallers or splashback?

    col = f'{rcol}/{r200col}'
    plot_radial_distribution(args, subs, col)


def plot_radial_distribution(args, subs, rcol):
    x = subs[rcol][subs.satellite_mask]
    #rbins = np.linspace(0, 5, 41)
    logrbins = np.linspace(-2, 1, 41)
    logrx = (logrbins[:-1]+logrbins[1:]) / 2
    rbins = 10**logrbins
    rx = 10**logrx
    # no need for pi if we show units consistently
    vol = (rbins[1:]**3 - rbins[:-1]**3)
    counts = np.histogram(x, rbins)[0]
    #tbins = [0, 1, 2, 4, 8, 12]
    tbins = np.hstack([np.arange(0, 2, 0.5), np.arange(2, 5, 1),
                       np.arange(5, 15, 3)])
    t = subs['history:first_infall:time'][subs.satellite_mask]
    counts_binned = binstat2d(
        x, t, None, 'count', (rbins,tbins)).statistic
    ic(counts.shape, counts_binned.shape)
    fig, ax = plt.subplots()
    ax.plot(rx, counts/vol, 'ko-')
    lines = ax.plot(rx, counts_binned/vol[:,None], '-', lw=1)
    #labels = [f'Bin {i}' for i in range(len(lines))]
    labels = [f'({tbins[i-1]:.1f},{tbins[i]:.1f})' if i < 5 else
              f'({tbins[i-1]:.0f},{tbins[i]:.0f})'
              for i in range(1, tbins.size)]
    ax.legend(lines, labels, fontsize=12, loc='upper right', ncol=2)
    ax.set(xscale='log', yscale='log', xlabel='$R/R_\mathrm{200m}$',
           ylabel='$n(R/R_\mathrm{200m})$ ($R_\mathrm{200}^{-3}$)',
           ylim=(0.01, 1e8))
    output = os.path.join(subs.sim.plot_path, 'radialdist', 'radialdist.png')
    savefig(output, fig=fig)
    return


def plot_r_vs_t(args, subs):
    # make a 2d histogram of R/R200m vs time-since-infall,
    # particularly for R>R200m
    return


main()
