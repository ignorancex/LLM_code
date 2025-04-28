import cmasher as cmr
from icecream import ic
from itertools import count
from matplotlib import cm, colors, pyplot as plt, ticker
import multiprocessing as mp
import numpy as np
import os
from scipy.stats import (binned_statistic_2d as binstat2d, gaussian_kde)
import seaborn as sns
from time import time
from tqdm import tqdm

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.helpers.plot_auxiliaries import (
    format_filename, get_axlabel)
from hbtpy.helpers.plot_definitions import xbins
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos
from hbtpy.track import Track

from plottery.plotutils import savefig, update_rcParams
update_rcParams()

try:
    sns.set_color_palette('flare')
except AttributeError:
    pass


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation, args.root)
    reader = HBTReader(sim.path)

    isnap = -1
    kwargs = dict(logMmin=0, logM200Mean_min=13, logMstar_min=9)
    subs = Subhalos(reader.LoadSubhalos(isnap), sim, isnap, **kwargs)
    # subs.catalog['Depth'] = np.min(
    #     [subs.catalog['Depth'], 3*np.ones(subs.catalog['Depth'].size)], axis=0)
    #subs.catalog['Depth'][subs['Depth'] > 3] = 3

    ic(np.unique(subs['Depth']), np.unique(subs['history:first_infall:Depth']))
    db = np.unique(subs['history:first_infall:Depth'])
    print('unique history:first_infall:Depths', db)
    db = db[np.isfinite(db)]
    db = np.append(db-0.5, db[-1:]+0.5)
    for d in np.unique(subs['Depth']):
        jd = (subs['Depth'] == d)
        print(f'of the {jd.sum():.0f} that have Depth={d:.0f}:')
        x = subs['history:first_infall:Depth'][jd]
        for i, c in zip(*np.unique(x, return_counts=True)):
            print(f'  {c:4d} ({c/jd.sum():.2f}) fell in with Depth={i}')
    print()
    db = np.unique(subs['Depth'])
    print('unique Depths', db)
    db = np.append(db-0.5, db[-1:]+0.5)
    print()
    for d in np.unique(subs['history:first_infall:Depth']):
        jd = (subs['history:first_infall:Depth'] == d)
        print(f'of the {jd.sum():.0f} that fell in with Depth={d:0f}:')
        x = subs['Depth'][jd]
        for i, c in zip(*np.unique(x, return_counts=True)):
            print(f'  {c:4d} ({c/jd.sum():.2f}) have Depth={i}')
    print()

    dbins = np.append(np.arange(-0.5, 3.5), np.array([100]))
    fig, ax = plt.subplots()
    d = (dbins[:-1]+dbins[1:]) / 2
    d[-1] = d[-2] + 1
    ic(dbins, d)
    h2d = np.histogram2d(
        subs['history:first_infall:Depth'], subs['Depth'], dbins)[0]
    h2d[h2d == 0] = np.nan
    im = ax.imshow(h2d.T/np.nansum(h2d), origin='lower',
                   extent=(dbins[0],dbins[-2]+1,dbins[0],dbins[-2]+1))
    plt.colorbar(im, ax=ax)
    for i, d_i in enumerate(d):
        for j, d_j in enumerate(d):
            ax.annotate(
                f'{h2d[i,j]:.0f}', xy=(d_i,d_j), fontsize=14,
                ha='center', va='center', color='C1', fontweight='bold')
    ax.set(xlabel='history:first_infall:Depth', ylabel='Depth')
    output = os.path.join(sim.plot_path, 'depth', 'depth_today_infall.png')
    savefig(output)
    return

    cols = ['TrackId', 'Depth', 'history:first_infall:time',
            'history:cent:time', 'Mstar/history:first_infall:Mstar',
            'Mbound/history:first_infall:Mbound', 'Mbound/Mstar']

    # ti = time()
    # depth_history(args, subs, sim, n=0)
    # print(f'depth_history in {time()-ti:.1f} s')

    #plot_snaps_till_sat(args, subs)

    #return

    # ti = time()
    # plot_depth_history(args, reader, sim)
    # print(f'plot_depth_history in {time()-ti:.1f} s')

    xcol = 'history:first_infall:time'
    ycol = 'history:cent:time-history:first_infall:time'
    plot_fractions(subs, cols, xcol, ycol)
    ycol = ycol.split('-')[0]
    plot_fractions(subs, cols, xcol, ycol)
    return

    #pairplot_scipy(subs, cols)
    pairplot_sns(subs, cols)

    return


def depth_history(args, subs, sim, n=100):
    rdm = np.random.default_rng(seed=1)
    sample = subs['TrackId']
    if n == 0 or n > trackids:
        print(f'Using all {sample.size} subhaloes')
        trackids = sample
    else:
        print(f'Choosing {n} subhaloes among {sample.size}')
        trackids = rdm.choice(sample, size=n, replace=False)
    if args.ncores == 1:
        depth_history = [_track_depth_history(trackid, sim)
                         for trackid in tqdm(trackids)]
    else:
        pool = mp.Pool(args.ncores)
        out = [pool.apply_async(_track_depth_history, args=(trackid,sim))
               for trackid in trackids]
        pool.close()
        pool.join()
        depth_history = [i.get() for i in out]
    tracks = np.array([i[0] for i in depth_history])
    depth_history = np.array([i[1] for i in depth_history])
    ic(depth_history.shape)
    # ic(depth_history[:,0])
    output = os.path.join(sim.data_path, 'depth_history')
    np.savetxt(f'{output}.txt',
               np.concatenate((tracks[:,None], depth_history), axis=1),
               fmt='%d')
    print(f'Saved to {output}')
    return
        


def plot_fractions(subs, cols, xcol, ycol):
    plot_fractions_1d(subs, cols, f'{ycol}-{xcol}')
    plot_fractions_2d(subs, cols, xcol, ycol)
    return


def plot_fractions_1d(subs, cols, xcol):
    tbins = np.arange(0, 13.6)
    return


def plot_fractions_2d(subs, cols, xcol, ycol):
    tbins = np.arange(0, 13.6, 0.5)
    #dbins = np.arange(-13.5, 13.6, 0.5)
    dbins = np.arange(-1, 1.01, 0.1) - 0.05
    bins = [dbins if '-' in col else tbins for col in (ycol,xcol)]
    extent = [bins[1][0], bins[1][-1], bins[0][0], bins[0][-1]]
    s = subs.satellite_mask
    h = np.histogram2d(subs[ycol][s], subs[xcol][s], bins)[0]
    h1 = binstat2d(
            subs[ycol], subs[xcol], (subs['Depth'] == 1).astype(int),
            statistic='sum', bins=bins).statistic
    h2 = binstat2d(
            subs[ycol], subs[xcol], (subs['Depth'] >= 2).astype(int),
            statistic='sum', bins=bins).statistic
    h3 = binstat2d(
            subs[ycol], subs[xcol], (subs['Depth'] >= 3).astype(int),
            statistic='sum', bins=bins).statistic
    imkwargs = dict(
        origin='lower', cmap='magma', vmin=0, vmax=1, extent=extent,
        aspect='auto')
    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[10,1])
    axes = [fig.add_subplot(grid[0,i]) for i in range(3)]
    caxes = [fig.add_subplot(grid[1,0]), fig.add_subplot(grid[1,1:])]
    im = axes[0].imshow(
        h, origin='lower', cmap='cmr.arctic_r', extent=extent, aspect='auto')
    axes[0].set_title('Depth = 1', fontsize=15)
    plt.colorbar(im, cax=caxes[0], orientation='horizontal', label='N')
    im = axes[1].imshow(h2/h, **imkwargs)
    axes[1].set_title('Depth $\geq$ 2', fontsize=15)
    axes[2].imshow(h3/h, **imkwargs)
    axes[2].set_title('Depth $\geq$ 3', fontsize=15)
    axes[0].set_ylabel(get_axlabel(ycol))
    for ax in axes:
        ax.axhline(0, ls='--', color='C3', lw=2)
        ax.set_xlabel(get_axlabel(xcol))
    plt.colorbar(im, cax=caxes[1], orientation='horizontal', label='Fraction')
    #fig.tight_layout()
    path = os.path.join(subs.sim.plot_path, 'depth')
    os.makedirs(path, exist_ok=True)
    output = format_filename(f'depth_fraction__{xcol}__{ycol}.pdf')
    savefig(os.path.join(path, output), fig=fig, tight=False)
    return


def _track_depth_history(trackid, sim, fill=True):
    ic(trackid)
    track = Track(trackid, sim)
    depth = track['Depth'].values
    if fill and track.length < sim.snapshots.size:
        return trackid, np.pad(depth, (sim.snapshots.size-track.length, 0),
                               'constant', constant_values=(-1, 0))
    return trackid, depth


def plot_depth_history(args, reader, sim, z=0, nsub=20, seed=1):
    # if I choose z>0 then maybe I should be able to adjust
    # logM200Mean_min
    isnap = sim.snapshot_index_from_redshift(z)
    ic(isnap)
    subs = Subhalos(
        reader.LoadSubhalos(isnap), sim, isnap, logMmin=0, logMstar_min=9,
        logM200Mean_min=13)
    subs.sort('Mbound')
    rng = np.random.default_rng(seed=seed)
    idx = rng.choice(subs['TrackId'][subs.satellite_mask], size=nsub)
    early_sat = (subs['history:sat:time'] > 12)
    ic(early_sat.shape, early_sat.sum())
    idx = np.append(
        rng.choice(subs['TrackId'][subs.satellite_mask & ~early_sat],
                   size=nsub//2),
        rng.choice(subs['TrackId'][subs.satellite_mask & early_sat],
                   size=nsub//2))
    ic(idx, idx.size)
    #idx = idx[np.argsort(subs['Mbound'].iloc[idx].to_numpy())]
    ic(idx)
    show_grad_axes = False
    n0 = int(show_grad_axes)
    nrows, ncols = 4, 5
    kwargs = dict(figsize=(3.5*ncols,(3.5+n0)*nrows), constrained_layout=True)
    if show_grad_axes:
        kwargs['height_ratios'] = nrows * [1, 0.25]
    fig, axes = plt.subplots((1+n0)*nrows, ncols, **kwargs)
    if show_grad_axes:
        grad_axes = np.reshape(axes[1::2], -1)
        axes = np.reshape(axes[::2], -1)
    else:
        axes = np.reshape(axes, -1)
        # dummy
        grad_axes = axes
    if args.ncores > 1:
        pool = mp.Pool(args.ncores)
        _ = [pool.apply_async(
                plot_depth_track,
                args=(subs,trackid,ax,gax,i,nsub,ncols,nrows,show_grad_axes))
             for i, (ax, gax, trackid) in enumerate(zip(axes, grad_axes, idx))]
        pool.close()
        pool.join()
    else:
        for i, (ax, gax, trackid) in tqdm(
                enumerate(zip(axes, grad_axes, idx)), total=nsub):
            plot_depth_track(
                subs, trackid, ax, gax, i, nsub, ncols, nrows, show_grad_axes)
    output = os.path.join(sim.plot_path, 'depth', 'depth_history_mixsat.pdf')
    savefig(output, fig=fig, tight=False)
    return


def plot_depth_track(subs, trackid, ax, gax, i, nsub, ncols, nrows, show_grad_axes):
    track = Track(trackid, subs.sim)
    ic(track, track.birth_snapshot)
    sub = (subs['TrackId'] == trackid).to_numpy()
    if i == 0:
        ic(np.sort(track.colnames))
        ic(sub, type(sub))
    depth_history = track['Depth']
    life = np.s_[track.birth_snapshot[0]:] if track.birth_snapshot[0] > 0 \
        else np.ones(depth_history.size, dtype=bool)
    t = subs.sim.t_lookback[life]
    ic(trackid, depth_history.shape, track.birth_snapshot)
    ax.plot(t, depth_history, '-', color='k')
    ic(subs['history:first_infall:time'][sub])
    tinf = subs['history:first_infall:time'][sub].values[0]
    tacc = subs['history:last_infall:time'][sub].values[0]
    ic(tinf, type(tinf))
    ax.plot(tinf, 0.5, '|', color='C0', ms=20, mew=3)
    ax.plot(tacc, 0.8, '|', color='C3', ms=20, mew=3)
    ax.plot(subs['history:birth:time'][sub].values,
            subs['history:birth:Depth'][sub].values,
            'x', color='0.4', ms=12, mew=3)
    ax.set_title(f'Track {trackid}', fontsize=14)
    ax.axhline(0.5, ls='--', color='0.5')
    yrng = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.set(xlim=(-0.5, 14), ylim=(-0.15, 5.15), yticks=range(5))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    if i >= nsub - ncols:
        gax.set(xlabel='$t_\mathrm{lookback}$ (Gya)')
    # mass history
    rax = ax.twinx()
    rax.plot(t, np.log10(track['Mstar']), 'C1-', lw=3)
    rax.plot(t, np.log10(track['Mbound']), 'C2-', lw=3)
    if show_grad_axes:
        gax.plot(t, np.gradient(np.log10(track['Mstar'])), 'C1-', lw=3)
        gax.plot(t, np.gradient(np.log10(track['Mbound'])), 'C2-', lw=3)
        gax.axhline(0, ls='--', color='k', lw=1)
        gax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    if i % ncols == 0:
        ax.set(ylabel='Depth')
    if i % ncols == ncols - 1:# and i % nrows == 1:
        rax.set_ylabel('log $\{m_\mathrm{sub},m_{\u2605}\}/$M$_\odot$')
    return


def plot_snaps_till_sat(args, subs):
    ic(subs.sim.t_lookback[1:] - subs.sim.t_lookback[:-1])
    ic(np.unique(subs['history:birth:Depth'], return_counts=True))
    t_till_sat = subs['history:birth:time'] - subs['history:sat:time']
    snaps_till_sat = subs['history:sat:isnap'] - subs['history:birth:isnap']
    ibins = np.arange(22) - 0.5
    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(snaps_till_sat, ibins, color='C0', lw=3, histtype='step')
    ax.set(xlabel='Snapshots from birth to sat', ylabel='Differential')
    rax = ax.twinx()
    rax.hist(snaps_till_sat, ibins, color='C1', lw=3, histtype='step',
             cumulative=True)
    rax.set(ylabel='Cumulative')
    output = os.path.join(subs.sim.plot_path, 'tsat', 'snaps_till_sat.png')
    savefig(output, fig=fig, tight=False)
    return


def plot_times(subs, cols):
    xcol = 'history:cent:time'
    ycol = 'history:first_infall:time'
    fig, ax = plt.subplots()
    sns.kdeplot(
        subs.catalog[subs['Depth'] > 0], x=xcol, y=ycol, hue='Depth', ax=ax,
        bw_method=0.2, legend=False,
        palette=cmr.get_sub_cmap('magma', 0.3, 0.8), levels=[0.1,0.5,0.8])
    plt.legend(loc='upper left', fontsize=14)
    path = os.path.join(
        subs.sim.plot_path, 'depth')
    os.makedirs(path, exist_ok=True)
    output = format_filename(f'depth__{xcol}__{ycol}.pdf')
    savefig(os.path.join(path, output), fig=fig)
    return


def pairplot_scipy(subs, cols):
    nc = len(cols) - 1
    samples = [subs['Depth'] == 1, subs['Depth'] == 2, subs['Depth'] >= 3]
    ic([s.sum() for s in samples])
    fig, axes = plt.subplots(
        nc, nc, figsize=(2*nc,2*nc), constrained_layout=True)
    for i, xcol in enumerate(cols[1:]):
        for j, ycol in enumerate(cols[1:]):
            ax = axes[i,j]
            if j == i:
                plot_diagonal(ax, subs, samples, xcol)
            # lower triangle
            elif j < i:
                #plot_offdiag(ax, subs, xcol, ycol)
                ...
    return


def plot_diagonal(ax, subs, samples, col):
    bins = xbins[col]
    x = (bins[1:]+bins[:-1]) / 2
    for i, s in enumerate(samples):
        color = f'C{i}'
        kde = gaussian_kde(subs[col][samples])
        h = kde(x)
        ax.fill_between(x, h, color=color)
    return


def pairplot_sns(subs, cols):
    palette = cmr.get_sub_cmap('cmr.bubblegum', 0.2, 0.8)
    palette = sns.color_palette('flare', 3)
    pair_grid = sns.pairplot(
        subs[cols][subs.satellite_mask], hue='Depth', kind='kde', corner=True,
        plot_kws=dict(levels=[0.1, 0.5]),
        grid_kws=dict(despine=False),
        palette=palette)
    #pair_grid._legend.remove()
    pair_grid._legend.set(label=('1', '2', '$\geq3$'))
    fig = pair_grid.figure
    #pair_grid.map_diag(sns.kdeplot)
    #pair_grid.map_upper(sns.kdeplot, levels=[0.25,0.75])
    pair_grid.map_lower(sns.kdeplot, levels=[0.5,0.9], legend=False)
    for ax in fig.axes:
        
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if 'Mbound' in xlabel:# or 'Mstar' in xlabel:
            ax.set_xscale('log')
        # do not make diagonals log scale in y
        if xlabel != ylabel and ('Mbound' in ylabel):# or 'Mstar' in ylabel):
            ax.set_yscale('log')
        # custom limits
        # tcent
        # if xlabel == cols[2]: ax.set_xlim((0, 10))
        # if ylabel == cols[2]: ax.set_ylim((0, 10))
        # Mstar/Mstar_infall
        if xlabel == cols[3]: ax.set_xlim((0, 5))
        if ylabel == cols[3]: ax.set_ylim((0, 5))
        # Msub/Msub_infall
        if xlabel == cols[4]: ax.set_xlim((0.003, 2))
        if ylabel == cols[4]: ax.set_ylim((0.003, 2))
        # Msub/Mstar
        if xlabel == cols[5]: ax.set_xlim((1, 200))
        if ylabel == cols[5]: ax.set_ylim((1, 200))
        xlabel = get_axlabel(xlabel).replace('$$', '$')
        ylabel = get_axlabel(ylabel).replace('$$', '$')
        ax.set(xlabel=xlabel, ylabel=ylabel)
        for spine in ('bottom', 'right', 'top', 'left'):
            ax.spines[spine].set_color('k')
            ax.spines[spine].set_linewidth(2)
    fig.legend(labels=('Depth $\geq3$', 'Depth = 2', 'Depth = 1'),
               loc='upper center')
    savefig(os.path.join(subs.sim.plot_path, 'pairplots', 'depth.png'), fig=fig)

    return


main()