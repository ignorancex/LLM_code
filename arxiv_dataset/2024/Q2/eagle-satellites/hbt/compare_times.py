import cmasher as cmr
from icecream import ic
from matplotlib import cm, colors as mplcolors, pyplot as plt, ticker, rcParams
from matplotlib.colors import LogNorm
import multiprocessing as mp
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic, binned_statistic_2d, gaussian_kde, pearsonr
import seaborn as sns
from sklearn.neighbors import KernelDensity
from time import time
import warnings

from plottery.plotutils import savefig, update_rcParams

update_rcParams()
rcParams["text.latex.preamble"] += r",\usepackage{color}"

from HBTReader import HBTReader

# local
from hbtpy import hbt_tools
from hbtpy.helpers.plot_definitions import binlabel
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos

adjust_kwargs = dict(
    left=0.10, right=0.95, bottom=0.05, top=0.98, wspace=0.3, hspace=0.1
)

warnings.simplefilter("once", RuntimeWarning)
# warnings.simplefilter('ignore', PearsonRConstantInputWarning)


def main():
    print("Running...")
    args = (
        (
            "--demographics",
            {"action": "store_true"},
        ),
        ("--investigate", {"action": "store_true"}),
    )
    args = hbt_tools.parse_args(args=args)
    sim = Simulation(args.simulation, args.root)

    to = time()
    reader = HBTReader(sim.path)
    print(f"Loaded reader in {time()-to:.1f} seconds")
    to = time()
    subs = Subhalos(
        reader.LoadSubhalos(-1),
        sim,
        -1,
        as_dataframe=True,
        logMmin=None,
        logM200Mean_min=None,
        logMstar_min=9,
    )
    # subs.sort(order='Mbound')
    print(f"Loaded subhalos in {(time()-to)/60:.2f} minutes")
    ic(subs)
    ic(subs.central_mask.sum(), subs.satellite_mask.sum())

    print(
        "In total there are {0} central and {1} satellite subhalos".format(
            subs.centrals.shape[0], subs.satellites.shape[0]
        )
    )

    centrals = Subhalos(subs.centrals, sim, -1, load_any=False, logMstar_min=9)
    satellites = Subhalos(
        subs.satellites,
        sim,
        -1,
        load_any=False,
        logMmin=None,
        logM200Mean_min=13,
        logMstar_min=9,
    )
    print(np.median(satellites["history:max_Mstar:time"]))
    ic(satellites.size)

    ic(np.log10(satellites["M200Mean"].min()))
    ic((satellites["history:max_Mbound:z"] < satellites["history:cent:z"]).sum())
    ic(
        (
            satellites["history:max_Mbound:time"] > satellites["history:cent:time"] + 1
        ).sum()
    )
    ic(
        (
            satellites["history:max_Mbound:time"] < satellites["history:cent:time"] - 1
        ).sum()
    )
    ic(
        (satellites["history:max_Mstar:time"] < 0.1).sum(),
        (satellites["history:max_Mstar:time"] == 0).sum(),
    )
    # print(np.sort(satellites.colnames))
    # ic(satellites.catalog)
    # return

    review_subsamples(args, satellites)
    # return

    # some default configs
    kwargs = {
        "corr": dict(
            cmap="cmr.ember_r", cmap_rng=(0.35, 1), norm=LogNorm(vmin=0.2, vmax=0.9)
        ),
        "Mstar": dict(
            cmap="cmr.toxic_r", cmap_rng=(0.35, 1), norm=LogNorm(vmin=2e9, vmax=1e11)
        ),
        "Mbound": dict(
            cmap="cmr.cosmic", cmap_rng=(0.35, 1), norm=LogNorm(vmin=2e9, vmax=1e11)
        ),
        "Mbound/Mstar": dict(
            cmap="cmr.neon_r", cmap_rng=(0.1, 1), norm=LogNorm(vmin=2, vmax=150)
        ),
        "M200Mean": dict(
            cmap="inferno_r", cmap_rng=(0.3, 1), norm=LogNorm(vmin=1e13, vmax=2e14)
        ),
        # not setting vmin,vmax,log for this one
        "ratio": dict(cmap="cmr.rainforest_r", cmap_rng=(0.35, 1)),
        "ratio-over-max": dict(
            cmap="cmr.rainforest_r", cmap_rng=(0.35, 1), vmin=0, vmax=1
        ),
    }

    if args.ncores > 1:
        pool = mp.Pool()
        run = pool.apply_async
    else:
        run = lambda f: f
    run(
        plot_times(
            satellites,
            c_lower="corr",
            c_upper="Mbound/Mstar",
            kwargs_lower=kwargs["corr"],
            kwargs_upper=kwargs["Mbound/Mstar"],
        )
    )
    # run(plot_times(
    #     satellites, c_lower='history:max_Mstar:Mstar', c_upper='Mstar',
    #     kwargs_lower=kwargs['Mstar'], kwargs_upper=kwargs['Mstar']))
    # run(plot_times(
    #     satellites, c_lower='Mbound', c_upper='history:first_infall:Mbound',
    #     kwargs_lower=kwargs['Mbound'], kwargs_upper=kwargs['Mbound']))
    # run(plot_times(
    #     satellites, c_lower='Mbound', c_upper='history:max_Mbound:Mbound',
    #     kwargs_lower=kwargs['Mbound'], kwargs_upper=kwargs['Mbound']))
    # run(plot_times(
    #     satellites, c_lower='Mbound/history:first_infall:Mbound',
    #     c_upper='Mstar/history:first_infall:Mstar',
    #     kwargs_lower={**kwargs['ratio'], 'norm': LogNorm(vmin=0.1, vmax=2)},
    #     kwargs_upper={**kwargs['ratio'], 'norm': LogNorm(vmin=0.5, vmax=5)}))
    # run(plot_times(
    #     satellites, c_lower='Mbound/history:max_Mbound:Mbound',
    #     c_upper='Mstar/history:max_Mstar:Mstar',
    #     kwargs_lower=kwargs['ratio-over-max'],
    #     kwargs_upper=kwargs['ratio-over-max']))
    # run(plot_times(
    #     satellites, c_lower='history:first_infall:Mbound/history:max_Mbound:Mbound',
    #     c_upper='history:first_infall:Mstar/history:max_Mstar:Mstar',
    #     kwargs_lower=kwargs['ratio-over-max'],
    #     kwargs_upper={**kwargs['ratio'], 'vmin': 0.4, 'vmax': 1}))
    # run(plot_times(
    #     satellites, c_lower='history:first_infall:Mstar', c_upper='M200Mean',
    #     kwargs_lower=kwargs['Mstar'], kwargs_upper=kwargs['M200Mean']))
    # run(plot_times(
    #     satellites, c_lower='history:sat:Mstar', c_upper='Mstar',
    #     cmap_lower='cmr.toxic_r', cmap_upper='cmr.toxic_r',
    #     vmin_lower=9.5, vmax_lower=11.5, vmin_upper=9.3, vmax_upper=11))
    if args.ncores > 1:
        pool.close()
        pool.join()
    return


def review_subsamples(args, s):
    plot_path = os.path.join(s.sim.plot_path, "compare_times")
    tlb = s.sim.t_lookback.value
    print("\n*** Special subsamples ***")
    if not args.debug:
        ic.enable()
    print()
    nsat = s.size
    ic(nsat)
    review_tmstar(args, s, tlb, nsat, plot_path)
    review_tsat(args, s, tlb, nsat, plot_path)
    review_tacc(args, s, tlb, nsat, plot_path)
    if not args.debug:
        ic.disable()
    return


def review_tmstar(args, s, tlb, nsat, plot_path):
    print("** max mstar **")
    tmstar = s["history:max_Mstar:time"]
    tlate = tlb[-2]
    ic(s.sim.t_lookback[-20:])
    hmstar, hmstar_bins = np.histogram(tmstar, tlb[-30:][::-1])
    ic(hmstar_bins)
    ic(hmstar)
    ic(hmstar[0] / nsat)
    late_mstar = tmstar < tlate
    ic(late_mstar.sum())
    diffbins = np.arange(-13.5, 13.6, 0.5)
    ic(diffbins)
    for event in (
        "cent",
        "sat",
        "first_infall",
        "last_infall",
        "max_Mbound",
        "max_Mgas",
    ):
        h = f"history:{event}"
        ti = s[f"{h}:time"]
        ic(h, np.median(ti[late_mstar]), np.median(ti[~late_mstar]))
        ic((ti > tmstar).sum(), (ti > tmstar).sum() / nsat)
        ic(np.median(tmstar[late_mstar] - ti[late_mstar]))
        ic(np.median(tmstar[~late_mstar] - ti[~late_mstar]))
        # ic(binned_statistic(
        #     tmstar, ti-tmstar, np.nanmedian, diffbins)[0])
        # ic(binned_statistic(
        #     tmstar[late_mstar], (ti-tmstar)[late_mstar],
        #     np.nanmedian, diffbins)[0])
        # ic(binned_statistic(
        #     tmstar[~late_mstar], (ti-tmstar)[~late_mstar],
        #     np.nanmedian, diffbins)[0])
    for m in ("Mbound", "Mstar", "M200Mean"):
        ic(
            m,
            np.log10(np.median(s[m][late_mstar])),
            np.log10(np.median(s[m][~late_mstar])),
            np.median(s[m][late_mstar]) / np.median(s[m][~late_mstar]),
        )
    fit, cov = curve_fit(
        lambda x, a, b: a + b * x,
        s["history:max_Mbound:time"][~late_mstar],
        s["history:max_Mstar:time"][~late_mstar],
        p0=(5.1, 0.3),
    )
    ic(fit, np.diag(cov) ** 0.5)
    return


def review_tsat(args, s, tlb, nsat, plot_path):
    print("** t_sat **")
    tsat = s["history:sat:time"]
    ic(tlb[:20])
    htsat, htsat_bins = np.histogram(tsat, tlb[::-1])
    # fig, ax = plt.subplots(constrained_layout=True)
    # ax.plot(htsat_bins[1:], np.cumsum(htsat)/nsat)
    # ax.set(xlabel='$t_\mathrm{sat}^\mathrm{lookback}$ (Gyr)',
    #        ylabel='$N(<t_\mathrm{sat}^\mathrm{lookback})$')
    # output = os.path.join(plot_path, 'tsat_cdf.png')
    # savefig(output, fig=fig, tight=False)
    # ic(htsat_bins)
    # ic(htsat)
    # ic(np.cumsum(htsat[::-1])/nsat)
    early_tsat = tsat > 12
    nearly = early_tsat.sum()
    ic(nearly, nearly / nsat)
    # fig, ax = plt.subplots(figsize=(5,4))
    # ax.hist(s['history'])
    for event in (
        "cent",
        "sat",
        "first_infall",
        "last_infall",
        "max_Mbound",
        "max_Mgas",
        "max_Mstar",
    ):
        h = f"history:{event}"
        ic(
            h,
            np.median(s[f"{h}:time"][early_tsat]),
            np.median(s[f"{h}:time"][~early_tsat]),
        )
    for m in ("Mbound", "Mstar", "M200Mean"):
        ic(
            m,
            np.log10(np.median(s[m][early_tsat])),
            np.log10(np.median(s[m][~early_tsat])),
        )


def review_tacc(args, s, tlb, nsat, plot_path):
    print("** t_acc **")
    tacc = s["history:last_infall:time"]
    htacc, htacc_bins = np.histogram(tacc, tlb[-30:][::-1])
    ic(htacc_bins)
    ic(htacc)
    j = np.argmax(htacc)
    j1 = np.s_[j + 1 : j + 2]
    j2 = np.s_[j + 2 : j + 3]
    spike1 = (tacc >= htacc_bins[j1][0]) & (tacc <= htacc_bins[j1][-1])
    ic(htacc_bins[j1], htacc[j1])
    ic(spike1.sum(), spike1.sum() / nsat)
    ic(np.unique(s["HostHaloId"][spike1], return_counts=True))
    ic((s["HostHaloId"] == 1).sum())
    spike2 = (tacc >= htacc_bins[j2][0]) & (tacc <= htacc_bins[j2][-1])
    ic(htacc_bins[j2], htacc[j2])
    ic(spike2.sum(), spike2.sum() / nsat)
    ic(np.unique(s["HostHaloId"][spike2], return_counts=True))
    ic((s["HostHaloId"] == 5).sum())
    return


def plot_times(
    satellites,
    c_lower="corr",
    c_upper="Mstar",
    use_lookback=True,
    stat_lower=np.nanmean,
    stat_upper=np.nanmean,
    kwargs_lower={},
    kwargs_upper={},
):
    """
    To do:
        - Implement option whether colorbar should be log-normed
    """
    ic(kwargs_lower["cmap"])
    kwargs = [kwargs_lower, kwargs_upper]
    for i, kw in enumerate(kwargs):
        if "cmap_rng" in kw:
            cmap = kw.get("cmap", "viridis")
            cmap = cmr.get_sub_cmap(cmap, kw.get("vmin", 0), kw.get("vmax", 1))
            kwargs[i]["cmap"] = cmap
    kwargs_lower, kwargs_upper = kwargs
    ic(kwargs_lower["cmap"])
    # events = ('birth', 'cent', 'sat', 'first_infall', 'last_infall', 'max_Mdm',
    #           'max_Mstar', 'max_Mgas')
    events = (
        "cent",
        "sat",
        "first_infall",
        "last_infall",
        "max_Mdm",
        "max_Mstar",
        "max_Mgas",
    )
    nc = len(events)
    fig, axes = plt.subplots(
        nc, nc, figsize=(2 * nc, 2.4 * nc), constrained_layout=True
    )
    tx = np.arange(0, 13.6, 0.5)
    t1d = np.arange(0, 13.6, 0.1)
    extent = (tx[0], tx[-1], tx[0], tx[-1])
    xlim = extent[:2]
    # to convert lookback times into Universe ages
    tmax = 13.7
    iname = 1
    for i, ev_i in enumerate(events):
        xcol = f"history:{ev_i}:time"
        x = satellites[xcol] if use_lookback else tmax - satellites[xcol]
        m = ["Mbound", "Mstar", "Mgas"]  # , 'M200Mean']
        # m += [f'history:{ev}:{mi}' for mi in m for ev in ('birth',)+events]
        m += [f"history:{ev}:{mi}" for mi in m for ev in events]
        ic("---")
        ic(xcol)
        # for mi in m:
        #     r = pearsonr(x, satellites[mi])[0]
        #     ic(mi, r)
        ic("---")
        # also high-M200
        massive = satellites["M200Mean"] > 1e14
        lowmass = satellites["M200Mean"] < 3e13
        ic(
            massive.sum(),
            (massive & (satellites["Rank"] == 1)).sum(),
            lowmass.sum(),
            (lowmass & (satellites["Rank"] == 1)).sum(),
        )
        for j, ev_j in enumerate(events):
            ax = axes[i, j]
            ycol = f"history:{ev_j}:time"
            ic(xcol, ycol)
            y = satellites[ycol] if use_lookback else tmax - satellites[ycol]
            # ic(x, y)
            # diagonal
            if j == i:
                plot_times_hist(x, t1d, ax, xlim, ylim=(0, 0.4), color="k")
                ic("high-M200")
                plot_times_hist(
                    x,
                    t1d,
                    ax,
                    xlim,
                    mask=massive,
                    format_ax=False,
                    color="C1",
                    lw=2.5,
                    fill=False,
                    median_kwargs={"ls": "--"},
                )
                # also low-M200
                ic("low-M200")
                plot_times_hist(
                    x,
                    t1d,
                    ax,
                    xlim,
                    mask=lowmass,
                    format_ax=False,
                    color="C0",
                    lw=2.5,
                    fill=False,
                    median_kwargs={"ls": "--"},
                )
                if xcol == "history:max_Mstar:time":
                    late = x < 0.1
                    n = x.size
                    late_massive = x[massive] == 0
                    late_lowmass = x[lowmass] == 0
                    ic(
                        late.sum() / n,
                        late_massive.sum() / massive.sum(),
                        late_lowmass.sum() / lowmass.sum(),
                    )
                    ic(np.median(x[massive]), np.median(x[lowmass]))
                    # sys.exit()
            # lower triangle
            elif j < i:
                if c_lower is None:
                    ax.axis("off")
                else:
                    im_lower = plot_times_2d(
                        satellites,
                        x,
                        y,
                        tx,
                        ax,
                        i,
                        j,
                        iname,
                        extent,
                        c_lower,
                        # cmap=cmap_lower, cmap_rng=cmap_lower_rng,
                        # vmin=vmin_lower, vmax=vmax_lower, stat=stat_lower)
                        stat=stat_lower,
                        **kwargs_lower,
                    )
            # upper triangle
            elif j > i:
                if c_upper is None:
                    ax.axis("off")
                else:
                    im_upper = plot_times_2d(
                        satellites,
                        x,
                        y,
                        tx,
                        ax,
                        i,
                        j,
                        iname,
                        extent,
                        c_upper,
                        # cmap=cmap_upper, cmap_rng=cmap_upper_rng,
                        # vmin=vmin_upper, vmax=vmax_upper, stat=stat_upper)
                        stat=stat_upper,
                        **kwargs_upper,
                    )
            ax.set(xlim=xlim)
            if j != i:
                ax.plot(tx, tx, "k--", lw=1)
                ax.set(ylim=xlim)
                ax.tick_params(which="both", length=3)
            format_ax(ax, i, j, xlim, nc)
            iname += 1
    # colorbars
    # quick fix for now
    cmap_lower = kwargs_lower.get("cmap", "viridis")
    cmap_lower_rng = kwargs_lower.get("cmap_rng", (0, 1))
    cmap_upper = kwargs_upper.get("cmap", "viridis")
    cmap_upper_rng = kwargs_upper.get("cmap_rng", (0, 1))
    show_colorbar(
        axes,
        im_lower,
        c_lower,
        cmap_lower,
        cmap_lower_rng,
        stat_lower,
        location="bottom",
    )
    show_colorbar(
        axes, im_upper, c_upper, cmap_upper, cmap_upper_rng, stat_upper, location="top"
    )
    # save!
    c_lower = hbt_tools.format_colname(c_lower)
    c_upper = hbt_tools.format_colname(c_upper)
    output = f"compare_times/comparetimes__{c_lower}__{c_upper}"
    if "birth" in events:
        output = f"{output}__birth"
    hbt_tools.save_plot(fig, output, satellites.sim, tight=False, h_pad=0.2)
    return


def format_ax(ax, i, j, xlim, ncols, fs=18, labelpad=5):
    # axlabels = ['birth', 'cent', 'sat', 'infall', 'acc',
    axlabels = [
        "cent",
        "sat",
        "infall",
        "acc",
        "$m_\mathrm{sub}^\mathrm{max}$",
        "$m_\mathrm{\u2605}^\mathrm{max}$",
        "$m_\mathrm{gas}^\mathrm{max}$",
    ]
    axlabels = [f"{i} (Gya)" for i in axlabels]
    kwargs = {"fontsize": fs, "labelpad": labelpad}
    # diagonal
    if i == j and i < ncols - 1:
        ax.set(xticks=[])
    # else:
    #     xcol = f'history:{events[i]}:time'
    #     ax.set_xlabel(axlabels[i], **kwargs)
    # left
    if j == 0 and i > 0:
        ax.set_ylabel(axlabels[i], **kwargs)
    else:
        ax.set(ylabel="", yticklabels=[])
    # right
    if j == ncols - 1 and i < ncols - 1:
        rax = ax.twinx()
        rax.plot([], [])
        rax.set_ylabel(axlabels[i], **kwargs)
        rax.set(ylim=xlim)
        format_ticks(rax)
    # bottom
    # if i == ncols - 1:# and j < ncols - 1:
    ax.set_xlabel(axlabels[j], **kwargs)
    if i < ncols - 1:
        ax.set(xlabel="", xticklabels=[])
    # top
    if i == 0:  # and j > 0:
        tax = ax.twiny()
        tax.plot([], [])
        tax.set(xlim=xlim)
        kwargs["labelpad"] += 2
        tax.set_xlabel(axlabels[j], **kwargs)
        format_ticks(tax)
    format_ticks(ax, (i == j))
    return


def format_ticks(ax, diagonal=False):
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_locator(ticker.FixedLocator([5, 10]))
    if diagonal:
        ax.yaxis.set_major_locator(ticker.NullLocator())
    else:
        ax.yaxis.set_major_locator(ticker.FixedLocator([5, 10]))
    ax.tick_params(length=5, width=2)
    return


def plot_times_2d(
    satellites,
    x,
    y,
    bins,
    ax,
    i,
    j,
    axname,
    extent,
    c,
    cmap,
    cmap_rng=(0, 1),
    stat=np.nanmean,
    annotate_r=True,
    **kwargs,
):
    """
    kwargs passed to ``plt.imshow``
    """
    is_lower = j < i
    # these are for correlations and for annotations
    try:
        r, pr = pearsonr(x, y)
    except ValueError:
        r = np.nan
    h2d = np.histogram2d(x, y, bins=bins)[0]
    ntot = x.size
    if c == "corr":
        ic(cmap, cmap_rng)
        cmap = cmr.get_sub_cmap(cmap, *cmap_rng)
        ic(cmap)
        ic(r)
        try:
            color = cmr.take_cmap_colors(cmap, N=1, cmap_range=(r, r))[0]
        except ValueError:
            color = "C0"
        cmap_ij = mplcolors.LinearSegmentedColormap.from_list(
            "cmap_ij", [[1, 1, 1], color]
        )
        # making vmin slightly <0 so that the colormaps don't include white
        # but those equal to zero should be nan so empty space is still white
        # h2d[h2d == 0] = np.nan
        im = ax.imshow(
            h2d,
            extent=extent,
            cmap=cmap_ij,
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=0.025 * ntot,
        )
        h2d[np.isnan(h2d)] = 0
        ic(np.triu(h2d).sum() / ntot, np.tril(h2d).sum() / ntot)
    else:
        ic(cmap)
        # cmap = cmr.get_sub_cmap(cmap, *cmap_rng)
        m2d = binned_statistic_2d(x, y, satellites[c], stat, bins=bins)[0]
        # if not is_lower:
        # m2d = m2d.T
        # im = ax.imshow(
        #     np.log10(m2d), origin='lower', aspect='auto', vmin=vmin,
        #     vmax=vmax, cmap=cmap, extent=extent)
        # if vmin is not None: vmin = 10**vmin
        # if vmax is not None: vmax = 10**vmax
        im = ax.imshow(
            m2d,
            origin="lower",
            aspect="auto",
            extent=extent,
            # cmap=cmap, norm=mplcolors.LogNorm(vmin=vmin, vmax=vmax))
            **kwargs,
        )
    # annotate correlation coefficient
    if annotate_r:
        # annot_top = (f'{r:.2f}\n({axname})', (0.05, 0.95), 'left', 'top')
        # annot_bottom = (f'({axname})\n{r:.2f}', (0.95, 0.05), 'right', 'bottom')
        annot_top = (f"{r:.2f}", (0.03, 0.97), "left", "top")
        annot_bottom = (f"{r:.2f}", (0.97, 0.03), "right", "bottom")
        ic(axname, is_lower, r, np.triu(h2d, 2).sum() / h2d.sum())
        if np.triu(h2d.T, 2).sum() / h2d.sum() < 0.2:
            annot = annot_top  # if is_lower else annot_bottom
            # annot = annot_top
        else:
            annot = annot_bottom  # if is_lower else annot_top
            # annot = annot_top
        label, xy, ha, va = annot
        ax.annotate(label, xy=xy, xycoords="axes fraction", ha=ha, va=va, fontsize=14)
    return im


def plot_times_hist(
    x,
    tx,
    ax,
    xlim,
    bw=0.1,
    ylim=None,
    mask=None,
    color="C0",
    fill=True,
    lw=1.5,
    format_ax=True,
    use_sns=True,
    show_median=True,
    median_kwargs={},
    **kwargs,
):
    if mask is not None:
        x = x[mask]
        ic(mask.sum(), mask.sum() / mask.size)
        # x[~mask] = -1
    if use_sns:
        sns.kdeplot(x, bw_method=bw, ax=ax, color=color, fill=fill, lw=lw, **kwargs)
    else:
        kde = gaussian_kde(x, bw)
        kde = kde(tx)
        if mask is not None:
            kde = kde * mask.sum() / mask.size
        ax.plot(tx, kde, color=color)
        if fill:
            ax.fill_between(tx, kde, color=color, alpha=0.2)
    ic(np.median(x))
    if show_median:
        ax.axvline(np.median(x), color=color, lw=1.5, **median_kwargs)
    if format_ax:
        ax.tick_params(which="both", length=5)
        ax.set(yticks=[], xlim=xlim, ylim=ylim, ylabel="")
    return


def show_colorbar(axes, im, c, cmap, cmap_rng, stat, location="left", logstat=False):
    assert location in ("left", "right", "bottom", "top")
    orientation = "vertical" if location in ("left", "right") else "horizontal"
    if c == "corr":
        cmap = cmr.get_sub_cmap(cmap, *cmap_rng)
        cbar = cm.ScalarMappable(
            norm=mplcolors.Normalize(vmin=0.2, vmax=0.9), cmap=cmap
        )
        cbar = plt.colorbar(
            cbar,
            ax=axes,
            location=location,
            fraction=0.1,
            aspect=30,
            label="Correlation",
        )
    else:
        if isinstance(stat, str):
            statlabel = stat
        elif stat in (np.mean, np.nanmean):
            statlabel = "mean"
        elif stat in (np.median, np.nanmedian):
            statlabel = "median"
        if logstat:
            statlabel = f"log {statlabel}"
        if "/" in c:
            c = c.split("/")
            label = f"{binlabel[c[0]]}/{binlabel[c[1]]}"
        else:
            label = binlabel[c]
        if stat in ("mean", np.mean, np.nanmean):
            label = rf"$\langle {label} \rangle$"
        elif stat in ("median", np.median, np.nanmedian):
            label = f"med(${label}$)"
        elif stat in ("std", np.std, np.nanstd):
            label = rf"$\sigma({label})$"
        cbar = plt.colorbar(
            im,
            ax=axes,
            location=location,
            fraction=0.1,
            aspect=30,
            # label=f'{statlabel}(${label}$)')
            label=label,
        )
        ic(im.norm, isinstance(im.norm, LogNorm))
        if isinstance(im.norm, LogNorm):
            cbar_ticks = cbar.get_ticks()
            ic(cbar_ticks)
            ax = cbar.ax.xaxis if orientation == "horizontal" else cbar.ax.yaxis
            # if (cbar_ticks.min() >= 0.001) and (cbar_ticks.max() <= 1000):
            if (im.norm.vmin > 1e-4) and (im.norm.vmax < 10000):
                fmt = "%d" if im.norm.vmin > 0.1 else "%s"
                ax.set_major_formatter(ticker.FormatStrFormatter(fmt))
    return


main()
