from astropy import units as u
import cmasher as cmr
from icecream import ic
from matplotlib import cm, colors as mplcolors, pyplot as plt, ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import multiprocessing as mp
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic as binstat, binned_statistic_dd as binstat_dd
import sys
from time import time
from tqdm import tqdm
import warnings

from plottery.plotutils import colorscale, savefig, update_rcParams
from plottery.statsplots import contour_levels

update_rcParams()

from astro.clusters.conversions import rsph
from lnr import to_linear, to_log

from ..subhalo import Subhalos
from ..hbt_tools import format_colname, save_plot

# from hbtpy.hbtplots import RelationPlotter
# from hbtpy.hbtplots.core import ColumnLabel
from .plot_auxiliaries import (
    binning,
    definitions,
    get_axlabel,
    get_bins,
    get_label_bincol,
    logcenters,
    massbins,
    plot_line,
)
from .plot_definitions import (
    ccolor,
    scolor,
    massnames,
    units,
    xbins,
    binlabel,
    event_names,
    axlabel,
    ylims,
)


def run(args, sim, subs, logM200Mean_min, which_relations, ncores=1):
    def print_header(relation_type):
        print(f"\n########\n### {relation_type} relations\n########")

    use_mp = ncores > 1
    # just using for labels for now
    # plotter = RelationPlotter(sim, subs)
    # label = ColumnLabel()

    plot_args = []
    # x-axis: time since historical event
    for stat in args.stats:
        if "time" in which_relations:
            print_header("time")
            out = wrap_relations_time(args, stat)
            plot_args.extend(out)

        # x-axis: cluster-centric distance
        if "distance" in which_relations:
            print_header("distance")
            out = wrap_relations_distance(args, stat)
            plot_args.extend(out)

        relation_kwargs = dict(
            satellites_label="Satellites today",
            centrals_label="Centrals today",
            show_centrals=True,
            min_hostmass=logM200Mean_min,
            show_ratios=False
            # , xlim=(3e7,1e12))
        )

        # historical HSMR binned by present-day quantities
        if "hsmr-history" in which_relations:
            print_header("HSMR history")
            out = wrap_relations_hsmr_history(args, stat, **relation_kwargs)
            plot_args.extend(out)

        # present-day HSMR binned by historical quantities
        relation_kwargs["satellites_label"] = "All satellites"
        relation_kwargs["centrals_label"] = "Centrals today"
        if "hsmr" in which_relations:
            print_header("HSMR")
            out = wrap_relations_hsmr(args, stat, **relation_kwargs)
            plot_args.extend(out)

    nplots = len(plot_args)
    print()
    print()
    print(f"Producing {nplots} plots. Let's go!")
    print()
    print()

    if use_mp:
        verbose = lambda i: (nplots <= 10) or (i % (nplots // 10) == 0)
        with mp.Pool(args.ncores) as pool:
            # pbar = tqdm(total=nplots) if not verbose else None
            # with tqdm(total=len(plot_args)) as pbar:
            #     for args_i, kwargs_i in plot_args:
            #         pool.apply_async(
            #             wrap_relation, args=(args,sim,subs,*args_i),
            #             kwds={**kwargs_i, 'verbose': False})
            #         pbar.update()
            _ = [
                pool.apply_async(
                    wrap_relation,
                    args=(args, sim, subs, *args_i),
                    kwds={**kwargs_i, "verbose": verbose(i)},
                )
                for i, (args_i, kwargs_i) in enumerate(plot_args)
            ]
            pool.close()
            pool.join()
            # if pbar is not None:
            #     pbar.close()
    else:
        verbose = nplots <= 10
        iterator = plot_args if verbose else tqdm(plot_args, total=nplots)
        _ = [
            wrap_relation(args, sim, subs, *args_i, **kwargs_i, verbose=verbose)
            for args_i, kwargs_i in iterator
        ]
    print(f"\nProduced {nplots} plots!")
    return


def wrap_relation(
    args,
    sim,
    subs,
    xcol,
    ycol,
    bincol,
    x_bins=10,
    bins=None,
    xscale="log",
    logbins=None,
    show_satellites=False,
    statistic="mean",
    pbar=None,
    **kwargs,
):
    label = get_label_bincol(bincol)
    # print('\n'*5)
    ic()
    ic(bincol, bins, logbins)
    if logbins is None:
        if bincol is not None:
            logbins = not (
                "time" in bincol
                or (
                    "/" in bincol
                    and (bincol.count("Mbound") == 2 or bincol.count("Mstar") == 2)
                )
            )
        else:
            logbins = False
    if bins is None:
        bins = get_bins(bincol, logbins)
    if "yscale" not in kwargs:
        kwargs["yscale"] = "log" if statistic == "mean" else "linear"
    if statistic not in ("count", "mean") and (
        "ylim" not in kwargs or kwargs["ylim"] is None
    ):
        kwargs["ylim"] = (0, 0.6) if statistic == "std" else (0, 1.5)
    if "xbins" in kwargs:
        x_bins = kwargs.pop("xbins")
    if statistic == "count":
        kwargs["show_centrals"] = False
        show_satellites = False
    plot_relation(
        sim,
        subs,
        xcol=xcol,
        ycol=ycol,
        xbins=x_bins,
        statistic=statistic,
        xscale=xscale,
        bincol=bincol,
        binlabel=label,
        bins=bins,
        logbins=logbins,
        xlabel=get_axlabel(xcol, "mean"),
        ylabel=get_axlabel(ycol, statistic),
        show_satellites=show_satellites,
        **kwargs,
    )
    # selection=selection, selection_min=selection_min,
    # show_satellites=show_satellites, show_centrals=show_centrals)
    if pbar is not None:
        pbar.update()
    return


def wrap_relations_distance(args, stat):
    xscale = "log"
    # xb = np.logspace(-2, 0.5, 8) if xscale == 'log' \
    #     else np.linspace(0, 3, 8)
    xb = np.logspace(-2, 0.7, 10) if xscale == "log" else np.linspace(0, 3, 8)
    plot_args = []
    # convenience aliases
    d = "ComovingMostBoundDistance"
    vpec = "PhysicalMostBoundPeculiarVelocity"
    vdisp = "PhysicalMostBoundHostVelocityDispersion"
    # defined here to avoid repetition
    ycols_present = [
        "Mbound/Mstar",
        "Mstar/Mbound",
        "Mstar",
        "Mbound",
        "Mbound/M200Mean",
        f"{vpec}2/{vdisp}2",
    ]
    bincols_present = ["M200Mean", "Mbound", "Mstar"]
    events = (
        "max_Mstar",
        "max_Mbound",
        "max_Mdm",
        "sat",
        "cent",
        "first_infall",
        "last_infall",
    )
    # events = ("max_Mbound",)
    for ie, event in enumerate(events):
        h = f"history:{event}"
        ycols = [
            f"Mbound/{h}:Mbound",
            f"Mstar/{h}:Mstar",
            f"{h}:time",
            f"{h}:Mbound/{h}:Mstar",
            f"{h}:Mbound",
            f"{h}:Mstar",
        ] + ycols_present
        # ycols = ycols_present
        bincols = [
            f"{h}:time",
            f"{h}:Mstar",
            f"{h}:Mbound",
            f"{h}:Mbound/{h}:Mstar",
        ] + bincols_present
        # ycols = ["Mbound/Mstar"]
        # bincols = ['Mstar']
        # uncomment when testing
        # ycols = ['Mbound']
        # bincols = ['Mstar']
        # ycols = ['Mbound/Mstar', 'Mbound/history:first_infall:Mbound',
        #          'Mbound/history:max_Mbound:Mbound']
        bincols = [
            f"{h}:Mstar",
            "history:max_Mstar:Mstar",
            f"{h}:Mbound",
            "history:max_Mbound:Mbound",
            f"{h}:time",
            f"Mbound/{h}:Mbound",
            f"Mstar/{h}:Mstar",
            f"{h}:Mbound/{h}:Mstar",
        ] + bincols_present
        # bincols = ["Mstar"]
        # change by hand for now
        ev = event_names[event]
        selection_kwds = [
            # {},
            {
                "selection": f"{h}:Depth",
                "selection_min": [-0.5, 0.5],
                "selection_max": [0.5, 10],
                "selection_labels": [
                    "Direct infallers",
                    "Indirect infallers",
                    "Infall as 2nd+ order",
                ],
            },
            # {'selection': 'Depth',
            #     'selection_min': [0.5, 1.5],
            #     'selection_max': [1.5, 10],
            #     'selection_labels': ['1st order today',
            #                         '2nd+ order today',
            #                         '3rd+ order today']},
            # {
            #     "selection": f"{h}:Mstar",
            #     "selection_min": [1e9, 1e10],
            #     "selection_max": [1e10, 3e10, 1e11],
            #     "selection_labels": [
            #         f"$9<\log m_{{\u2605}}^\mathrm{{{ev}}}<10$",
            #         f"$10<\log m_{{\u2605}}^\mathrm{{{ev}}}<10.5$",
            #         f"$10.5<\log m_{{\u2605}}^\mathrm{{{ev}}}<11$",
            #     ],
            # },
            # {'selection': 'Mstar',
            #     'selection_min': [3e9, 3e10],
            #     'selection_max': [1e10, 1e11],
            #     'selection_labels': ['$9.5<\log m_{\u2605}<10$',
            #                          '$10.5<\log m_{\u2605}<11$']},
            # {'selection': f'{h}:Mbound',
            #     'selection_min': [3e10, 1e11],
            #     'selection_max': [1e11, 3e11],
            #     'selection_labels':
            #         [f'$10.5<\log m_\mathrm{{sub}}^\mathrm{{{ev}}}<11$',
            #          f'$11<\log m_\mathrm{{sub}}^\mathrm{{{ev}}}<11.5$']},
            # {'selection': 'Mbound',
            #     'selection_min': [1e10, 3e10],
            #     'selection_max': [3e10, 1e11],
            #     'selection_labels': ['$10<\log m_\mathrm{sub}<10.5$',
            #                          '$10.5<\log m_\mathrm{sub}<11$']}
        ]
        # ycols = ["Mbound/Mstar"]
        # bincols = ["history:first_infall:time"]
        for ycol in ycols:
            if stat == "mean" and ycol in ("Mbound/Mstar", f"{h}:Mbound/{h}:Mstar"):
                ylim = (2, 100)
            else:
                ylim = None
            # ylim = None
            for xyz in ("01", ""):
                # for xyz in ('01',):
                # for xcol in (f'{d}{xyz}/R200MeanComoving', f'{d}{xyz}'):
                for xcol in (f"{d}{xyz}/R200MeanComoving",):
                    # 2d histogram
                    if stat == "count":
                        kwds = dict(
                            x_bins=np.logspace(-2, 0.7, 26),
                            ybins=20,
                            selection="Mstar",
                            selection_min=1e8,
                            xscale="log",
                            bins=None,
                            statistic=stat,
                            lines_only=False,
                            ylim=ylims.get(ycol),
                        )
                        plot_args.append([[xcol, ycol, None], kwds.copy()])
                    # line plots
                    else:
                        lit = (len(xyz) > 0) and stat == "mean"
                        kwds = dict(
                            x_bins=xb,
                            xscale=xscale,
                            literature=lit,
                            selection="Mstar",
                            selection_min=1e8,
                            ylim=ylim,
                            show_satellites=False,
                            show_centrals=False,
                            show_ratios=False,
                            statistic=stat,
                        )
                        for bincol in bincols:
                            # avoid repetition and trivial plots
                            if (
                                ie > 0
                                and bincol in bincols_present
                                and ycol in ycols_present
                            ) or bincol == ycol:
                                continue
                            logbins = "time" not in bincol
                            if bincol == "Mstar":
                                kwds["bins"] = np.logspace(9, 11, 5)
                            else:
                                kwds["bins"] = get_bins(
                                    bincol, logbins, n=4 + ("time" in bincol)
                                )
                            if ycol == "Mbound/Mstar" and bincol == "M200Mean":
                                kwds["ylim_ratios"] = (0.5, 1.5)
                            else:
                                kwds["ylim_ratios"] = None
                            for selkwds in selection_kwds:
                                plot_args.append(
                                    [[xcol, ycol, bincol], {**kwds, **selkwds}]
                                )
        #             break
        #         break
        #     break
        # break
    return plot_args


def wrap_relations_hsmr(args, stat, **relation_kwargs):
    x_bins = {"Mstar": np.logspace(7.5, 11.7, 10), "Mbound": np.logspace(8.5, 12.7, 10)}
    events = [
        f"history:{e}"
        for e in (
            "first_infall",
            # "last_infall",
            # "cent",
            "sat",
            "max_Mstar",
            "max_Mbound",
            "max_Mdm",
        )
    ]
    history_bincols = [
        (
            # f"{h}:Mbound",
            # f"{h}:Mstar",
            # f"{h}:Mdm",
            # f"{h}:Mstar/{h}:Mbound",
            # f"{h}:Mbound/{h}:Mstar",
            # f"{h}:Mdm/{h}:Mbound",
            # f"{h}:Mstar/{h}:Mdm",
            # f"Mbound/{h}:Mbound",
            # f"Mstar/{h}:Mbound",
            # f"Mdm/{h}:Mdm",
            f"{h}:time",
        )
        for h in events
    ]
    bincols = [
        # "ComovingMostBoundDistance",
        # "ComovingMostBoundDistance0",
        # "ComovingMostBoundDistance/R200MeanComoving",
        # "ComovingMostBoundDistance0/R200MeanComoving",
        # "LastMaxMass",
        # "Mbound/LastMaxMass",
        # "Mstar/LastMaxMass",
        "M200Mean",
        # "Mbound/M200Mean",
        "Mbound",
    ]
    bincols = (
        bincols
        + [col for cols in history_bincols for col in cols]
        # + ["history:sat:time-history:first_infall:time"]
    )
    # bincols = ["history:first_infall:time"]
    # bincols = ["history:sat:time-history:first_infall:time"]
    # bincols = ['history:first_infall:time', 'history:sat:time',
    #            'history:max_Mbound:time']
    # bincols = ['Mstar/history:last_infall:Mbound']
    # bincols = ['ComovingMostBoundDistance/R200MeanComoving',
    #            'ComovingMostBoundDistance01/R200MeanComoving']
    # bincols = ["M200Mean"]
    # ic(bincols)
    kwds_count = dict(
        x_bins=20,
        ybins=20,
        selection="Mstar",
        selection_min=1e8,
        force_selection=True,
        statistic="count",
        lines_only=False,
        show_satellites=False,
    )
    # for ycol in ('Mbound', 'Mbound/Mstar', 'Mtotal', 'Mtotal/Mstar'):
    plot_args = []
    # ycols = ['NboundType4']
    # ylims = [None]
    ycols = ("Mbound/Mstar", "Mstar/Mbound", "Mbound", "M200Mean")
    ylims = ((3, 200), (2e-3, 0.5), (5e8, 2e14), (1e13, 5e14))
    # ycols = ycols[:1]
    # bincols = ['M200Mean']
    selection_kwds = [
        # {},
        {
            "selection": f"history:first_infall:Depth",
            "selection_min": [-0.5, 0.5],
            "selection_max": [0.5, 10],
            "selection_labels": [
                "Direct infallers",
                "Indirect infallers",
                "Infall as 2nd+ order",
            ],
        },
        # {
        #     "selection": f"history:sat:time-history:first_infall:time",
        #     "selection_min": [0, 2, 5][:3],
        #     "selection_max": [2, 5, 14][:3],
        #     "selection_labels": [
        #         "$t_\mathrm{sat}-t_\mathrm{infall} < 2$",
        #         "$2 < t_\mathrm{sat}-t_\mathrm{infall} < 5$",
        #         "$t_\mathrm{sat}-t_\mathrm{infall} > 5$",
        #     ],
        # },
        # {
        #     "selection": f"history:first_infall:time",
        #     "selection_min": [0, 0.01, 4],
        #     "selection_max": [0.01, 4, 10],
        #     "selection_labels": [
        #         "$t_\mathrm{infall} < 5$",
        #         "$5 < t_\mathrm{infall} < 10$",
        #     ],
        # },
        # {
        #     "selection": "M200Mean",
        #     "selection_min": [1e13, 1e14],
        #     "selection_max": [1e14, 1e15],
        #     "selection_labels": [
        #         "$M_\mathrm{200m}<10^{14}$ M$_\odot$",
        #         "$M_\mathrm{200m}>10^{14}$ M$_\odot$",
        #     ],
        # },
    ]
    for ycol, ylim in zip(ycols[:1], ylims):
        # histograms
        if stat == "count":
            kwds = {**kwds_count, **dict(ylim=ylim)}
            kwds["yscale"] = "linear" if "/" in ycol else "log"
            # plot_args.append([['Mstar', ycol, None], kwds.copy()])
        # lines
        else:
            for bincol in bincols:
                # if bincol != 'M200Mean': continue
                logbins = "time" not in bincol
                bins = get_bins(bincol, logbins, n=8 + ("time" in bincol))
                bins = get_bins(bincol, logbins, n=5)
                if bincol.count("time") == 1:
                    bins = np.array([0, 4, 8, 12])
                kwds = {
                    **dict(
                        bins=bins,
                        statistic=stat,
                        show_ratios=False,
                        show_satellites=False,
                        show_centrals=True,
                    ),
                    **relation_kwargs,
                }
                kwds["ylim"] = ylim if stat == "mean" else None
                for xcol in ("Mstar",):  # "Mbound"):
                    if stat == "std":
                        xb = get_bins(xcol, ("time" not in xcol), 6)
                    else:
                        xb = xbins.get(xcol, 6)
                    for selkwds in selection_kwds:
                        plot_args.append(
                            [[xcol, ycol, bincol, xb], {**kwds, **selkwds}]
                        )
                    # break
        # break
    return plot_args


def wrap_relations_hsmr_history(
    args, stat, do_mass=True, do_ratios=False, xlim=None, **relation_kwargs
):
    """Plot historical quantities on the x-axis"""
    plot_args = []
    kwds = dict(statistic=stat, **relation_kwargs)
    # plot_args.append(
    #     [['Mstar/history:first_infall:Mbound',
    #       'Mbound/history:first_infall:Mbound', None],
    #      dict(statistic='count')])
    events = ("first_infall", "last_infall", "cent", "sat", "max_Mstar", "max_Mbound")
    events = (
        "first_infall",
        # "sat",
        # "max_Mbound",
    )
    selection_kwds = [
        # {},
        {
            "selection": f"history:first_infall:Depth",
            "selection_min": [-0.5, 0.5],
            "selection_max": [0.5, 10],
            "selection_labels": [
                "Direct infallers",
                "Indirect infallers",
                "Infall as 2nd+ order",
            ],
        },
        # {
        #     "selection": "M200Mean",
        #     "selection_min": [1e13, 1e14],
        #     "selection_max": [1e14, 1e15],
        #     "selection_labels": [
        #         "$M_\mathrm{200m}<10^{14}$ M$_\odot$",
        #         "$M_\mathrm{200m}>10^{14}$ M$_\odot$",
        #     ],
        # },
        # {
        #     "selection": f"history:sat:time-history:first_infall:time",
        #     "selection_min": [0, 2, 4][:3],
        #     "selection_max": [2, 4, 14][:3],
        #     "selection_labels": [
        #         "$t_\mathrm{sat}-t_\mathrm{infall} < 2$",
        #         "$2 < t_\mathrm{sat}-t_\mathrm{infall} < 4$",
        #         "$t_\mathrm{sat}-t_\mathrm{infall} > 4$",
        #     ],
        # },
    ]
    for ie, event in enumerate(events):
        h = f"history:{event}"
        # ratios in the x-axis
        if do_ratios:
            xcols = (
                [f"{m}/{h}:{m}" for m in ("Mbound", "Mdm", "Mgas", "Mstar")]
                + [f"{h}:{m}/{h}:Mbound" for m in ("Mdm", "Mgas", "Mstar")]
                + [
                    f"{m}:history:max_{m}:{m}"
                    for m in ("Mbound", "Mdm", "Mgas", "Mstar")
                ]
            )
            ycols = (
                [f"{m}/{h}:{m}" for m in ("Mbound", "Mdm", "Mgas", "Mstar")]
                + [f"{h}:{m}/{h}:Mbound" for m in ("Mdm", "Mgas", "Mstar")]
                + [
                    f"{m}:history:max_{m}:{m}"
                    for m in ("Mbound", "Mdm", "Mgas", "Mstar")
                ]
            )
            ic(np.sort(xcols))
            ic(np.sort(ycols))
            bincols = [
                f"{h}:z",
                f"{h}:time",
                "M200Mean",
                "Mbound/M200Mean",
                "Mstar",
                f"{h}:Mstar",
            ] + [f"history:{e}:time-{h}:time" for e in events if e != event]
            kwds["show_centrals"] = False
            kwds["show_1to1"] = True
            kwds["yscale"] = "log"
            kwds["xbins"] = 25
            kwds["ybins"] = 25
            for xc in xcols:
                for yc in ycols:
                    if "max_Mstar" not in xc and "max_Mdm" not in yc:
                        continue
                    if xc == yc:
                        continue
                    if stat == "count":
                        plot_args.append([[xc, yc, None], kwds.copy()])
                        continue
                    for bincol in bincols:
                        if xc == bincol or yc == bincol:
                            continue
                        logbins = "time" not in bincol
                        ic(bincol, logbins)
                        bins = get_bins(bincol, logbins, n=5)
                        kwds["logbins"] = logbins
                        plot_args.append([[xc, yc, bincol], kwds.copy()])
                        # for xc, yc, xb in zip(xcol, ycol, x_bins):
                        #     kwds['xlim'] = (xb[0], xb[-1])
                        #     kwds['ylim'] = (xb[0], xb[-1])
                        #     plot_args.append([[xc, yc, bincol, xb], kwds.copy()])
        # mass in the x-axis
        if do_mass:
            bincols = (
                f"{h}:z",
                f"{h}:time",
                "Mstar",
                "Mbound",
                "Mstar/Mbound",
                "Mbound/Mstar",
                "Mdm/Mstar",
                "LastMaxMass",
                "Mbound/LastMaxMass",
                "Mstar/LastMaxMass",
                "M200Mean",
                "Mbound/M200Mean",
                "ComovingMostBoundDistance",
                "ComovingMostBoundDistance0",
                "ComovingMostBoundDistance/R200Mean",
                "ComovingMostBoundDistance0/R200Mean",
            )
            # bincols = ('Mstar',)
            bincols = [f"{h}:time"]
            ycols = [
                f"{h}:Mbound",
                f"{h}:Mbound/{h}:Mstar",
                f"{h}:Mbound/Mstar",
                f"Mbound/{h}:Mbound",
                f"Mstar/{h}:Mstar",
            ]
            ylims = [(1e9, 1e13), (3, 200), (30, 200), (0.01, 2), (0.01, 2)]
            ycols = ycols + [f"{h}:{m}" for m in ("Mbound", "Mdm", "Mgas", "Mstar")]
            ylims = ylims + [(1e8, 1e13), (1e8, 1e13), (1e7, 1e11), (1e7, 1e12)]
            # ycols = ['Mbound']
            # ylims = [(1e8, 1e13)]
            ycols = ycols[1:2]
            # ylims = ylims[1:2]
            ylims = [(20, 200)]
            if stat == "count":
                kwds = {"statistic": "count", **relation_kwargs}
                for ycol in ycols:
                    # for xcol in ('Mstar', f'{h}:Mstar'):
                    for xcol in (f"{h}:Mbound",):
                        if xcol == ycol:
                            continue
                        kwds["xbins"] = 25
                        kwds["ybins"] = 25
                        kwds["yscale"] = "log"
                        plot_args.append([[xcol, ycol, None], kwds.copy()])
                        kwds.pop("xbins")
                        kwds.pop("ybins")
                continue
            for bincol in bincols:
                # if 'time' not in bincol: continue
                logbins = "time" not in bincol and bincol.split(":")[-1] != "z"
                ic(bincol, logbins)
                if bincol.count("time") == 1:
                    bins = np.array([0, 4, 8, 12])
                else:
                    bins = get_bins(
                        bincol, logbins, n=None if bincol.split(":")[-1] == "z" else 4
                    )
                kwds = {
                    **dict(bins=bins, statistic=stat, logbins=logbins),
                    **relation_kwargs,
                }
                kwds["show_centrals"] = True
                kwds["show_satellites"] = False
                kwds["show_ratios"] = False
                for ycol, ylim in zip(ycols, ylims):
                    # for ycol in ycols
                    # else:
                    # for xcol in ('Mstar', f'{h}:Mstar'):#, f'{h}:Mbound'):
                    for xcol in (f"{h}:Mstar",):
                        # if 'history' in xcol or False:
                        #     xb = np.logspace(9, 11.5, 11)
                        #     kwds['show_centrals'] = False
                        # else:
                        #     kwds['show_centrals'] = True
                        xb = xbins[xcol.split(":")[-1]]
                        kwds["ylim"] = ylim if stat == "mean" else (0, 0.6)
                        for selkwds in selection_kwds:
                            plot_args.append(
                                [[xcol, ycol, bincol, xb], {**selkwds, **kwds}]
                            )
                continue
                #
                x_bins = np.logspace(9, 13, 10)
                kwds["xlim"] = (x_bins[0], x_bins[-1])
                for ycol in ("Mbound", "Mstar"):
                    plot_args.append(
                        [[f"{h}:Mbound", ycol, bincol, x_bins], kwds.copy()]
                    )
                # break
                # a few additional ones
                # plot_args.append(
                #     [['history:sat:Mstar', 'history:first_infall:Mstar',
                #     np.logspace(9.5, 12.5, 7)], kwds.copy()])
                # plot_args.append(
                #     [['history:sat:Mbound', 'history:first_infall:Mbound',
                #     np.logspace(10, 13, 7)], kwds.copy()])
    return plot_args


def wrap_relations_time(args, stat, do_time_differences=False):
    # x_bins = np.arange(0, 14, 1)
    plot_args = []
    kwds_count = dict(
        x_bins=30,
        ybins=30,
        selection="Mstar",
        selection_min=1e8,
        xscale="linear",
        statistic="count",
        yscale="log",
        show_satellites=True,
        lines_only=False,
    )
    kwds = dict(
        x_bins=xbins["time"],
        logbins=True,
        selection="Mstar",
        statistic=stat,
        xscale="linear",
        yscale="log",
        selection_min=1e8,
        show_satellites=False,
        show_ratios=False,
    )
    events = (
        "max_Mstar",
        "max_Mbound",
        "max_Mdm",
        "sat",
        # "cent",
        "first_infall",
        # "last_infall",
    )
    # events = ('first_infall',)
    # events = ('max_Mdm', 'max_Mstar', 'first_infall')
    for ie, event in enumerate(events):
        # for event in ('max_Mbound', 'max_Mstar'):
        # if 'first' not in event: continue
        h = f"history:{event}"
        xcols = [f"{h}:z", f"{h}:time", "history:first_infall:time"] + [
            f"history:{e}:time-{h}:time" for e in events if e != event
        ]
        xcols = [f"{h}:time"]
        xcols = ["history:first_infall:time"]
        for xcol in xcols:
            if "time" not in xcol:
                continue
            # for xcol in (f'{h}:time',):
            # xcol = f'{h}:time'
            # histograms
            if stat == "count":
                # ycols = [f'{h}:Mbound/{h}:Mstar']
                # ycols = [f'{h}:time',
                #          f'history:first_infall:Mbound/{h}:Mbound',
                #          f'history:first_infall:Mdm/{h}:Mdm',
                #          f'history:first_infall:Mstar/{h}:Mstar',
                #          f'{h}:Mbound/LastMaxMass',
                #          'ComovingMostBoundDistance/R200MeanComoving',
                #          'ComovingMostBoundDistance0/R200MeanComoving']
                # for ycol in ('Mbound', 'Mbound/Mstar', f'{h}:Mbound',
                #              f'Mbound/{h}:Mbound', f'Mstar/{h}:Mbound',
                #              f'Mstar/{h}:Mstar', 'M200Mean', 'Mbound/M200Mean',
                #              f'Mdm/{h}:Mdm', f'Mgas/{h}:Mgas',
                #              f'Mstar/Mbound', f'Mdm/Mbound', f'Mgas/Mbound',
                #              'ComovingMostBoundDistance/R200MeanComoving'):
                ycols = [
                    f"{h}:Mstar",
                    f"Mstar/{h}:Mstar",
                    "Mstar",
                    f"{h}:Mbound",
                    f"Mbound/{h}:Mbound",
                    f"{h}:Mbound/{h}:Mstar",
                    "Mbound",
                    f"Mstar-{h}:Mstar",
                    f"Mbound-{h}:Mbound",
                ] + [f"history:{e}:time-{h}:time" for e in events if e != event]
                # ycols = [f'{h}:Mbound']
                if "time" in xcol:
                    ycols = ycols + [
                        f"history:{e2}:time-{h}:time" for e2 in events if e2 != event
                    ]
                for ycol in ycols:
                    kwds_count["ylim"] = ylims.get(ycol)
                    kwds_count["yscale"] = (
                        "linear" if ("Distance" in ycol or "time" in ycol) else "log"
                    )
                    kwds_count["ybins"] = (
                        np.linspace(0, 1.5, 15) if "Distance" in ycol else 20
                    )
                    plot_args.append([[xcol, ycol, None], kwds_count.copy()])
                if do_time_differences:
                    for i, e in enumerate(events):
                        if i == ie:
                            continue
                        hi = f"history:{e}"
                        ycols = (f"{hi}:time", f"{hi}:time-{xcol}")
                        for ycol in ycols:
                            if xcol == ycol:
                                continue
                            kwds_count["ylim"] = ylims.get(ycol)
                            kwds_count["yscale"] = (
                                "linear"
                                if ("Distance" in ycol or "time" in ycol)
                                else "log"
                            )
                            kwds_count["ybins"] = (
                                np.linspace(0, 1.5, 15) if "Distance" in ycol else 20
                            )
                            plot_args.append([[xcol, ycol, None], kwds_count.copy()])
            # lines
            else:
                ycols_ = (
                    [
                        "Mbound/Mstar",
                        # "Mbound/M200Mean",
                        # "ComovingMostBoundDistance/R200MeanComoving",
                        # "ComovingMostBoundDistance0/R200MeanComoving",
                    ]
                    + [f"{m}/{h}:{m}" for m in ("Mbound", "Mdm", "Mgas", "Mstar")]
                    + [f"{h}:{m}/{h}:Mbound" for m in ("Mdm", "Mgas", "Mstar")]
                    + [f"{h}:Mbound/{h}:{m}" for m in ("Mdm", "Mgas", "Mstar")]
                )
                # ycols_ = ycols_ + [
                #     f"{h}:Mstar/history:max_Mstar:Mstar",
                #     f"{h}:Mdm/history:max_Mdm:Mdm",
                #     f"{h}:Mbound/history:max_Mbound:Mbound",
                #     # f"{h}:time-history:max_Mstar:time",
                # ]
                if event == "first_infall":
                    ycols = ycols_
                else:
                    ycols = ycols_ + [
                        f"history:first_infall:Mbound/{h}:Mbound",
                        f"history:first_infall:Mdm/{h}:Mdm",
                        f"history:first_infall:Mgas/{h}:Mgas",
                        f"history:first_infall:Mstar/{h}:Mstar",
                    ]
                bincols = [
                    f"{h}:Mbound",
                    f"{h}:Mbound/{h}:Mstar",
                    "Mstar",
                    f"{h}:Mstar",
                    f"{h}:Mbound/{h}:M200Mean",
                    f"{h}:time",
                    f"{h}:Mgas",
                    f"{h}:Mgas/{h}:Mstar",
                    f"{h}:Mgas/{h}:Mbound",
                    f"{h}:Depth",
                    f"Mstar/{h}:Mstar",
                    "M200Mean",
                ]  # + [f"history:{e}:time-{h}:time" for e in events if e != events]
                # ycols = [f'Mstar/{h}:Mstar']
                # ycols = [
                #     f"history:first_infall:Mstar/{h}:Mstar",
                #     f"history:first_infall:Mdm/{h}:Mdm",
                # ]
                # bincols = ["M200Mean"]
                # ycols = ["Mbound/Mstar"]
                # bincols = ['Mstar', f'{h}:Mstar']
                # bincols = [f"{h}:Mbound/{h}:Mstar"]
                # ycols = [f'Mdm/{h}:Mdm', f'Mstar/{h}:Mstar',
                #          f'Mbound/{h}:Mbound']
                # bincols = ['Depth']
                # bincols = [f'{h}:Mbound', f'{h}:Mgas', f'{h}:Mstar']
                # bincols = [f'{h}:Mgas']
                # bincols = [f'{h}:Mgas', f'{h}:Mgas/{h}:Mbound',
                #            f'{h}:Mgas/{h}:Mstar']
                # bincols = [f'{h}:Mbound', 'Mbound']
                # bincols = ['Mstar']
                # change by hand for now
                selection_kwds = [
                    # {},
                    {
                        "selection": f"history:first_infall:Depth",
                        "selection_min": [-0.5, 0.5],
                        "selection_max": [0.5, 10],
                        "selection_labels": [
                            "Direct infallers",
                            "Indirect infallers",
                            "Infall as 2nd+ order",
                        ],
                    },
                    # {
                    #     "selection": "Depth",
                    #     "selection_min": [0.5, 1.5],
                    #     "selection_max": [1.5, 10],
                    #     "selection_labels": [
                    #         "1st order today",
                    #         "2nd+ order today",
                    #         "3rd+ order today",
                    #     ],
                    # },
                    # {
                    #     "selection": "M200Mean",
                    #     "selection_min": [1e13, 1e14],
                    #     "selection_max": [1e14, 1e15],
                    #     "selection_labels": [
                    #         "$M_\mathrm{200m}<10^{14}$ M$_\odot$",
                    #         "$M_\mathrm{200m}>10^{14}$ M$_\odot$",
                    #     ],
                    # },
                    # {
                    #     "selection": f"{h}:Mbound",
                    #     "selection_min": [1e10, 5e11],
                    #     "selection_max": [5e11, 3e12],
                    #     "selection_labels": [
                    #         rf'{get_label_bincol(f"{h}:Mbound")} $<5\times10^{{11}}$ M$_\odot$',
                    #         rf'{get_label_bincol(f"{h}:Mbound")} $>5\times10^{{11}}$ M$_\odot$',
                    #     ],
                    # },
                ]
                for ycol in ycols:
                    for bincol in bincols:
                        if bincol == ycol or bincol in xcol:
                            continue
                        if bincol == f"{h}:Mbound":
                            bins = np.logspace(10.5, 12.5, 5)
                        elif "Depth" in bincol:
                            bins = np.arange(5) - 0.5
                        else:
                            bins = (
                                np.logspace(9, 11.3, 5)
                                if ("Mstar" in bincol and "/" not in bincol)
                                else get_bins(bincol, n=4)
                            )
                        kwds["yscale"] = (
                            "linear"
                            if ycol in (f"Mstar/{h}:Mstar", f"Mdm/{h}:Mdm")
                            else "log"
                        )
                        kwds["yscale"] = "linear"
                        kwds["ylim"] = (0, 1) if ycol == f"Mdm/{h}:Mdm" else None
                        # kwds['ylim'] = (0.5, 2)
                        kwds["bins"] = bins
                        kwds["logbins"] = not ("time" in bincol or "Depth" in bincol)
                        # plot_args.append([[xcol, ycol, bincol], kwds.copy()])
                        for selkwds in selection_kwds:
                            plot_args.append(
                                [[xcol, ycol, bincol], {**kwds, **selkwds}]
                            )
        # break
    return plot_args


################################################
################################################


def apply_bins(subs, mask, bincol, bins, statistic, logbins, binlabel, cmap):
    has_iterable_bins = np.iterable(bins)
    bindata = subs[bincol]
    mask = mask & np.isfinite(bindata)
    # let's make sure the early-tsat's are not messing things up e.g.
    # when looking at infall time minus sat time
    if "sat:time" in bincol:
        mask = mask & (subs["history:sat:time"] < 12)
    if not has_iterable_bins:
        if logbins:
            j = mask & (bindata > 0)
            ic(bindata.min(), bindata.max(), j.sum())
            vmin, vmax = np.log10(np.percentile(bindata[j], [1, 99]))
            bins = np.logspace(vmin, vmax, bins)
        else:
            vmin, vmax = np.percentile(bindata[mask], [1, 99])
            bins = np.linspace(vmin, vmax, bins)
        ic(vmin, vmax)
        ic(bins)
    if not binlabel:
        binlabel = f"{statistic}({bincol})"
    if logbins:
        lb = np.log10(bins)
        bin_centers = 10 ** ((lb[1:] + lb[:-1]) / 2)
    else:
        bin_centers = (bins[1:] + bins[:-1]) / 2
    ic(bin_centers)
    if has_iterable_bins:
        vmin = bin_centers[0]
        vmax = bin_centers[-1]
    # colors, colormap = colorscale(array=bin_centers, log=logbins)
    normfunc = mplcolors.LogNorm if logbins else mplcolors.Normalize
    colormap = cm.ScalarMappable(
        normfunc(vmin=bin_centers[0], vmax=bin_centers[-1]), cmap
    )
    colors = colormap.to_rgba(bin_centers)
    return mask, bins, bin_centers, bindata, binlabel, vmin, vmax, colormap, colors


def do_xbins(X, mask, xbins, xlim=None, xscale="log"):
    ic()
    ic(xlim)
    if xlim is not None:
        X = X[(xlim[0] <= X) & (X <= xlim[1])]
    mask = mask & np.isfinite(X)
    ic(X.shape, mask.shape, mask.sum())
    if isinstance(xbins, int):
        # ic(X[mask], X[mask].shape)
        # ic(X[mask].min(), X[mask].max(), np.min(X[mask]), np.max(X[mask]),
        # X.loc[mask].min(), X.loc[mask].max())
        if xscale == "log":
            mask = mask & (X > 0)
            xbins = np.linspace(
                np.log10(np.min(X[mask])), np.log10(np.max(X[mask])), xbins + 1
            )
        else:
            xbins = np.linspace(X[mask].min(), X[mask].max(), xbins + 1)
        if xscale == "log":
            xbins = 10**xbins
    if xscale == "log":
        xb = np.log10(xbins)
        xcenters = 10 ** ((xb[:-1] + xb[1:]) / 2)
    else:
        xcenters = (xbins[:-1] + xbins[1:]) / 2
    return xbins, xcenters


def plot_segregation_literature(ax, xcol, ycol):
    ## Sifón+18
    # Rsat (Mpc) - missing normalization
    xlit = np.array([0.23, 0.52, 0.90, 1.55])
    if "/R200Mean" in xcol:
        xlit /= 2.36
    logylit = [10.49, 11.60, 11.55, 11.46]
    ylit, ylitlo = to_linear(logylit, [0.47, 0.15, 0.21, 0.33], which="lower")
    ylit, ylithi = to_linear(logylit, [0.35, 0.16, 0.21, 0.25], which="upper")
    if ycol == "Mbound/Mstar":
        mstarlit = 10 ** np.array([9.97, 10.03, 10.07, 10.24])
        ylit /= mstarlit
        ylitlo /= mstarlit
        ylithi /= mstarlit
    ax.errorbar(
        xlit,
        ylit,
        (ylitlo, ylithi),
        fmt="o",
        ms=10,
        elinewidth=3,
        color="k",
        zorder=100,
        # label=r'Sifón+18 ($\log\langle' \
        #     ' m_{\u2605}/\mathrm{M}_\odot' \
        #     r' \rangle=10.1$)')
        label=r"Sifón+18 (1.3)",
    )
    # Kumar+22
    xlit = np.array([0.2, 0.38, 0.58, 0.78])
    if "/R200Mean" in xcol:
        mcl_kumar = 10 ** np.array([14.31, 14.33, 14.36, 14.39])
        r200m_kumar = rsph(mcl_kumar, 0.26, ref="200m")
        xlit /= r200m_kumar
    logylit = [11.86, 12.17, 12.05, 12.11]
    ylit, ylitlo = to_linear(logylit, [0.20, 0.06, 0.06, 0.05], which="lower")
    ylit, ylithi = to_linear(logylit, [0.16, 0.05, 0.06, 0.05], which="upper")
    if ycol == "Mbound/Mstar":
        mstarlit = 10 ** np.array([10.48, 10.46, 10.50, 10.51])
        ylit /= mstarlit
        ylitlo /= mstarlit
        ylithi /= mstarlit
    ax.errorbar(
        xlit,
        ylit,
        (ylitlo, ylithi),
        fmt="s",
        ms=8,
        elinewidth=3,
        color="C2",
        mfc="w",
        mew=3,
        zorder=100,
        # label=r'Kumar+22 ($\log\langle' \
        #     ' m_{\u2605}/\mathrm{M}_\odot' \
        #     r' \rangle=10.5$)')
        label=r"Kumar+22 (3.2)",
    )
    # Wang+23
    xlit = np.array([0.13, 0.26, 0.44, 0.59, 0.71, 0.89])
    if "/R200Mean" in xcol:
        mcl_wang = 5e14
        r200m_wang = rsph(mcl_wang, 0.3, ref="200m")
        xlit /= r200m_wang
    logylit = np.array([11.38, 11.94, 12.25, 12.47, 12.63, 12.72])
    ylit, ylitlo = to_linear(
        logylit, [0.11, 0.07, 0.10, 0.14, 0.12, 0.20], which="lower"
    )
    ylit, ylithi = to_linear(
        logylit, [0.09, 0.07, 0.09, 0.12, 0.11, 0.17], which="upper"
    )
    mstarlit = 10 ** np.array([10.69, 10.71, 10.77, 10.79, 10.81, 10.83])
    # mean stellar mass
    nmstarlit = np.array([82501, 90250, 41047, 8071, 7997, 3191])
    mstarlit_mean = np.sum(mstarlit * nmstarlit) / nmstarlit.sum() / 1e10
    if ycol == "Mbound/Mstar":
        ylit /= mstarlit
        ylitlo /= mstarlit
        ylithi /= mstarlit
    ic(xlit, ylit, ylitlo, ylithi)
    ax.errorbar(
        xlit,
        ylit,
        (ylitlo, ylithi),
        fmt="^",
        ms=8,
        elinewidth=3,
        color="C0",
        mfc="w",
        mew=3,
        zorder=100,
        label=f"Wang+23 ({mstarlit_mean:.1f})",
    )
    return


def relation_centrals(
    ax,
    subs,
    xcol,
    ycol,
    bincol,
    xbins,
    xcenters,
    bins,
    statistic,
    centrals_label="_none_",
    color="k",
    selection=None,
    selection_min=None,
    selection_max=None,
):
    def get_relation_centrals(subs, mask):
        yc = "/".join([i.split(":")[-1] for i in ycol.split("/")])
        cydata = subs[yc][mask]
        if statistic == "std":
            cydata = np.log10(cydata)
        if "time" in xcol or "Distance" in xcol or xcol.split(":")[-1] == "z":
            ncen = -np.ones(xcenters.size, dtype=int)
            # cenrel would be the same
            return ncen, ncen
        else:
            xc = "/".join([i.split(":")[-1] for i in xcol.split("/")])
            cxdata = subs[xc][mask]
            ncen = np.histogram(cxdata, xbins)[0]
        # in this case just show overall mean
        if (
            "time" in xcol
            or ("Distance" in xcol and "/" in ycol)
            or xcol.split(":")[-1] == "z"
        ):
            if statistic == "mean":
                c0 = np.mean(cydata)
            elif statistic == "std":
                c0 = np.std(cydata)
            else:
                c0 = np.std(cydata) / np.mean(cydata)
            cenrel = c0 * np.ones(xcenters.size)
        elif "/" in statistic:
            st = statistic.split("/")
            cenrel = (
                binstat(cxdata, cydata, st[0], xbins)[0]
                / binstat(cxdata, cydata, st[1], xbins)[0]
            )
        else:
            st = count_stat if statistic == "count" else statistic
            try:
                st = getattr(np, f"nan{st}")
            except AttributeError:
                pass
            cenrel = binstat(cxdata, cydata, st, xbins)[0]
        cenrel[cenrel == 0] = np.nan
        return cenrel, ncen

    jcen = subs["Rank"] == 0
    # shouldn't we ignore selection for centrals?
    if selection is not None:
        if selection_min is not None:
            jcen = jcen & (subs[selection] >= selection_min)
        if selection_max is not None:
            jcen = jcen & (subs[selection <= selection_max])
    #
    if bincol is not None and "M200Mean" not in bincol:
        if "Distance" not in bincol and "time" not in bincol:
            bc = "/".join([i.split(":")[-1] for i in bincol.split("/")])
            if bc != "z":
                jcen = jcen & (subs[bc] >= bins[0]) & (subs[bc] <= bins[-1])
    if jcen.sum() > 0:
        # specifically historical mbound/mstar vs mstar binned by time
        if (
            "history" in xcol
            and "Mstar" in xcol
            and bincol.count("time") == 1
            and "Mbound" in ycol
            and "Mstar" in ycol
            and ycol.count("history") == 2
        ):
            f = os.path.join(subs.sim.data_path, f"chsmr_{statistic}.txt")
            cen_file = os.path.join(subs.sim.data_path, f"chsmr_{statistic}.txt")
            cen_time = np.loadtxt(cen_file)
            for tref, ls in zip((0, 5, 10), ("o--", "s-.", "^:")):
                # load stellar masss bins
                with open(cen_file) as cf:
                    xcen = np.array(
                        [cf.readline() for i in range(3)][-1][2:].split(","),
                        dtype=float,
                    )
                xcen = (xcen[:-1] + xcen[1:]) / 2
                jcentime = np.argmin(np.abs(cen_time[:, 1] - tref))
                ic(tref, jcentime)
                ycentime = cen_time[jcentime, 3:]
                ycentime[ycentime == 0] = np.nan
                ic(ycentime)
                jjcen = xcen <= xcenters[-1]
                # plot_line(
                #     ax, xcen[jjcen], ycentime[jjcen], ls=ls, lw=2,
                #     color='C3', ms=4, label=f'Centrals {tref} Gyr ago')
        # else:
        cenrel, ncen = get_relation_centrals(subs, jcen)
        ls = "--" if "Distance" in xcol else "o--"
        plot_line(
            ax,
            xcenters,
            cenrel,
            ls="o--",
            lw=2,
            color=color,
            ms=6,
            label=centrals_label,
            zorder=100,
        )
        # subs.distance2massive()
        # #cenfield = jcen & (subs['ComovingMostBoundDistanceToMassive'] > 10)
        # cenrel, ncen = get_relation_centrals(subs, cenfield)
        # plot_line(
        #     ax, xcenters, cenrel, ls='o--', lw=2, color=color, ms=6,
        #     label=centrals_label, zorder=100)
        # show an earlier time too if dealing with historical quantities
        if "Distance" not in xcol and "history" in ycol:
            z_old = np.linspace(0, 2, 1000)
            j_old = np.argmin(
                np.abs(
                    subs.sim.cosmology.lookback_time(z_old)[:, None] - [5, 10] * u.Gyr
                ),
                axis=0,
            )
            z_old = z_old[j_old]
            ic(z_old, subs.sim.cosmology.lookback_time(z_old))
            for zi, lsi, c in zip(z_old, (":", "-."), ("0.4", "0.6")):
                isnap_old = subs.sim.snapshot_index_from_redshift(zi)
                subs_old = Subhalos(
                    subs.reader.LoadSubhalos(isnap_old),
                    subs.sim,
                    isnap_old,
                    logM200Mean_min=None,
                    logMstar_min=8.9,
                    logMmin=None,
                    load_any=False,
                )
                crold, ncold = get_relation_centrals(subs_old, subs_old.central_mask)
                tlb = subs.sim.cosmology.lookback_time(zi).to("Gyr")
                ic(zi, isnap_old, tlb, crold / cenrel)
                plot_line(
                    ax,
                    xcenters,
                    crold,
                    ls=f"o{lsi}",
                    lw=2,
                    color=c,
                    ms=4,
                    label=f"Centrals {tlb.value:.0f} Gya",
                    zorder=50,
                )
                # split past centrals by whether they are still centrals
                # old_today = subs_old.merge(
                #     subs, on='TrackId', how='left', suffixes=('', '_today'),
                #     in_place=False)
                # ic(old_today)
                # cen_old_cen_today = (old_today['Rank'] == 0) \
                #     & (old_today['Rank_today'] == 0)
                # ic(cen_old_cen_today.size, cen_old_cen_today.sum())
                # crold, ncold = get_relation_centrals(old_today, cen_old_cen_today)
                # plot_line(
                #     ax, xcenters, crold, ls=f'x{lsi}', lw=2, color='C6', ms=6,
                #     label=f'Centrals {tlb.value:.0f} Gya, centrals today',
                #     zorder=100)
                # cen_old_sat_today = (old_today['Rank'] == 0) \
                #     & (old_today['Rank_today'] > 0)
                # ic(cen_old_sat_today.size, cen_old_sat_today.sum())
                # crold, ncold = get_relation_centrals(old_today, cen_old_sat_today)
                # plot_line(
                #     ax, xcenters, crold, ls=f'+{lsi}', lw=2, color='C9', ms=8,
                #     label=f'Centrals {tlb.value:.0f} Gya, satellites today',
                #     zorder=100)
                # show field centrals
                if (tlb.value < 6) and False:
                    # I've found a specific snapshot that has these
                    subs_old.host_properties()
                    subs_old.distance2host()
                    subs_old.distance2massive()
                    cen_old_field = subs_old.central_mask & (
                        subs_old["ComovingMostBoundDistanceToMassive"] > 10
                    )
                    cfold, ncfold = get_relation_centrals(subs_old, cen_old_field)
                    plot_line(
                        ax,
                        xcenters,
                        cfold,
                        ls=f"x-",
                        lw=3,
                        color="C9",
                        ms=6,
                        label=f"Field centrals {tlb.value:.0f} Gya",
                        zorder=100,
                    )
                # break
    return cenrel, ncen


def plot_relation_lines(
    xdata,
    ydata,
    xbins,
    xcenters,
    statistic,
    mask,
    bindata,
    bins,
    ax,
    colors,
    ls="-",
    lw=4,
    zorder=1,
    alpha_uncertainties=0.25,
    nmin=2,
):
    relation = relation_lines(
        xdata,
        ydata,
        xbins,
        statistic,
        mask,
        bindata,
        bins,
        return_err=(alpha_uncertainties > 0),
    )
    if alpha_uncertainties:
        relation, (err_lo, err_hi) = relation
        ic(err_lo[0], err_hi[0])
    else:
        err_lo = err_hi = np.ones_like(relation)
    if len(relation.shape) == 1:
        relation = relation[None]
        err_lo = err_lo[None]
        err_hi = err_hi[None]
    ic(xcenters)
    ic(relation, relation.shape)
    for i, (r, c) in enumerate(zip(relation, colors)):
        if np.isnan(r).sum() > r.size / 2:
            continue

        ax.plot(xcenters, r, ls, color=c, lw=lw, zorder=10 + i)
        jmr = np.isfinite(r)
        meanratio = np.nansum((r / relation[0] / err_lo[i])[jmr] ** 2) / np.sum(
            1 / err_lo[i][jmr] ** 2
        )
        ic(i, r / relation[0], meanratio)
        if alpha_uncertainties:
            ax.fill_between(
                xcenters,
                err_lo[i],
                err_hi[i],
                color=c,
                zorder=10 + i - 1,
                alpha=alpha_uncertainties,
                lw=0,
            )
    if alpha_uncertainties:
        return relation, (err_lo, err_hi)
    return relation


def relation_lines(
    x, y, xbins, statistic, mask=None, bindata=None, bins=10, return_err=False
):
    if mask is not None:
        x = x[mask]
        y = y[mask]
        if bindata is not None:
            bindata = bindata[mask]
    ic(type(bindata), statistic)
    ic(xbins, np.nanmin(x), np.nanmax(x))
    # std in dex
    if statistic == "std":
        y = np.log10(y)
    statistic = statistic.split("/")
    if bindata is None or statistic == ["count"]:
        func = binstat
        args = [x, y]
        bin_arg = xbins
    else:
        func = binstat_dd
        args = [[bindata, x], y]
        bin_arg = (bins, xbins)
    ic(func)
    ic(statistic)
    if bindata is not None:
        ic(bindata.shape)
    ic(x.shape, y.shape)
    relation = func(*args, statistic[0], bin_arg).statistic
    if len(statistic) == 2:
        relation = relation / func(*args, statistic[1], bin_arg).statistic
    ic(relation.shape)
    # this will make the plot look nicer
    N = func(*args, "count", bin_arg).statistic
    ic(N.shape)
    relation[N <= 1] = np.nan
    if "std" in statistic:
        relation[relation == 0] = np.nan
    if return_err:
        # this may not work for ratio statistics
        errlo = func(
            *args,
            lambda x: (np.mean(x) - np.percentile(x, 16)) / x.size**0.5,
            bin_arg,
        ).statistic
        errhi = func(
            *args,
            lambda x: (np.percentile(x, 84) - np.mean(x)) / x.size**0.5,
            bin_arg,
        ).statistic
        ic(errlo.shape, errhi.shape)
        if statistic == "std":
            errlo /= 2**0.5
            errhi /= 2**0.5
        return relation, (relation - errlo, errhi + relation)
    return relation


def relation_surface(
    x,
    y,
    xbins,
    ybins,
    statistic,
    mask=None,
    bindata=None,
    bins=10,
    logbins=True,
    cmap="viridis",
):
    ic()
    ic(bindata)
    ic(statistic)
    assert isinstance(logbins, bool)
    if mask is not None:
        x = x[mask]
        y = y[mask]
        if bindata is not None:
            bindata = bindata[mask]
    if bindata is None or statistic == "count":
        relation = np.histogram2d(x, y, (xbins, ybins))[0]
        vmin, vmax = np.percentile(relation, [1, 99])
        colornorm = mplcolors.LogNorm()
    elif "/" in statistic:
        stat = statistic.split("/")
        relation = (
            binstat_dd([x, y], bindata, stat[0], bins=(xbins, ybins)).statistic
            / binstat_dd([x, y], bindata, stat[1], bins=(xbins, ybins)).statistic
        )
    else:
        relation = binstat_dd([x, y], bindata, statistic, bins=(xbins, ybins)).statistic
    # maybe everything below should be a separate function
    if True:
        ic(np.percentile(relation, [0, 1, 50, 99, 100]))
        ic(vmin, vmax)
        colors, cmap = colorscale(
            array=relation, vmin=vmin, vmax=vmax, log=logbins, cmap=cmap
        )
    # except AssertionError as err:
    # ic(xcol, ycol, vmin, vmax)
    # ic(err)
    # sys.exit()
    if bindata is not None:
        # color scale -- still need to set vmin,vmax
        ic(colors)
        ic(relation)
        ic(relation.shape)
        colornorm = mplcolors.Normalize(vmin=vmin, vmax=vmax)
        if "std" in statistic:
            relation[relation == 0] = np.nan
        if with_alpha:
            alpha = np.histogram2d(x, y, bins=(xbins, ybins))[0]
            ic(alpha)
            alpha_log = np.log10(alpha)
            ic(alpha_log.min(), alpha_log.max())
            alpha = alpha_log
            # should go between 0 and 1
            alpha_min = 0.4
            alpha[~np.isfinite(alpha)] = 0
            alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min()) + alpha_min
            alpha[alpha > 1] = 1
            ic(alpha)
            ic(alpha.shape, alpha.min(), alpha.max())
            # to add alpha
            ic(colors.shape)
            colors[:, :, -1] = alpha
            colors = np.transpose(colors, axes=(2, 0, 1))
            colors = colors.reshape((4, alpha.size))
            ic(colors.shape)
            cmap = ListedColormap(colors.T)
    return relation, colors, cmap


def set_yscale_log(axes, show_ratios, ylim_ratios):
    ylim = axes[0].get_ylim()
    ic(ylim)
    # yticks = axes[0].yticks()
    # if len(yticks[0]) == 1:
    if ylim[1] / ylim[0] < 20:
        # axes[0].yaxis.set_major_locator(ticker.LogLocator(subs="all"))
        axes[0].yaxis.set_major_locator(ticker.FixedLocator([20, 50, 100, 200]))
    axes[0].yaxis.set_minor_formatter(ticker.NullFormatter())
    if ylim[0] >= 0.001 and ylim[1] <= 1000:
        if ylim[0] >= 1:
            fmt = "%d"
        elif ylim[0] > 0.1:
            fmt = "%.1f"
        elif ylim[0] >= 0.01:
            fmt = "%.2f"
        else:
            fmt = "%.3f"
        axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
    if show_ratios:
        if ylim_ratios is not None:
            axes[1].set_ylim(ylim_ratios)
        ylim = axes[1].get_ylim()
        ic(ylim)
        ic(ylim[0] > 0 and ylim[1] / ylim[0] >= 50, ylim[0] <= 0 and ylim[1] >= 20)
        if (ylim[0] > 0 and ylim[1] / ylim[0] >= 50) or (
            ylim[0] <= 0 and ylim[1] >= 20
        ):
            axes[1].set_yscale("log")
            axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%s"))
    return


#############################
##                         ##
##      Main function      ##
##                         ##
#############################


def plot_relation(
    sim,
    subs,
    xcol="Mstar",
    ycol="Mbound",
    lines_only=True,
    statistic="mean",
    selection="Mstar",
    selection_min=1e8,
    selection_max=None,
    selection_labels=None,
    force_selection=False,
    xlim=None,
    ylim=None,
    xbins=12,
    xscale="log",
    ybins=12,
    yscale="log",
    hostmass="M200Mean",
    min_hostmass=13,
    show_hist=False,
    bindata=None,
    bincol=None,
    bins=6,
    logbins=False,
    binlabel="",
    mask=None,
    xlabel=None,
    ylabel=None,
    with_alpha=False,
    cmap="inferno",
    cbar_ax=None,
    lw=4,
    colornorm=mplcolors.LogNorm(),
    cmap_range=(0.2, 0.7),
    show_contours=True,
    contour_kwargs={},
    alpha_uncertainties=0.4,
    show_satellites=True,
    show_centrals=False,
    show_satellites_err=False,
    show_centrals_scatter=False,
    satellites_label="All satellites",
    centrals_label="Centrals",
    literature=False,
    show_ratios=False,
    ylim_ratios=None,
    show_1to1=False,
    axes=None,
    output=True,
    store_txt=True,
    verbose=True,
):
    """Plot the relation between two quantities, optionally
    binning by a third

    ``bincol`` and ``bins`` allow the relations to be binned in a
    third quantity

    If ``bincol`` is ``None`` then ``statistic`` is forced to ``count``

    """
    ic(xcol, xscale, xbins)
    ic(ycol, yscale, ybins)
    ic(bincol, logbins, bins)
    ic(selection, selection_min, selection_max)
    # doesn't make much sense otherwise
    if statistic != "mean":
        show_contours = False
    count_stat = np.nanmedian
    statmap = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "std": np.nanstd,
        "std/mean": lambda x: np.nanstd(x) / np.nanmean(x),
        "logmean": lambda x: 10 ** np.nanmean(np.log10(x)),
        "logstd": lambda x: np.nanstd(np.log10(x)),
    }
    cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat = definitions(
        subs, hostmass=hostmass, min_hostmass=min_hostmass
    )
    xdata = subs[xcol]
    ydata = subs[ycol]
    ic(xdata.shape, ydata.shape)
    mask = np.isfinite(xdata) & np.isfinite(ydata)
    if "history:sat:time" in ycol:
        mask = mask & (subs["history:sat:time"] < 12)
    ic(mask.sum())
    cmap = cmr.get_sub_cmap(cmap, *cmap_range)
    # additional binning?
    if bincol is None:
        statistic = "count"
        # this just for easier integration of code in the plotting bit
        bins = np.zeros(1)
        colors = "k"
        with_alpha = False
        lines_only = False
    else:
        (
            mask,
            bins,
            bin_centers,
            bindata,
            binlabel,
            vmin,
            vmax,
            colormap,
            colors,
        ) = apply_bins(subs, mask, bincol, bins, statistic, logbins, binlabel, cmap)
    # mask = mask & (subs[hostmass] >= 10**min_hostmass)
    # these cases should be controlled with xbins rather than selection
    if xcol == selection and not force_selection:
        selection = None
    binned_selection = False
    if selection is not None:
        # if we are generating bins in some "selection" quantity,
        # make sure the definitions are consistent
        if (
            np.iterable(selection_min)
            or np.iterable(selection_max)
            and (len(selection_min) > 1)
        ):
            assert (
                np.iterable(selection_max)
                and np.iterable(selection_max)
                and (len(selection_min) == len(selection_max))
            ), (
                f"selection_min {selection_min} inconsistent with"
                f" selection_max {selection_max}"
            )
            binned_selection = True
        # this should only apply if we're selecting a single bin
        else:
            seldata = subs[selection]
            assert isinstance(
                selection, (str, np.str_)
            ), "``selection`` must be a single column name"
            if selection_min is not None:
                mask = mask & (seldata >= selection_min)
            if selection_max is not None:
                mask = mask & (seldata <= selection_max)
    ic(selection, selection_min, selection_max, binned_selection)
    # ic(mask.sum())
    xdata = xdata[mask]
    ydata = ydata[mask]
    ic(xdata.shape, ydata.shape)
    if bincol is not None:
        bindata = bindata[mask]
    sat = sat[mask]
    dark = dark[mask]
    # cen = cen[mask]
    gsat = sat & ~dark
    ic(xbins)
    # calculate x-binning if necessary
    xbins, xcenters = do_xbins(xdata, gsat, xbins, xscale=xscale)
    logx = np.log10(xcenters)
    ic(xcol, xbins)
    ic(xdata.min(), xdata.max())
    ic(np.histogram(xdata, xbins)[0])
    ic(xcenters)
    if show_contours or not lines_only:
        ybins, ycenters = do_xbins(
            ydata,
            gsat,
            ybins,
            xlim=ylim if statistic == "mean" else None,
            xscale=yscale,
        )
        logy = np.log10(ycenters)
        ic(ycol, ybins)
        ic(ydata.min(), ydata.max())
        ic(np.histogram(ydata, ybins)[0])
        ic(ycenters)
    ic(xlim, ylim)

    if bindata is None:
        mask_ = np.ones(xdata.size, dtype=bool)
    else:
        mask_ = (bindata >= bins[0]) & (bindata <= bins[-1])
    relation_overall = relation_lines(
        xdata[mask_], ydata[mask_], xbins, statistic, gsat[mask_]
    )
    ic(relation_overall)
    # uncertainties on overall relation
    if statistic in ("mean", "std"):
        ic(gsat.shape, mask_.shape)
        relation_overall_std = relation_lines(
            xdata[mask_], ydata[mask_], xbins, "std", gsat[mask_]
        )
        relation_overall_cts = np.histogram(xdata[mask_ & gsat], xbins)[0]
        relation_overall_err = relation_overall_std / relation_overall_cts**0.5
        if statistic == "std":
            relation_overall_err /= 2**0.5
        ic(relation_overall_err)
        # fit a power law just in case
        # try:
        #     pl = lambda x, a, b: a * x**b
        #     plfit, plfitcov = curve_fit(
        #         pl,
        #         xcenters[:-1],
        #         relation_overall[:-1],
        #         sigma=relation_overall_err[:-1],
        #         absolute_sigma=True,
        #         p0=(1.2, 0.8),
        #     )
        #     plfiterr = np.diag(plfitcov) ** 0.5
        #     ic(statistic, plfit, plfitcov, plfiterr)
        # except ValueError as err:
        #     wrn = f"Could not fit relation: {err}"
        #     warnings.warn(wrn, RuntimeWarning)
    # at least for now
    show_ratios = show_ratios * lines_only

    if axes is None:
        if show_ratios:
            fig = plt.figure(figsize=(8, 8), constrained_layout=True)
            gs = GridSpec(
                2,
                1,
                height_ratios=(5, 2),
                hspace=0.05,
                left=0.15,
                right=0.9,
                bottom=0.1,
                top=0.95,
            )
            fig.add_subplot(gs[0])
            fig.add_subplot(gs[1])  # , sharex=fig.axes[0])
            # fig.add_gridspec(2, 1, height_ratios=(5,2))
            axes = fig.axes
        else:
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            axes = [ax]
    elif not hasattr(axes, "__iter__"):
        fig = None
        axes = [axes]
    ax = axes[0]

    if show_contours:
        for key, val in zip(("zorder", "color", "linestyles"), (5, "k", "solid")):
            contour_kwargs[key] = val
        try:
            if "levels" not in contour_kwargs:
                contour_kwargs["levels"] = contour_levels(
                    xdata[gsat], ydata[gsat], 12, (0.1, 0.5, 0.9)
                )
            ax.contour(xdata[gsat], ydata[gsat], contour_kwargs["levels"])
        except:  # which exception was it again?
            pass
        else:
            counts = relation_surface(
                xdata, ydata, xbins, ybins, "count", gsat, logbins=logbins, cmap=cmap
            )
            plt.contour(xcenters, ycenters, counts, **contour_kwargs)

    if lines_only:
        ic(np.percentile(bindata, [1, 50, 99]))
        ic(bins, np.histogram(bindata, bins)[0])
        # just sanity-checking the power-law fit. Comment out later
        # ax.plot(xcenters, relation_overall, 'k-', lw=1)
        # ax.plot(xcenters, pl(xcenters, *plfit), 'k:')
        # ax.annotate(fr'$\alpha={plfit[1]:.2f}\pm{plfiterr[1]:.2f}$',
        #             xy=(0.9,0.1), xycoords='axes fraction', ha='right', va='bottom')
        if binned_selection:
            relation = []
            ls = ["-", "--", "-.", ":"]
            ic(selection)
            if selection_labels is None:
                selection_labels = ["_none_" for i in selection_min]
            # for scol, smin, smax, label, ls_ \
            #         in zip(selection, selection_min, selection_max,
            #                selection_labels, ls):
            seldata = subs[selection]
            alpha_uncertainties = 0.25
            for smin, smax, label, ls_ in zip(
                selection_min, selection_max, selection_labels, ls
            ):
                ic(seldata.shape, gsat.shape, smin, smax)
                mask = gsat & (seldata >= smin) & (seldata < smax)
                if "history:sat:time" in selection:
                    mask = mask & (subs["history:sat:time"] < 12)
                ic(seldata.min(), seldata.max(), smin, smax, mask.sum())
                relation.append(
                    plot_relation_lines(
                        xdata,
                        ydata,
                        xbins,
                        xcenters,
                        statistic,
                        mask,
                        bindata,
                        bins,
                        ax,
                        colors,
                        ls_,
                        lw=3,
                        alpha_uncertainties=alpha_uncertainties,
                    )
                )
                ax.plot([], [], "k", ls=ls_, label=label)
            if alpha_uncertainties:
                err_lo = [r[1][0] for r in relation]
                err_hi = [r[1][1] for r in relation]
                relation = [r[0] for r in relation]
                # so it can be decomposed below in all cases
                relation = relation, (err_lo, err_hi)
            if selection_labels is not None:
                ax.legend(fontsize=16, ncol=1 + (statistic == "std"))
        else:
            relation = plot_relation_lines(
                xdata,
                ydata,
                xbins,
                xcenters,
                statistic,
                gsat,
                bindata,
                bins,
                ax,
                colors,
                lw=3,
                zorder=10,
                # alpha_uncertainties=0.4 + 0.3*(alpha_uncertainties > 0))
                alpha_uncertainties=alpha_uncertainties,
            )
        if alpha_uncertainties:
            relation, (err_lo, err_hi) = relation
    else:
        relation, colors, _ = relation_surface(
            xdata,
            ydata,
            xbins,
            ybins,
            statistic,
            gsat,
            bindata,
            bins,
            logbins=logbins,
            cmap=cmap,
        )
        xgrid, ygrid = np.meshgrid(xbins, ybins)
        ic(cmap.name)
        colormap = ax.pcolormesh(
            xgrid,
            ygrid,
            relation.T,  # cmap=cmap_alpha if with_alpha else cmap)
            cmap=cmap,
            norm=colornorm,
            aa=False,
            rasterized=True,
        )
        # see https://stackoverflow.com/questions/32177718/use-a-variable-to-set-alpha-opacity-in-a-colormap
        # apparently it is necessary to save before adding alpha
        # plt.savefig('tmp.png')
        # for i, a in zip(pcm.get_facecolors(), alpha.flatten()):
        #     i[3] = a

    # compare to overall satellite relation?
    j = mask & gsat
    nsat = binstat(xdata[j], ydata[j], "count", xbins).statistic
    if "/" in statistic:
        st = statistic.split("/")
        satrel = (
            binstat(xdata[j], ydata[j], st[0], xbins)[0]
            / binstat(xdata[j], ydata[j], st[1], xbins)[0]
        )
    else:
        st = count_stat if statistic == "count" else statistic
        y = np.log10(ydata) if st == "std" else ydata
        satrel = binstat(xdata[j], y[j], st, xbins).statistic
    # centrals have not been masked
    # remember we are ignoring selections for centrals now
    if show_centrals:
        cenrel, ncen = relation_centrals(
            ax,
            subs,
            xcol,
            ycol,
            bincol,
            xbins,
            xcenters,
            bins,
            statistic,
            centrals_label=centrals_label,
            color=ccolor,
        )
        ic(cenrel, ncen)
    else:
        cenrel = -np.ones(xcenters.size)
        ncen = np.zeros(xcenters.size)
    if show_satellites:
        # skipping the last element because it's usually very noisy
        # (might have to check for specific combinations)
        s_ = np.s_[:-1]
        plot_line(
            ax,
            xcenters,
            satrel,
            ls="-",
            lw=4,
            color=scolor,
            label=satellites_label,
            zorder=100,
        )
        ic(f"{st} for all satellites:")
        ic(f"{xcol} = {xcenters}")
        ic(f"{ycol} = {satrel}")

    # for the evolution of segregation
    # these are the relations
    if (
        xcol == "ComovingMostBoundDistance/R200MeanComoving"
        and ycol == "Mbound/Mstar"
        and bincol == "history:first_infal:time"
    ):
        y_ = np.array(
            [
                [
                    np.nan,
                    np.nan,
                    85.82513952,
                    77.36847005,
                    89.00092327,
                    106.35368904,
                    129.39137044,
                    72.62425431,
                    58.1202016,
                ],
                [
                    np.nan,
                    15.74054909,
                    38.40288615,
                    33.51268974,
                    41.33518134,
                    70.92508075,
                    83.27832823,
                    61.6354377,
                    np.nan,
                ],
                [
                    2.25297443,
                    19.41838336,
                    24.48618204,
                    18.94995981,
                    35.02521122,
                    50.78103654,
                    71.26916845,
                    92.92957282,
                    np.nan,
                ],
                [
                    3.02426151,
                    10.68869368,
                    9.06117115,
                    16.08333238,
                    24.73777981,
                    40.70680249,
                    56.92432825,
                    60.93744483,
                    np.nan,
                ],
                [
                    3.52177548,
                    4.98019104,
                    8.11739877,
                    11.7225467,
                    20.31574626,
                    28.90067646,
                    51.77081904,
                    np.nan,
                    np.nan,
                ],
            ]
        )
        for yi_, c in zip(y_, colors):
            ax.plot(xcenters, yi_, "--", lw=2.5, color=c)

    # show bottom panel with ratios?
    if lines_only and show_ratios:
        ov = cenrel if show_centrals else relation_overall
        for i, (r, c) in enumerate(zip(relation, colors)):
            axes[1].plot(xcenters, r / ov, "-", color=c, lw=4, zorder=10 + i)
        axes[1].axhline(1, ls="--", color="k", lw=1)
        if alpha_uncertainties:
            ylim_ratio = axes[1].get_ylim()
            # this raises an error: err_lo does not exist
            for i, (lo, hi, c) in enumerate(zip(err_lo, err_hi, colors)):
                axes[1].fill_between(
                    xcenters, lo / ov, hi / ov, color=c, zorder=-10 + i, alpha=0.4, lw=0
                )
            axes[1].set_ylim(ylim_ratio)
        if show_satellites:
            plot_line(
                axes[1], xcenters, satrel / ov, ls="-", lw=4, color=scolor, zorder=100
            )
        axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    ic(bins.shape, colors.shape)
    if lines_only:
        # add discrete colorbar
        try:
            cmap_lines = plt.get_cmap(cmap, bins.size - 1)
        except AttributeError:
            cmap_lines = cmr.take_cmap_colors(
                cmap, bins.size - 1, cmap_range=cmap_range
            )
        boundary_norm = mplcolors.BoundaryNorm(bins, cmap.N)
        sm = cm.ScalarMappable(cmap=cmap_lines, norm=boundary_norm)
        sm.set_array([])
        if cbar_ax is None:
            cbar = plt.colorbar(sm, ax=axes, ticks=bins)
        else:
            cbar = plt.colorbar(sm, cax=cbar_ax, ticks=bins)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        if "time" in bincol:
            cbar.ax.yaxis.set_minor_locator(ticker.NullLocator())
    elif cbar_ax is None:
        cbar = plt.colorbar(colormap, ax=axes)
    else:
        cbar = plt.colorbar(colormap, cax=cbar_ax)
    if statistic == "count":
        cbar.set_label("$N$")
        if relation.max() < 1000:
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    else:
        cbar.set_label(binlabel)
    if logbins:
        cbar.ax.set_yscale("log")
        if bins[0] >= 0.001 and bins[-1] <= 1000:
            if bins[0] >= 1:
                fmt = "%d"
            elif bins[0] > 0.09:
                fmt = "%.1f"
            elif bins[0] >= 0.009:
                fmt = "%.2f"
            else:
                fmt = "%.3f"
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
    if (
        "Distance" in xcol
        and "Mbound" in ycol
        and ("Mstar" in ycol or ycol.count("Mbound") == 2)
        and (xcol.split("/")[0][-1] not in "012")
        and statistic == "mean"
    ):
        t = np.logspace(-0.7, 0, 10)
        ax.plot(t, 15 * t**1, "-", color="0.5", lw=2)
        ax.text(
            0.6,
            7,
            r"$\propto R/R_\mathrm{200m}$",
            va="center",
            ha="center",
            color="0.5",
            rotation=60,
            fontsize=14,
        )

    if literature:
        xcol_split = xcol.split(":")
        if len(xcol_split) <= 3 and xcol_split[-1] == "Mstar":
            xlit = 10 ** np.array([9.51, 10.01, 10.36, 10.67, 11.01])
            # ylit = read_literature('sifon18_mstar', 'Msat_rbg')
        elif "Distance" in xcol and xcol.split("/")[0][-1] in "012" and "Mstar" in ycol:
            plot_segregation_literature(ax, xcol, ycol)
    if literature or show_centrals:
        ax.legend(
            fontsize=16 - 1.6 * (statistic == "std"),
            loc="upper right",
            ncol=1 + (statistic == "std"),
        )
    if xlabel is None:
        # xlabel = r'$\log\,{0}$'.format(sim.masslabel(mtype='stars'))
        xlabel = get_axlabel(xcol, "mean")
    if ylabel is None:
        # ylabel = r'$\log\,{0}$'.format(sim.masslabel(mtype='total'))
        ylabel = get_axlabel(ycol, statistic)
    # cheap hack
    xlabel = xlabel.replace("$$", "$")
    ylabel = ylabel.replace("$$", "$")
    # if xlim is None:
    # xlim = np.transpose([ax.get_xlim() for ax in axes])
    # xlim = (np.min(xlim[0]), np.max(xlim[1]))
    # ic(xlim)
    for ax in axes:
        ax.set(xscale=xscale)
        # ax.set_xlim(xlim)
    axes[0].set(ylabel=ylabel, yscale=yscale)
    if ylim is None:
        if statistic == "std":
            ylim = (0, 1.2)
        elif statistic == "std/mean":
            ylim = (0, 1.5)
    if ylim is not None:
        axes[0].set_ylim(ylim)
    axes[-1].set(xlabel=xlabel)
    if show_ratios:
        axes[0].set(xticklabels=[])
        ylabel = "Sat/Cen" if show_centrals else "Ratios"
        axes[1].set(ylabel=ylabel)
    if "Distance" in xcol and xscale == "log":
        axes[-1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    elif "time" in xcol:
        # for ax in axes:
        #     ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        axes[-1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    if yscale == "log":
        set_yscale_log(axes, show_ratios, ylim_ratios)

    # diagonal line when applicable (we do this after setting the limits)
    if show_1to1:
        xlim = axes[0].get_xlim()
        ylim = axes[0].get_ylim()
        d0 = min([xlim[0], ylim[0]])
        d1 = max([xlim[1], ylim[1]])
        axes[0].plot([d0, d1], [d0, d1], "k--")
        axes[0].set(xlim=xlim, ylim=ylim)

    # add a random number to see whether they update
    # ax.annotate(
    #     str(np.random.randint(1000)), xy=(0.96,0.96), xycoords='axes fraction',
    #     fontsize=14, va='top', ha='right', color='C3')
    # format filename
    if output == False:
        return relation, fig, axes
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1, hspace=0.05)
    ic(axes[0].get_xlim())
    ic(axes[0].get_ylim())
    output = set_output_name(
        xcol,
        ycol,
        bincol,
        statistic,
        show_centrals,
        show_satellites,
        show_ratios,
        selection if binned_selection else None,
    )
    output = save_plot(fig, output, sim, tight=False, verbose=verbose)
    if store_txt:
        txt = output.replace("plots/", "data/").replace(".pdf", ".txt")
        os.makedirs(os.path.split(txt)[0], exist_ok=True)
        # note that this is only the mean relation, not binned by bincol
        np.savetxt(
            txt,
            np.transpose([xcenters, nsat, satrel, ncen, cenrel]),
            fmt="%.5e",
            header=f"{xcol} Nsat {ycol}__sat Ncen {ycol}__cen",
        )
    return relation


def set_output_name(
    xcol,
    ycol,
    bincol,
    statistic,
    show_centrals,
    show_satellites,
    show_ratios,
    binned_selection=None,
    dirname="relations",
):
    xcol = format_colname(xcol)
    ycol = format_colname(ycol)
    statistic = statistic.replace("/", "-over-")
    outcols = f"{xcol}__{ycol}"
    outdir = f"{statistic}__{outcols}"
    if bincol is None:
        outname = outdir
    else:
        bincol = format_colname(bincol)
        outname = f"{outdir}__bin__{bincol}"
    if binned_selection:
        binned_selection = format_colname(binned_selection)
        outname = f"{outname}__sbin__{binned_selection}"
    if show_centrals:
        outname = f"{outname}__withcen"
    if show_satellites:
        outname = f"{outname}__withsat"
    if show_ratios:
        outname = f"{outname}__ratios"
    output = os.path.join(dirname, ycol, outcols, outdir, outname)
    return output
