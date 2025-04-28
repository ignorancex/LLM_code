import cmasher as cmr
from glob import glob
from icecream import ic
from matplotlib import patheffects as pe, pyplot as plt, ticker, rcParams
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter, LogFormatterMathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import multiprocessing as mp
import numpy as np
import os
from scipy.integrate import trapz
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic as binstat, norm
import sys
from time import sleep, time

from plottery.plotutils import colorscale, savefig, update_rcParams

update_rcParams()
rcParams["text.latex.preamble"] += r",\usepackage{color}"

from HBTReader import HBTReader

# local
from hbtpy.hbt_tools import parse_args, save_plot
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos  # , Track
from hbtpy.track import Track
from hbtpy.helpers.plot_definitions import axlabel, xbins

adjust_kwargs = dict(
    left=0.10, right=0.95, bottom=0.05, top=0.98, wspace=0.3, hspace=0.1
)


def main():
    print("Running...")
    args = (
        (
            "--demographics",
            {"action": "store_true"},
        ),
    )
    args = parse_args(args=args)
    sim = Simulation(args.simulation, args.root)
    h = sim.cosmology.h
    # h = 1

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
        logM200Mean_min=9,
        exclude_non_FoF=False,
    )
    # subs.sort(order='Mbound')
    print(f"Loaded subhalos in {(time()-to)/60:.2f} minutes")

    print(
        "In total there are {0} central and {1} satellite subhalos".format(
            subs.centrals.size, subs.satellites.size
        )
    )

    centrals = Subhalos(
        subs.centrals,
        sim,
        -1,
        load_distances=False,
        load_velocities=False,
        load_history=False,
        logM200Mean_min=9,
        exclude_non_FoF=False,
    )
    satellites = Subhalos(
        subs.satellites,
        sim,
        -1,
        load_distances=False,
        load_velocities=False,
        load_history=False,
        logM200Mean_min=13,
        exclude_non_FoF=True,
    )
    # print(np.sort(satellites.colnames))

    # some numbers
    # m = centrals['M200Mean'].values
    # ms = satellites['M200Mean'].values
    # ic(m.size, ms.size)
    # s = (satellites['Mstar'] > 1e9)
    # ic(s.sum())
    # ic(satellites['Mstar'][s].min()/1e9)
    # ic(np.sort(m)[-10:])
    # b = np.array([1e13, 2e13, 5e13, 8e13, 1e14, 2e14, 5e14])
    # ic(b)
    # h = np.histogram(m, b)[0]
    # ic(h)
    # ic(np.cumsum(h[::-1])[::-1])
    # hs = np.histogram(ms[s], b)[0]
    # ic(hs)
    # ic(np.cumsum(hs[::-1])[::-1])
    # return

    # numbers following Niemiec's binning
    xbins = np.arange(9, 12.1, 0.5)
    logm = np.log10(satellites["Mstar"])
    for i in range(1, xbins.size):
        j = (logm > xbins[i - 1]) & (logm <= xbins[i])
        print(f"{xbins[i-1]:5.1f} - {xbins[i]:5.1f}: {j.sum():6d}")
    print(f"Total: {logm.size}")

    # fit HSMR
    func = double_power_niemiec_log
    mstarbins = np.logspace(9, 12, 50)
    msubbins = np.logspace(9, 14.3, 40)

    mstar_fit_min = 1e9
    mstar_fit_max = 2e11
    # not applying mstar_fit_max to centrals
    cen_for_fit = centrals["Mstar"] >= mstar_fit_min
    fit_cent, fitcov_cent = fit_hsmr(
        func,
        centrals,
        "Mstar",
        "Mbound",
        p0=(10.7, 0.6, 0.7, 6.3),
        log=True,
        label="centrals",
        mask=cen_for_fit,
    )
    ic(fit_cent, np.diag(fitcov_cent) ** 0.5)
    sat_for_fit = (
        (satellites["M200Mean"] >= 1e13)
        & (satellites["Mstar"] >= mstar_fit_min)
        & (satellites["Mstar"] <= mstar_fit_max)
    )
    ic(sat_for_fit.sum())
    fit_sat, fitcov_sat = fit_hsmr(
        func,
        satellites,
        "Mstar",
        "Mbound",
        mask=sat_for_fit,
        p0=(10.7, 0.6, 0.7, 6.3),
        log=True,
        label="satellites",
    )
    ic(fit_sat, np.diag(fitcov_sat) ** 0.5)

    logmstar_ref = 10
    ratio_ref = 10 ** (func(logmstar_ref, *fit_cent) - func(logmstar_ref, *fit_sat))
    print(
        f"at logmstar={logmstar_ref}, mcen/msat={ratio_ref:.3f}"
        f" while msat/mcen={1/ratio_ref:.3f}"
    )

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    xgrid, ygrid = np.meshgrid(mstarbins, msubbins)
    # centrals
    # n = np.histogram2d(
    #     centrals['Mstar'], centrals['Mbound'], (mstarbins,msubbins))[0]
    # cmap = cmr.get_sub_cmap('cmr.freeze_r', 0.1, 0.8)
    # im = ax.pcolormesh(
    #     xgrid, ygrid, n.T, cmap=cmap, norm=LogNorm(), rasterized=True)
    # satellites
    n = np.histogram2d(
        satellites["Mstar"], satellites["Mbound"], (mstarbins, msubbins)
    )[0]
    cmap = cmr.get_sub_cmap("cmr.ember_r", 0.1, 0.8)
    im = ax.pcolormesh(xgrid, ygrid, n.T, cmap=cmap, norm=LogNorm(), rasterized=True)
    ycent = 10 ** func(np.log10(mstarbins), *fit_cent)
    bins_in_fit = (mstarbins >= mstar_fit_min) & (mstarbins <= mstar_fit_max)
    ax.plot(
        mstarbins,
        ycent,
        "C4",
        lw=3,
        dashes=(4, 3),
        label="Centrals",
        path_effects=[pe.Stroke(linewidth=4.5, foreground="w"), pe.Normal()],
    )
    ysat = 10 ** func(np.log10(mstarbins), *fit_sat)
    bins_in_fit = (mstarbins >= mstar_fit_min) & (mstarbins <= mstar_fit_max)
    ax.plot(
        mstarbins[bins_in_fit],
        ysat[bins_in_fit],
        "C0-",
        lw=4,
        label="Satellites",
        path_effects=[pe.Stroke(linewidth=5, foreground="w"), pe.Normal()],
    )
    ax.plot(
        mstarbins[~bins_in_fit],
        ysat[~bins_in_fit],
        "C0",
        lw=4,
        dashes=(3, 4),
        path_effects=[pe.Stroke(linewidth=5, foreground="w"), pe.Normal()],
    )
    ax.plot(mstarbins, mstarbins, color="0.5", dashes=(6, 4))
    ax.text(
        8e9,
        5e9,
        "$m_\mathrm{sub}=m_\u2605$",
        va="bottom",
        ha="left",
        rotation=30,
        fontsize=12,
        color="0.5",
    )
    cbar = plt.colorbar(im, ax=ax, label="$N_\mathrm{sat}$")  # , fraction=0.045)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    ## Literature
    plot_sifon(ax, hnorm=True, h=h)
    plot_niemiec(ax, np.log10(mstarbins), h=h)
    # centrals
    jc = centrals["M200Mean"] > 1e13
    ax.scatter(
        centrals["Mstar"][jc],
        centrals["Mbound"][jc],
        c="k",
        s=4,
        marker="*",
        label="Cluster centrals",
        zorder=100,
    )
    for mclmin in (1e13, 5e13, 1e14):
        n = (centrals["M200Mean"] > mclmin).sum()
        print(f"{n} clusters above M200Mean={mclmin/1e14:.1f}e14 Msun")
        ns = (satellites["M200Mean"] > mclmin).sum()
        print(f"hosting {ns} satellites")
    print(f'max cluster mass: {centrals["M200Mean"].max()/1e14:.2f}e14 Msun')
    # scatter
    plot_scatter(
        ax,
        np.log10(satellites["Mstar"][sat_for_fit]),
        np.log10(satellites["Mbound"][sat_for_fit]),
        func,
        fit_sat,
    )
    # finish
    ax.legend(fontsize=14)
    ax.set(
        xlabel=axlabel["Mstar"],
        ylabel=axlabel["Mbound"],
        xscale="log",
        yscale="log",
        xlim=(1e9, 1.3e12),
        ylim=(1e9, 2e14),
    )
    output = "hsmr"
    save_plot(fig, output, sim, tight=False)

    return


def fit_hsmr(
    func,
    subs,
    xcol,
    ycol,
    mask="default",
    p0=None,
    log=True,
    label="",
    bins=12,
    plot=False,
):
    ic(label)
    if isinstance(mask, str) and mask == "default":
        mask = (subs["M200Mean"] > 1e13) & (subs["Mstar"] > 1e8)
    elif mask is None:
        mask = np.ones(subs.shape[0], dtype=bool)
    x = subs[xcol][mask]
    y = subs[ycol][mask]
    if log:
        x = np.log10(x)
        y = np.log10(y)
    if bins:
        yerr = binstat(x, y, "std", bins).statistic
        yerr = yerr / binstat(x, y, "count", bins).statistic ** 0.5
        ymean = binstat(x, y, "mean", bins).statistic
        xmean = binstat(x, x, "mean", bins).statistic
        fit, cov = curve_fit(func, xmean, ymean, sigma=yerr, absolute_sigma=True, p0=p0)
    else:
        fit, cov = curve_fit(func, x, y, p0=p0)
    err = np.diag(cov) ** 0.5
    if plot and p0 is not None:
        x1 = np.linspace(x.min(), x.max(), 100)
        plt.figure()
        plt.plot(x, y, "k,")
        if bins is not None:
            plt.errorbar(xmean, ymean, yerr, fmt="o")
        plt.plot(x1, func(x1, *p0), "--", label="initial")
        plt.plot(x1, func(x1, *fit), "-", label="best-fit")
        plt.legend()
        if label:
            plt.title(label)
        output = "test.png"
        if label:
            output = output.replace(".png", f"_{label}.png")
        plt.savefig(output)
    return fit, cov


def plot_niemiec(ax, logx, h=1):
    y = 10 ** double_power_niemiec_log(logx, 10.22, 0.65, 0.50, 2.38, h=h)
    ax.plot(
        10**logx,
        y,
        lw=2.5,
        color="0.2",
        dashes=(8, 6),
        label="Niemiec+22 (TNG)",
        path_effects=[pe.Stroke(linewidth=4, foreground="w"), pe.Normal()],
    )
    return y


def plot_scatter(ax, logx, logy, func, params):
    inset = ax.inset_axes([0.65, 0.1, 0.25, 0.25])
    mask = np.isfinite(logx) & (np.isfinite(logy))
    logx = logx[mask]
    logy = logy[mask]
    logypredicted = func(logx, *params)
    ydiff = logy - logypredicted
    mean = np.mean(ydiff)
    scatter = np.std(ydiff)
    n, bins = inset.hist(ydiff, "auto", color="C0", alpha=0.5)[:2]
    # lowm = (logx <= 10)
    # inset.hist(ydiff[lowm], bins, color='C1', histtype='step', lw=2)
    # highm = ~lowm & (logx <= 11.3)
    # inset.hist(ydiff[highm], bins, color='C2', histtype='step', lw=2)
    logx0 = (bins[1:] + bins[:-1]) / 2
    area = trapz(n, logx0)
    ic(mean)
    ic(scatter)
    # redefining for the curve
    logx0 = np.linspace(-1.5, 1.5, 200)
    curve = norm.pdf(logx0, mean, scatter)
    curve = area * curve / trapz(curve, logx0)
    inset.plot(logx0, curve, "C0", lw=1.5)
    inset.axvline(0, ls="--", color="k", lw=1)
    inset.tick_params(axis="both", which="both", labelsize=12, width=1.5, length=2)
    inset.tick_params(axis="both", which="major", length=4)
    for axis in ["top", "bottom", "left", "right"]:
        inset.spines[axis].set_linewidth(1.5)
    inset.set_yticklabels([])
    inset.set_xlabel(
        r"$\log(m_\mathrm{sub})-\log\langle m_\mathrm{sub}|"
        + "m_{\u2605}"
        + r"\rangle$",
        fontsize=12,
    )
    inset.set_xlim(-1.5, 1.5)
    ic(scatter)
    inset.set_title(rf"$\sigma={scatter:.2f}$", fontsize=14)
    return inset


def plot_sifon(ax, hnorm=True, h=1):
    x = 10 ** np.array([9.51, 10.01, 10.36, 10.67, 11.01])
    logy = [10.64, 11.41, 11.71, 11.84, 12.15]
    y, ylo = to_linear(logy, [0.53, 0.21, 0.17, 0.15, 0.20], which="lower")
    y, yhi = to_linear(logy, [0.39, 0.17, 0.15, 0.15, 0.17], which="upper")
    _, ylo_w = to_linear(
        logy, np.array([0.53, 0.21, 0.17, 0.15, 0.20]) + 0.03, which="lower"
    )
    _, yhi_w = to_linear(
        logy, np.array([0.39, 0.17, 0.15, 0.15, 0.17]) + 0.03, which="upper"
    )
    # use h=0.7 to convert to units of Msun/h?
    if hnorm:
        y, ylo, yhi, ylo_w, yhi_w = [i * 0.7 / h for i in (y, ylo, yhi, ylo_w, yhi_w)]
    ax.errorbar(x, y, (ylo_w, yhi_w), fmt="wo", ms=12, elinewidth=4, zorder=10)
    ax.errorbar(
        x, y, (ylo, yhi), fmt="ko", ms=10, elinewidth=2.5, label="Sifón+18", zorder=20
    )
    return y, np.array([ylo, yhi])


## --------------------------------------------------
## fitting functions
## --------------------------------------------------


def double_power_niemiec(mstar, m1, beta, gamma, N, h=1):
    x = mstar / m1
    return 2 * N * (x**-beta + x**gamma) * (mstar / h)


def double_power_niemiec_log(logmstar, logm1, beta, gamma, N, h=1):
    """Using eq. 3 in Niemiec+22"""
    x = 10 ** (logmstar - logm1)
    logy = np.log10(x**-beta + x**gamma) + (logmstar - np.log10(h))
    return np.log10(2 * N) + logy


## --------------------------------------------------
## taken from lnr
## --------------------------------------------------


def to_linear(logx, logxerr=[], base=10, which="average"):
    """
    Take log measurements and uncertainties and convert to linear
    values.
    Parameters
    ----------
    logx : array of floats
        logarithm of measurements to be linearized
    Optional Parameters
    -------------------
    logxerr : array of floats
        uncertainties on logx
    base : float
        base with which the logarithms have been calculated
    which : {'lower', 'upper', 'both', 'average'}
        Which uncertainty to report; note that when converting to/from
        linear and logarithmic spaces, errorbar symmetry is not
        preserved. The following are the available options:
            if which=='lower': xerr = logx - base**(logx-logxerr)
            if which=='upper': xerr = base**(logx+logxerr) - logx
        If `which=='both'` then both values are returned, and if
        `which=='average'`, then the average of the two is returned.
        Default is 'average'.
    Returns
    -------
    x : array of floats
        values in linear space, i.e., base**logx
    xerr : array of floats
        uncertainties, as discussed above
    """
    if np.iterable(logx):
        return_scalar = False
    else:
        return_scalar = True
        logx = [logx]
    logx = np.array(logx)
    if not np.iterable(logxerr):
        logxerr = [logxerr]
    if len(logxerr) == 0:
        logxerr = np.zeros(logx.shape)
    else:
        logxerr = np.array(logxerr)
    assert logx.shape == logxerr.shape, "The shape of logx and logxerr must be the same"
    assert which in ("lower", "upper", "both", "average"), (
        "Valid values for optional argument `which` are 'lower', 'upper',"
        " 'average' or 'both'."
    )
    x = base**logx
    lo = x - base ** (logx - logxerr)
    hi = base ** (logx + logxerr) - x
    if return_scalar:
        x = x[0]
        lo = lo[0]
        hi = hi[0]
    if which == "both":
        return x, lo, hi
    if which == "lower":
        xerr = lo
    elif which == "upper":
        xerr = hi
    else:
        xerr = 0.5 * (lo + hi)
    return x, xerr


def to_log(x, xerr=[], base=10, which="average"):
    """
    Take linear measurements and uncertainties and transform to log
    values.
    Parameters
    ----------
    x : array of floats
        measurements of which to take logarithms
    Optional Parameters
    -------------------
    xerr : array of floats
        uncertainties on x
    base : float
        base with which the logarithms should be calculated. FOR NOW USE
        ONLY 10.
    which : {'lower', 'upper', 'both', 'average'}
        Which uncertainty to report; note that when converting to/from
        linear and logarithmic spaces, errorbar symmetry is not
        preserved. The following are the available options:
            if which=='lower': logxerr = logx - log(x-xerr)
            if which=='upper': logxerr = log(x+xerr) - logx
        If `which=='both'` then both values are returned, and if
        `which=='average'`, then the average of the two is returned.
        Default is 'average'.
    Returns
    -------
    logx : array of floats
        values in log space, i.e., base**logx
    logxerr : array of floats
        log-uncertainties, as discussed above
    """
    assert (
        np.issubdtype(type(base), np.floating)
        or np.issubdtype(type(base), np.integer)
        or base == "e"
    )
    if np.iterable(x):
        return_scalar = False
    else:
        return_scalar = True
        x = [x]
    x = np.array(x)
    if not np.iterable(xerr):
        xerr = [xerr]
    if len(xerr) == 0:
        xerr = np.zeros(x.shape)
    else:
        xerr = np.array(xerr)
    assert xerr.shape == x.shape, "The shape of x and xerr must be the same"
    assert which in ("lower", "upper", "both", "average"), (
        "Valid values for optional argument `which` are 'lower', 'upper',"
        " 'average' or 'both'."
    )

    if base == 10:
        f = lambda y: np.log10(y)
    elif base in (np.e, "e"):
        f = lambda y: np.log(y)
    else:
        f = lambda y: np.log(y) / np.log(base)
    logx = f(x)
    logxlo = logx - f(x - xerr)
    logxhi = f(x + xerr) - logx
    if return_scalar:
        logx = logx[0]
        logxlo = logxlo[0]
        logxhi = logxhi[0]
    if which == "both":
        return logx, logxlo, logxhi
    if which == "lower":
        logxerr = logxlo
    elif which == "upper":
        logxerr = logxhi
    else:
        logxerr = 0.5 * (logxlo + logxhi)
    return (logx,)


if __name__ == "__main__":
    main()
