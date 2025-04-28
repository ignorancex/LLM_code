"""
Auxiliary plotting functions (a bit disorganized)
"""
from icecream import ic
from matplotlib import patheffects as pe
import numpy as np

from .plot_definitions import massnames, units, xbins, binlabel, axlabel


def binning(bins=None, sim=None, mtype="total", n=None, xmin=0, xmax=1, log=True):
    if bins is None:
        if sim is None:
            f = np.logspace if log else np.linspace
            bins = f(xmin, xmax, n + 1)
        else:
            bins = massbins(sim, mtype=mtype)
            if n is not None:
                bins = np.log10(bins[:: bins.size - 1])
                bins = np.logspace(bins[0], bins[1], n + 1)
    if log:
        x = logcenters(bins)
    else:
        x = (bins[:-1] + bins[1:]) / 2
        xlim = np.log10(x[:: x.size - 1])
    return bins, x, xlim
    """
    mbins = massbins(sim, mtype='total')
    m = logcenters(mbins)
    msbins = massbins(sim, mtype='stars')
    ms = logcenters(msbins)
    mlim = np.log10(m[::m.size-1])
    mslim = np.log10(ms[::ms.size-1])
    return mbins, m, mlim, msbins, ms, mslim
    """


def definitions(subs, hostmass="M200Mean", min_hostmass=13, as_dict=False):
    """Subhalo quantities

    Parameters
    ----------
    subs : ``Subhalos`` object
        subhalos
    hostmass : str, optional
        host mass definition
    as_dict : bool, optional
        whether to return a tuple (if ``False``) or a dictionary

    Returns either a dictionary or tuple with the following content:
     * cen : centrals mask
     * sat : satellites mask
     * mtot : total subhalo mass
     * mstar : stellar mass
     * mhost : Host mass as defined in the ``hostmass`` argument
     * dark : dark subhalos mask
     * Nsat : number of satellite subhalos associated with the same host
     * Ndark : number of dark satellite subhalos
     * Ngsat : number of satellites galaxies
    """
    if min_hostmass is None:
        min_hostmass = 0
    if min_hostmass < 20:
        min_hostmass = 10**min_hostmass
    mtot = subs.mass("total")
    mstar = subs.mass("stars")
    mhost = subs[hostmass]
    cen = subs["Rank"] == 0
    sat = (subs["Rank"] > 0) & (mhost > min_hostmass)
    dark = subs["IsDark"] == 1
    Nsat = subs["Nsat"]
    Ndark = subs["Ndark"]
    Ngsat = Nsat - Ndark
    if as_dict:
        return dict(
            cen=cen,
            sat=sat,
            mtot=mtot,
            mstar=mstar,
            mhost=mhost,
            dark=dark,
            Nsat=Nsat,
            Ndark=Ndark,
            Ngsat=Ngsat,
        )
    return cen, sat, mtot, mstar, mhost, dark, Nsat, Ndark, Ngsat


def format_filename(name):
    name = name.replace("-", "-minus-").replace("/", "-over-")
    name = name.replace(":", "-")
    return name


def get_axlabel(col, statistic="mean"):
    for op in ("/", "-"):
        if op in col:
            cols = col.split(op)
            label = [binlabel[col] if col in binlabel else col for col in cols]
            label = op.join(label)
            break
    else:
        if col in axlabel:
            label = axlabel[col]
        elif col in binlabel:
            label = binlabel[col]
        else:
            label = col
    if statistic in ("count", "mean"):
        return f"${label}$".replace("$$", "$")
    label = label.replace("$", "")
    label = label.split()
    if len(label) == 2:
        unit = label[1]
    label = label[0]
    if statistic == "std":
        lab = rf"$\sigma({label})$ (dex)"
        # because sigma is in dex the argument is unitless
        # if len(label) == 2:
        #     lab = fr'{lab} $({unit})$'
    elif statistic == "std/mean":
        lab = rf"$\sigma({label})/\langle {label} \rangle$"
    return lab


def get_bins(bincol, logbins=True, n=5):
    if bincol in xbins:
        bins = xbins[bincol]
        if n is None:
            return bins
        if logbins:
            if bincol.split(":")[-1] == "z":
                bins[0] = 0.01
            bins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), n + 1)
        else:
            bins = np.linspace(bins[0], bins[-1], n + 1)
    else:
        bins = n + 1
    return bins


def get_label_bincol(bincol):
    # let's assume this refers to a 2d histogram
    if bincol is None:
        return "$N$"
    label = f"{bincol}"
    for key, val in binlabel.items():
        if key[:7] == "history":
            label = label.replace(key, val)
    # now for non-history quantities
    for key, val in binlabel.items():
        label = label.replace(key, val).replace("$", "")
    # fix projected distances
    if "Distance" in bincol:
        label = label.replace("Comoving", "")
        label = label.split("/")
        ic(label)
        if label[0][-1] in "0123":
            p = "p" + label[0][-1].replace("0", "")
            label[0] = f"{label[0][:-1]}_\mathrm{{{p}}}"
        ic(label)
        label = "/".join(label)
    ic(bincol, label)
    if "/" not in bincol:
        if np.any([i in bincol for i in massnames]):
            unit = units["mass"]
        elif bincol[-4:] == "time":
            if bincol.count("time") == 2:
                unit = "\mathrm{Gyr}"
            else:
                unit = units["time"]
        elif "Distance" in bincol:
            unit = units["distance"]
        else:
            unit = None
        if unit:
            label = rf"{label}\,({unit})"
    return f"${label}$"


def logcenters(bins):
    logbins = np.log10(bins)
    return 10 ** ((logbins[:-1] + logbins[1:]) / 2)


def massbins(sim, mtype="stars"):
    if mtype in ("dm", "bound"):
        mtype = "total"
    bins = {
        "apostle": {"stars": np.logspace(3, 11, 21)},
        "eagle": {"stars": np.logspace(6, 12.5, 26), "total": np.logspace(8, 15, 31)},
    }
    return bins[sim.family][mtype]


def plot_line(ax, *args, ls="-", **kwargs):
    kwargs_bg = kwargs.copy()
    # if dashes is not None:
    # kwargs['dashes'] = dashes
    # the offsets probably depend on the size of the dashes...
    # kwargs_bg['dashes'] = (dashes[0]-2.2,dashes[1]-3.8)
    if "color" in kwargs:
        kwargs_bg.pop("color")
    if "label" in kwargs:
        kwargs_bg.pop("label")
    # kwargs_bg['ls'] = ls
    # if 'dashes' in kwargs or ls in ('-', '-.', ':', '--'):
    if "lw" not in kwargs:
        kwargs["lw"] = 4
    kwargs_bg["lw"] = kwargs["lw"] + 2
    # in case we're showing points, not lines
    if "ms" not in kwargs:
        kwargs["ms"] = 8
    if "mew" not in kwargs:
        kwargs["mew"] = 3
    kwargs_bg["ms"] = kwargs["ms"] + 2
    kwargs_bg["mew"] = kwargs["mew"] + 2
    path_effects = [pe.Stroke(linewidth=kwargs["lw"] + 2, foreground="w"), pe.Normal()]
    ax.plot(*args, ls, path_effects=path_effects, **kwargs)
    return
