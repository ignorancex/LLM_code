import cmasher as cmr
from icecream import ic
from itertools import count
from matplotlib import cm, colors, pyplot as plt, ticker
import multiprocessing as mp
import numpy as np
import os
from scipy.stats import binned_statistic_2d as binstat2d, gaussian_kde
import seaborn as sns
from time import time
from tqdm import tqdm

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.helpers.plot_auxiliaries import format_filename, get_axlabel
from hbtpy.helpers.plot_definitions import xbins
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos
from hbtpy.track import Track

from plottery.plotutils import savefig, update_rcParams

update_rcParams()

try:
    sns.set_color_palette("flare")
except AttributeError:
    pass


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation, args.root)
    reader = HBTReader(sim.path)

    isnap = -1
    kwargs = dict(logMmin=None, logM200Mean_min=9, logMstar_min=9)
    subs = Subhalos(reader.LoadSubhalos(isnap), sim, isnap, **kwargs)
    clusters = subs.centrals

    for mmin in np.logspace(11, 14.5, 8):
        cl = subs.central_mask & (subs["M200Mean"] >= mmin)
        sat = subs.satellite_mask & (subs["M200Mean"] >= mmin)
        print(
            f"{cl.sum():4d} clusters with log M200m >= {np.log10(mmin):.1f}"
            f" hosting {sat.sum():5d} satellites"
        )
    n = 5
    print(f"The {n} most massive are:")
    for j in np.argsort(clusters["M200Mean"])[-n:]:
        print(clusters[["TrackId", "HostHaloId", "M200Mean", "Nsat"]].iloc[j])

    subs = Subhalos(
        reader.LoadSubhalos(isnap),
        sim,
        isnap,
        logMmin=0,
        logMstar_min=9,
        logM200Mean_min=13,
    )
    early = subs.satellite_mask & (subs["history:sat:time"] > 12)
    unique_early_hosts, n_early = np.unique(
        subs["HostHaloId"][early], return_counts=True
    )
    n_sat, m200, mcen = np.zeros((3, unique_early_hosts.size))
    for i, (h, c) in enumerate(zip(unique_early_hosts, n_early)):
        allsats = subs.satellite_mask & (subs["HostHaloId"] == h)
        central = subs.central_mask & (subs["HostHaloId"] == h)
        n_sat[i] = allsats.sum()
        m200[i] = subs["M200Mean"][allsats].values[0]
        mcen[i] = subs["Mbound"][central]

    colorbar = "Mcen"
    if colorbar == "Mcen":
        c = mcen
        clabel = "$m_\mathrm{sub}^\mathrm{cen}$"
    else:
        c = m200
        clabel = "$M_\mathrm{200m}$ (M$_\odot$)"
    fig, ax = plt.subplots()
    c = ax.scatter(n_sat, n_early / n_sat, c=c, norm=colors.LogNorm())
    plt.colorbar(c, ax=ax, label=clabel)
    ax.set(xlabel="N satellites", ylabel="Fraction of early sats", xscale="log")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    output = os.path.join(sim.plot_path, "tsat/early_fraction.png")
    savefig(output, fig=fig)
    return


main()
