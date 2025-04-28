from matplotlib import pyplot as plt, ticker
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from time import time

import plottery
from plottery.plotutils import colorscale, savefig, update_rcParams

update_rcParams()

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.hbt_tools import save_plot, timer
from hbtpy.hbtplots import PlotterHelper

# from hbtpy.hbtplots.core import ColumnLabel
from hbtpy.simulation import Simulation
from hbtpy.subhalo import HostHalos, Subhalos


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation, args.root)

    to = time()
    reader = HBTReader(sim.path)
    print(f"Loaded reader in {time()-to:.1f} s")

    isnap = -1
    subs = reader.LoadSubhalos(isnap)
    print("Loaded subhalos!")

    subs = Subhalos(
        subs, sim, isnap, exclude_non_FoF=True, logMmin=8, logM200Mean_min=13
    )
    subs.catalog["Mbound/Mstar"] = subs["Mbound"] / subs["Mstar"]
    events = ("first_infall", "last_infall", "sat", "cent")

    do_correlations(subs, events)

    return


def do_correlations(subs, events):
    cols0 = ["Mstar", "Mbound", "Mbound/Mstar"]
    for event in events:
        h = f"history:{event}"
        to = time()
        cols = (
            cols0
            + [
                col
                for col in subs.catalog.columns
                if (
                    col.startswith(h)
                    and ("time" in col or "Mbound" in col or "Mstar" in col)
                )
            ]
            + [f"{h}:Mbound/{h}:Mstar", f"Mbound/{h}:Mbound", f"Mstar/{h}:Mbound"]
        )
        correlations(subs, cols=cols, suffix=f"_{event}", separators=[2])
        print(f"Correlations in {time()-to:.1f} s")

    ## times
    cols = cols0 + [col for col in subs.catalog.columns if col[-4:] == "time"]
    correlations(
        subs, cols=cols, suffix="_times", ha_ticklabels="center", separators=[2]
    )

    ## historical mass ratios
    cols = ["Mbound/Mstar"]
    for event in events:
        h = f"history:{event}"
        cols = cols + [f"{h}:Mbound/{h}:Mstar"]
    correlations(subs, cols=cols, suffix="_massratios", separators=[0])

    ## historical masses
    for m in ("Mstar", "Mbound"):
        cols = [m]
        for event in events:
            h = f"history:{event}"
            cols = cols + [f"{h}:{m}"]
        correlations(subs, cols=cols, suffix=f"_{m.lower()}", separators=[0])

    ## both masses
    cols = ["Mstar", "Mbound"]
    for event in events:
        h = f"history:{event}"
        cols = cols + [f"{h}:Mstar", f"{h}:Mbound"]
    correlations(subs, cols=cols, suffix=f"_masses", separators=range(len(cols), 2))

    return


def correlations(
    subs, cols=None, Mstar_min=1e8, suffix="", separators=None, ha_ticklabels="right"
):
    if cols is None:
        data = subs.catalog
    else:
        data = subs[cols]
    if Mstar_min > 0:
        data = data.loc[subs["Mstar"] > Mstar_min]
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    for i, c_i in enumerate(corr.values):
        for j, c_ij in enumerate(c_i):
            if i == j:
                continue
            if abs(c_ij) > 0.5:
                ax.annotate(
                    f"{c_ij:.2f}",
                    xy=(i, j),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=15,
                )
    if separators is not None:
        for sep in separators:
            ax.axhline(sep + 0.5, ls="-", color="k", lw=1)
            ax.axvline(sep + 0.5, ls="-", color="k", lw=1)
    # format labels
    ph = PlotterHelper(subs.sim)
    axlabels = [ph.axlabel(col, with_units=False) for col in cols]
    ticks = 0.5 + np.arange(len(cols))
    ax.set(xticks=ticks, yticks=ticks)
    ax.set_yticklabels(axlabels, rotation=45, ha=ha_ticklabels)
    ax.set_xticklabels(axlabels, rotation=45, ha=ha_ticklabels)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    cbar = plt.colorbar(im, ax=ax, label="Pearson's correlation coefficient")
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    output = os.path.join("correlations", f"corr{suffix}")
    save_plot(fig, output, subs.sim)
    return


main()
