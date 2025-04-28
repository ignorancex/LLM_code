from icecream import ic
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import binned_statistic as binstat, binned_statistic_dd as binstat_dd

from HBTReader import HBTReader
from hbtpy import hbt_tools
from hbtpy.helpers.plot_auxiliaries import get_bins, logcenters
from hbtpy.simulation import Simulation
from hbtpy.subhalo import Subhalos
from hbtpy.track import Track


def main():
    args = hbt_tools.parse_args()
    sim = Simulation(args.simulation, args.root)
    reader = HBTReader(sim.path)

    isnap = -1
    subs = Subhalos(
        reader.LoadSubhalos(isnap),
        sim,
        isnap,
        logM200Mean_min=13,
        verbose_when_loading=False,
    )

    distribution(args, subs, ("Mbound", "Mstar"), ["time"])

    return


def distribution(
    args, subs, cols, bin_cols, bin_when="first_infall", avg_when="first_infall"
):
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(bin_cols, str):
        bin_cols = [bin_cols]
    bins = [get_bins(i, logbins=("time" not in i), n=10) for i in bin_cols]
    ic(bins)
    events = ("cent", "sat", "first_infall", "last_infall")
    if bin_when in events:
        bin_cols = [f"history:{bin_when}:{col}" for col in bin_cols]
    if avg_when in events:
        cols = [f"history:{avg_when}:{col}" for col in cols]
    binfunc = binstat if len(bin_cols) == 1 else binstat_dd
    # or do I need to make histograms?
    X = binstat_dd(
        [subs[bc] for bc in bin_cols], [subs[c] for c in cols], "mean", bins
    ).statistic
    ic(X.shape)
    return X


main()
