from lib_plot_bin import plot_HRD, annotate_radii_hrd
from MESAreader import getSrcCol
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import paths

root = paths.data / "MESA_output"
root_bin = root / "binaries/Z_0.0019"
b1 = (
    root_bin
    / "m1_18.0000_m2_15.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/"
)
b2 = (
    root_bin
    / "m1_20.0000_m2_17.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/"
)
b3 = (
    root_bin
    / "m1_38.0000_m2_30.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/"
)

bin_folders = [b1, b2, b3]

fig = plt.figure()
gs = gridspec.GridSpec(110, 120)
gs.update(wspace=0, hspace=0)  # top=1.1)
ax = plt.subplot(gs[:, :])

# get default colors
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

for f in bin_folders:
    c = colors[bin_folders.index(f)]
    LOGS2 = f / "LOGS2/"
    hfile2 = str(LOGS2) + "/history.data"
    plot_HRD(
        hfile2, ax=ax, annotate_TAMS=True, annotate_RLOF=True, ls="-", c=c, zorder=10
    )
    LOGS1 = f / "LOGS1/"
    hfile1 = str(LOGS1) + "/history.data"
    plot_HRD(
        hfile1,
        ax=ax,
        annotate_TAMS=False,
        annotate_RLOF=True,
        ls="--",
        lw=2,
        c=c,
        zorder=0,
    )

annotate_radii_hrd(ax, [100, 200, 300, 500, 1000])
ax.set_xlim(xmin=3.6, xmax=4.75)
ax.invert_xaxis()
ax.plot(np.nan, np.nan, c="k", ls="--", lw=2, label="donor")
ax.plot(np.nan, np.nan, c="k", label="accretor")
ax.legend()
ax.set_xlabel(r"$\log_{10}(T_\mathrm{eff}/[\mathrm{K}])$")
ax.set_ylabel(r"$\log_{10}(L/L_\odot)$")
plt.savefig(paths.figures / "HRD.pdf")
