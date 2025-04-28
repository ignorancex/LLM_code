import paths
from lib_plot_bin import plot_BE_r
from lib_engineered import (
    get_M_boundary,
    plot_XY,
    plot_entropy,
    sorter_engineered_profiles,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import glob

fig = plt.figure()
gs = gridspec.GridSpec(100, 100)
ax = fig.add_subplot(gs[:50, :])
bx = fig.add_subplot(gs[50:, :])

# plot single star
init_model = (
    paths.data
    / "MESA_output/engineered_stars/TAMS_models/30_rot0_to_TAMS/LOGS/TAMS.data"
)
delta_M_bound, M_bound_min, M_bound_max = get_M_boundary(init_model, offset=0.05)
plot_entropy(init_model, ax, lw=3, ls="-", color="r", zorder=10, label="$30\,M_\odot$")
plot_XY(init_model, bx, lw=3, color="r", zorder=10)
# plot engineered models
root_grid30 = str(paths.data / "MESA_output/engineered_stars/same_core/grid30/")
grid_folders = sorted(
    glob.glob(root_grid30 + "/*.*/LOGS/"), key=sorter_engineered_profiles
)
colors = plt.cm.viridis(np.linspace(0, 1, len(grid_folders)))
# plot engineered models
for f in grid_folders:
    pfile = f + "/profile1.data"
    label = ""  # f.split('/')[-2]
    delta_M_bound, M_bound_max, M_bound_min = get_M_boundary(pfile, offset=0.05)
    ax.axvspan(M_bound_min, M_bound_max, fc="#808080", alpha=0.1, zorder=0)
    bx.axvspan(M_bound_min, M_bound_max, fc="#808080", alpha=0.1, zorder=0)
    label = f"{delta_M_bound:.1f}"  # f.split('/')[-2]
    c = colors[grid_folders.index(f)]
    plot_entropy(pfile, ax, c=c, lw=2)
    plot_XY(pfile, bx, c=c, lw=2)
bx.plot(np.nan, np.nan, ls="-", c="k", label=r"$^1\mathrm{H}$")
bx.plot(np.nan, np.nan, ls="--", c="k", label=r"$^4\mathrm{He}$")
bx.legend(handletextpad=0.4, handlelength=0.75, columnspacing=0.75, loc="center left")
ax.set_xticklabels([])
ax.legend(handletextpad=0.25, handlelength=0.5, columnspacing=0.5, loc="upper left")
bx.set_xlabel(r"$Mass \ [M_\odot]$")
bx.set_ylabel(r"$X_i$")
ax.set_ylabel(r"$s \ [k_{B}N_{A}]$")
plt.savefig(paths.figures / "engineered_TAMS.pdf")
