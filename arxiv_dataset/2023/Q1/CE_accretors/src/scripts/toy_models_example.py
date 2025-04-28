from lib_plot_bin import plot_BE_r
from lib_engineered import get_M_boundary, sorter_engineered_profiles
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import paths

fig = plt.figure()
gs = gridspec.GridSpec(100, 100)
ax = fig.add_subplot(gs[:, :])

init_model = (
    paths.data
    / "MESA_output/engineered_stars/TAMS_models/30_rot0_to_TAMS/LOGS/TAMS.data"
)
delta_M_bound, M_bound_min, M_bound_max = get_M_boundary(init_model, offset=0.05)


nonrot30 = paths.data / "MESA_output/single_stars/Z_0.0019/30_rot0.0"
pfile_normal = nonrot30 / "LOGS/500Rsun.data"
plot_BE_r(
    pfile_normal, ax, alpha_th=0.0, top_axis=False, lw=3, ls="--", c="r", zorder=10
)
plot_BE_r(
    pfile_normal,
    ax,
    alpha_th=1.0,
    top_axis=False,
    lw=3,
    ls="-",
    c="r",
    zorder=10,
    label=str(init_model).split("/")[-3].split("_rot")[0] + "$M_\odot$",
)


root_grid30 = str(paths.data / "MESA_output/engineered_stars/same_core/grid30/")
grid_folders = sorted(
    glob.glob(root_grid30 + "/*.*/LOGS/"), key=sorter_engineered_profiles
)
colors = plt.cm.viridis(np.linspace(0, 1, len(grid_folders)))
for f in grid_folders:
    pfile = f + "/profile1.data"
    delta_M_bound, M_bound_max, M_bound_min = get_M_boundary(pfile, offset=0.05)
    label = f"{delta_M_bound:.2f}"  # f.split('/')[-2]
    c = colors[grid_folders.index(f)]
    pfile_end = f + "/500Rsun.data"
    plot_BE_r(
        pfile_end, ax, alpha_th=0.0, top_axis=False, lw=2, ls="--", c=c
    )  # , label=label)
    plot_BE_r(
        pfile_end, ax, alpha_th=1.0, top_axis=False, lw=2, ls="-", c=c
    )  # , label=label)

ax.plot(np.nan, np.nan, ls="-", c="k", label=r"$\alpha_{th}=1.0$")
ax.plot(np.nan, np.nan, ls="--", c="k", label=r"$\alpha_\mathrm{th}=0.0$")
ax.legend(handlelength=0.95, handletextpad=0.05, columnspacing=0.75, loc="upper right")
ax.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")
ax.set_ylabel(r"$BE \ \mathrm{[erg]}$")
ax.text(
    11.5,
    1e47,
    r"$R=500\,R_\odot$",
    fontsize=30,
    va="bottom",
    ha="center",
    transform=ax.transData,
    bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=0.1"),
)

ax.set_yscale("log")
ax.set_ylim(1e46, 2e51)
ax.set_xlim(10, 14)

plt.savefig(paths.figures / "toy_models_example.pdf")
