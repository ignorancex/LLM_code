from lib_plot_bin import plot_BE_r
from lib_engineered import get_M_boundary
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import paths

init_model = (
    paths.data
    / "MESA_output/engineered_stars/TAMS_models/18_rot0_to_TAMS/LOGS/TAMS.data"
)
delta_M_bound, M_bound_min, M_bound_max = get_M_boundary(init_model)


fig = plt.figure()
gs = gridspec.GridSpec(100, 100)
ax = fig.add_subplot(gs[:, :])

rot_root = str(paths.data / "MESA_output/single_stars/Z_0.0019")
grid_folders = sorted(glob.glob(rot_root + "/18_rot0*"))
colors = plt.cm.plasma(np.linspace(0, 1, len(grid_folders)))
for f in grid_folders:
    label = f.split("/")[-1].split("_rot")[-1]
    if float(label) == 0:
        # skip non-rotating
        continue
    else:
        label = "$\omega/\omega_\mathrm{crit}=$" + label
        c = colors[grid_folders.index(f)]
        pfile_end = f + "/LOGS/500Rsun.data"
        plot_BE_r(
            pfile_end,
            ax,
            alpha_th=0.0,
            top_axis=False,
            lw=2,
            ls="--",
            c=c,
        )
        plot_BE_r(
            pfile_end,
            ax,
            alpha_th=1.0,
            top_axis=False,
            lw=2,
            ls="-",
            c=c,
            label=label,
        )

nonrot18 = paths.data / "MESA_output/single_stars/Z_0.0019/18_rot0.0/"
pfile_normal = nonrot18 / "LOGS/500Rsun.data"
plot_BE_r(
    pfile_normal,
    ax,
    alpha_th=0.0,
    top_axis=False,
    lw=3,
    c="c",
    zorder=10,
    ls="--",
)
plot_BE_r(
    pfile_normal,
    ax,
    alpha_th=1.0,
    top_axis=False,
    lw=3,
    c="c",
    zorder=10,
    ls="-",
    label=str(init_model).split("/")[-3].split("_rot")[0]
    + "$M_\odot, \omega/\omega_\mathrm{crit}=0$",
)


l1 = ax.legend(
    handletextpad=0.5,
    columnspacing=0.75,
    ncol=2,
    handlelength=0.5,
    fontsize=20,
    labelspacing=0.4,
    loc="lower left",
)
(a1,) = ax.plot(np.nan, np.nan, ls="-", c="k", label=r"$\alpha_{th}=1.0$")
(a0,) = ax.plot(np.nan, np.nan, ls="--", c="k", label=r"$\alpha_\mathrm{th}=0.0$")
l2 = ax.legend(
    [a1, a0],
    [r"$\alpha_{th}=1.0$", r"$\alpha_\mathrm{th}=0.0$"],
    handlelength=0.8,
    loc="upper right",
)
ax.add_artist(l1)


ax.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")
ax.set_ylabel(r"$BE\ [\mathrm{erg}]$")
ax.text(
    11.5,
    2.5e50,
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
plt.savefig(paths.figures / "rotation_models_example.pdf")
