from MESAreader import getSrcCol
from lib_plot_bin import plot_ratio_BE_r, get_ax_from_pfile, Rsun_cm
from lib_engineered import get_M_boundary
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import warnings
import paths

def get_rot_from_folder(f):
    """
    Parameters:
    -----------
    f: `string` or `path` for the folder. The
       folder name has to be formated as arbitrary/path/ending/in/*_rot/
    Returns:
    -------
    float with initial rotation rate assumed for the star.
    """
    return float(f.split('/')[-2].split('_rot')[-1])


def plot_one_outer_R(pfile_ref, ax, folders_rot, radius_string, color_ref='orange', legend=False):
    """ helper function to plot one panel """
    ratio = plot_ratio_BE_r(
            pfile_ref,
            pfile_ref,
            ax,
            alpha_th=1.0,
            alpha_rot=0.0,
            c=color_ref,
            ls="-",
            lw=2,
            zorder=2,
        )
    # highlight CEB boundaries for accretor
    delta_M_boundary, max_M_boundary, min_M_boundary = get_M_boundary(pfile_ref, offset=0.01)
    src, col = getSrcCol(pfile_ref)
    m = src[:, col.index("mass")]
    r = src[:, col.index("radius")]
    r_inner = r[np.argmin(np.absolute(m-min_M_boundary))]
    r_outer = r[np.argmin(np.absolute(m-max_M_boundary))]
    ax.axvspan(np.log10(r_inner*Rsun_cm), np.log10(r_outer*Rsun_cm), fc="#808080", alpha=0.5, zorder=0)
    colors = plt.cm.plasma(np.linspace(0, 1, len(folders_rot)))
    for f in folders_rot:
        p = glob.glob(f + "/LOGS/"+radius_string)[0]
        if legend:
            label = "$\omega/\omega_\mathrm{crit}="+str(get_rot_from_folder(f))+"$"
        else:
            label=""
        c=colors[folders_rot.index(f)]
        lw=2
        z=3
        if "rot0.0" in f:
            c='c'
            lw=3
            z=10
        ratio = plot_ratio_BE_r(
            p,
            pfile_ref,
            ax,
            alpha_th=1.0,
            alpha_rot=0.0,
            color=c,
            ls="-",
            lw=lw,
            zorder=z,
            label=label)
        if legend== True: ax.legend(ncol=2, handlelength=0.5, columnspacing=0.5, handletextpad=0.02)


def plot_single_div_rot_and_accretor_div_rot(accretor_root, folders_rot, fname=None):
    fig = plt.figure(figsize=(12,19.5))
    gs = gridspec.GridSpec(80, 115)
    ax1 = fig.add_subplot(gs[:20, :50])
    ax2 = fig.add_subplot(gs[20:40, :50])
    ax3 = fig.add_subplot(gs[40:60, :50])
    ax4 = fig.add_subplot(gs[60:80, :50])

    bx1 = fig.add_subplot(gs[:20, 65:])
    bx2 = fig.add_subplot(gs[20:40, 65:])
    bx3 = fig.add_subplot(gs[40:60, 65:])
    bx4 = fig.add_subplot(gs[60:80, 65:])

    axes = [ax1, ax2, ax3, ax4]
    for ax in axes:
        ax.axhline(1,0,1,lw='2', ls='--', c="#808080")
        ax.set_xlim(8.2, 14)
        ax.set_ylim(-0.05, 2.5)
        ax.text(
            0.05,
            0.85,
            "non-rotating more bound",
            fontsize=20,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            zorder=1,
        )
        ax.text(
            0.05,
            0.05,
            "non-rotating less bound",
            fontsize=20,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            zorder=1,
        )

        if ax != axes[-1]:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")

    bxes = [bx1, bx2, bx3, bx4]
    for bx in bxes:
        bx.axhline(1,0,1,lw='2', ls='--', c="#808080")
        bx.set_xlim(8.2, 14)
        bx.set_ylim(-0.05, 2.5)
        bx.text(
            0.05,
            0.85,
            "accretor more bound",
            fontsize=20,
            transform=bx.transAxes,
            va="bottom",
            ha="left",
            zorder=1,
        )
        bx.text(
            0.05,
            0.05,
            "accretor less bound",
            fontsize=20,
            transform=bx.transAxes,
            va="bottom",
            ha="left",
            zorder=1,
        )
        if bx != bxes[-1]:
            bx.set_xticklabels([])
        else:
            bx.set_xlabel(r"$\log_{10}(r/\mathrm{[cm]})$")
        dx = bx.twinx()
        if bx == bxes[0]:
            label = "$100\,R_\odot$"
            X = 100 * Rsun_cm
        if bx == bxes[1]:
            label = "$200\,R_\odot$"
            X = 200 * Rsun_cm
        if bx == bxes[2]:
            label = "$300\,R_\odot$"
            X = 300 * Rsun_cm
        if bx == bxes[3]:
            label = "$500\,R_\odot$"
            X = 500 * Rsun_cm
        dx.set_ylabel(label)
        bx.axvline(np.log10(X), 0, 1, ls=":", lw=2, c="#808080", zorder=1)
        axes[bxes.index(bx)].axvline(np.log10(X), 0, 1, ls=":", lw=2, c="#808080", zorder=1)
        dx.set_yticks(ax.get_yticks())
        dx.set_yticklabels([])
        dx.set_yticks(ax.get_yticks(minor=True), minor=True)
        dx.set_yticklabels([], minor=True)
        dx.set_ylim(ax.get_ylim())

    ax3.set_ylabel(r"$BE(\mathrm{non-rotating})/BE(\mathrm{rotating})$", horizontalalignment='right', y=1.75)
    bx3.set_ylabel(r"$BE(\mathrm{accretor})/BE(\mathrm{rotating})$", horizontalalignment='right', y=1.75)
    accretor_profiles = sorted(glob.glob(str(accretor_root) + "/LOGS2/" + "*Rsun.data"))
    for pfile_accretor in accretor_profiles:
        string = pfile_accretor.split("/")[-1]
        bx = get_ax_from_pfile(pfile_accretor, bxes)
        legend = False
        plot_one_outer_R(pfile_accretor, bx, folders_rot, string, legend=legend)
    non_rot_profiles  = sorted(glob.glob(str(folders_rot[0]) + "/LOGS/" + "*Rsun.data"))
    for pfile_non_rot in non_rot_profiles:
        string = pfile_non_rot.split("/")[-1]
        ax = get_ax_from_pfile(pfile_non_rot, axes)
        legend = False
        plot_one_outer_R(pfile_non_rot, ax, folders_rot, string, color_ref='c', legend=legend)
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    else:
        plt.show()

if __name__ == "__main__":
    root = paths.data / "MESA_output/"
    root_rot = root / "single_stars/Z_0.0019/"
    folders_rot = sorted(glob.glob(str(root_rot)+"/18_rot0*/"))
    root_accretors = root / "binaries/Z_0.0019/"
    accretor_root = str(root_accretors)+"/m1_18.0000_m2_15.0000_initial_z_0.0019_initial_period_in_days_1.0000e+02_grid_index_0_1/"
    plot_single_div_rot_and_accretor_div_rot(accretor_root, folders_rot, fname=paths.figures / "accretors_rotators_single.pdf")
