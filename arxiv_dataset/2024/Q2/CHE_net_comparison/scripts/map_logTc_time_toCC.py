import numpy as np
from MESAreader import getSrcCol
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_logTc_vs_time(hfile, ax, annotate = False, **kwargs):
    src, col = getSrcCol(hfile)
    logTc = src[:, col.index("log_center_T")]
    t = src[:, col.index("star_age")]
    logt_left = np.log10(t[-1]-t)
    ax.plot(logt_left, logTc, **kwargs)
    if annotate:
        s_center = src[:, col.index("center_entropy")]
        o16_center = src[:, col.index("center_o16")]
        ax.axvline(logt_left[np.argmin(np.absolute(s_center-4))], 0, 1, color=kwargs["c"], ls=":", zorder=0)
        ax.axvline(logt_left[np.argmin(np.absolute(o16_center-0.001))], 0, 1, color=kwargs["c"], ls="--", zorder=0)


if __name__ == "__main__":
    root = "../MESA_results/"
    small = root + '40_rot0.6_small_net/'
    LOGS_small = small + "LOGS1/"
    hfile_small = LOGS_small+'history.data'
    large = root + '40_rot0.6_large_net/'
    LOGS_large = large + 'LOGS1/'
    hfile_large = LOGS_large+'history.data'

    fig = plt.figure()
    gs = gridspec.GridSpec(120, 120)
    ax = fig.add_subplot(gs[:, :])

    plot_logTc_vs_time(hfile_small, ax, annotate=True, c="C0", lw=5, label="22 isotopes")
    plot_logTc_vs_time(hfile_large, ax, annotate=True, c="C1", label="128 isotopes")

    ax.invert_xaxis()
    ax.legend()
    ax.set_xlabel(r"$\log_{10}((t_\mathrm{core\ collapse} - t)/\mathrm{[yr]})$")
    ax.set_ylabel(r"$\log_{10}(T_{c}/\mathrm{[K]})$")
    plt.savefig("../logTc_vs_logt_left.pdf")
