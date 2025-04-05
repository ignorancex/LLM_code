import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
rc("font", **{"family":"sans-serif", "sans-serif":["Helvetica"]})
rc("text", usetex=True)


T = np.load("mf_ratios.npz")["ratio_tinker10"]
D = np.load("mf_ratios.npz")["ratio_despali16"]

M = np.logspace(10, 15, 100)
z = np.linspace(0, 0.5, 20)



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=len(T)):
    # https://matplotlib.org/api/colors_api.html
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


cmap = truncate_colormap(cm.Reds, 0.2, 1.0)
col = [cmap(i) for i in np.linspace(0, 1, len(T))]


fig, ax = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

# legend
ax[0].text(0.02, 0.87, "Tinker 2010", fontsize=17, transform=ax[0].transAxes)
ax[1].text(0.02, 0.87, "Despali 2016", fontsize=17, transform=ax[1].transAxes)

# eye guidance 1
ax[1].fill_between(M, T[0], T[-1], color="grey", alpha=0.15)
ax[0].fill_between(M, D[0], D[-1], color="grey", alpha=0.15)
# plots
[ax[0].semilogx(M, t, color=c) for t, c in zip(T, col)]
[ax[1].semilogx(M, t, color=c) for t, c in zip(D, col)]
# eye guidance 2
[xx.axhline(y=1, ls=":", lw=0.75, color="k") for xx in ax]

# axes limits
ax[0].set_ylim(T.min()/1.05, T.max()*1.05)
[xx.set_xlim(M.min(), M.max()) for xx in ax]
# show 2 decimals
[xx.yaxis.set_major_formatter(FormatStrFormatter("$%.1f$")) for xx in ax]
[xx.yaxis.set_minor_formatter(FormatStrFormatter("$%.1f$")) for xx in ax]

# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=z.min(), vmax=z.max()))
sm._A = []
cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cbar = fig.colorbar(sm, cax=cb_ax)
# cbar = fig.colorbar(sm)
ticks = cbar.get_ticks()
cbar.ax.invert_yaxis()
cbar.set_ticks(ticks[::-1])

# global params
ax[1].set_xlabel(r"$M_{200m} \mathrm{/ M_{\odot}}$", fontsize=17)
ax[0].set_ylabel(r"$\frac{n_{\mathrm{T}10}(M)}{n_{\mathrm{T}08}(M)}$", fontsize=17)
ax[1].set_ylabel(r"$\frac{n_{\mathrm{D}16}(M)}{n_{\mathrm{T}08}(M)}$", fontsize=17)

[xx.tick_params(which="both", labelsize="large") for xx in ax]

cbar.set_label("$z$", rotation=0, labelpad=15, fontsize=17)
cbar.ax.tick_params(labelsize="large")

plt.savefig("../notes/paper/mf_ratio.pdf", bbox_inches="tight")
plt.show()
