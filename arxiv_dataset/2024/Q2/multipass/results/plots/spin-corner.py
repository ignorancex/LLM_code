import sys

sys.path.append("..")
from analyse import *

samples_nospin = loadMC("../ppisn/long", "ppisn+trivial+trivial")
samples_spin = loadMC("../ppisn-spin/long", "ppisn+trivial+beta")


fig, axs = subplots(
    7+4,
    7+4,
    gridspec_kw={"hspace": 0, "wspace": 0},
    figsize=(1.5 * 7.84, 1.5 * 5.9),
    # sharex="col",
    # sharey="row",
)

transformations = [
    ("mgap", lambda a: a, "linear"),
    ("a", lambda a: a, "linear"),
    ("b", lambda a: a, "linear"),
    ("d", lambda a: a, "linear"),
    ("lam21", lambda a: 10**a, "log"),
    ("lam12", lambda a: 10**a, "log"),
    ("bq0", lambda a: a, "linear"),
    ("alpha1", lambda a: a, "linear"),
    ("beta1", lambda a: a, "linear"),
    ("alpha2", lambda a: a, "linear"),
    ("beta2", lambda a: a, "linear"),
]


sharedx = axs[0,0].get_shared_x_axes()
sharedy = axs[0,0].get_shared_y_axes()


for i2, (name2, t2, s2) in enumerate(transformations):
    for i1, (name1, t1, s1) in enumerate(transformations):
        if i1 == i2:
            if i1 < 7:
                density = samples_nospin.get1DDensityGridData(name1).normalize()
                axs[i1, i2].plot(t1(density.x), density.P, 'C1')

            density = samples_spin.get1DDensityGridData(name1).normalize()
            axs[i1, i2].plot(t1(density.x), density.P, 'C0')
            axs[i1, i2].get_yaxis().set_ticks([])
            axs[i1, i2].set_xscale(s1)

        elif i1 > i2:
            if i1 < 7:
                density = samples_nospin.get2DDensityGridData(name2, name1).normalize()
                axs[i1, i2].contourf(
                    *np.meshgrid(t2(density.x), t1(density.y)),
                    density.P,
                    levels=density.getContourLevels((0.95, 0.68, 1e-10)),
                    colors=['C1','C1'],
                    alpha=[0.2,0.5]
                )

            density = samples_spin.get2DDensityGridData(name2, name1).normalize()
            axs[i1, i2].contourf(
                *np.meshgrid(t2(density.x), t1(density.y)),
                density.P,
                levels=density.getContourLevels((0.95, 0.68, 1e-10)),
                colors=['C0','C0'],
                alpha=[0.2,0.5]
            )

            axs[i1, i2].set_xscale(s2)
            axs[i1, i2].set_yscale(s1)
            if i2 > 0:
                axs[i1, i2].get_yaxis().set_ticklabels([])
        else:
            axs[i1, i2].set_axis_off()


axs[1,0].set_ylabel("$a$")
axs[2,0].set_ylabel("$b$")
axs[3,0].set_ylabel("$d$")
axs[4,0].set_ylabel("$\\lambda_{21}$")
axs[5,0].set_ylabel("$\\lambda_{12}$")
axs[6,0].set_ylabel("$\\beta_q$")
axs[7,0].set_ylabel("$\\alpha_1$")
axs[8,0].set_ylabel("$\\beta_1$")
axs[9,0].set_ylabel("$\\alpha_2$")
axs[10,0].set_ylabel("$\\beta_2$")

axs[10,0].set_xlabel(r"$M_{\rm BHMG}\,/\,M_{\odot}$")
axs[10,1].set_xlabel("$a$")
axs[10,2].set_xlabel("$b$")
axs[10,3].set_xlabel("$d$")
axs[10,4].set_xlabel("$\\lambda_{21}$")
axs[10,5].set_xlabel("$\\lambda_{12}$")
axs[10,6].set_xlabel("$\\beta_q$")
axs[10,7].set_xlabel("$\\alpha_1$")
axs[10,8].set_xlabel("$\\beta_1$")
axs[10,9].set_xlabel("$\\alpha_2$")
axs[10,10].set_xlabel("$\\beta_2$")

lims = [
    (14.,143.),
    (-0.14,0.64),
    (-4.48, 0.34),
    (-13.6, 3.60),
    (2e-9, 25),
    (2e-9, 25),
    (-3.6, 14.7),
    (0.9, 11),
    (0.9, 11),
    (0.9, 11),
    (0.9, 11),
]

for i in range(11):
    for j, l in enumerate(lims):
        axs[i,j].set_xlim(l)

        if i < j:
            axs[j,i].set_ylim(l)

axs[0,5].legend(
    [matplotlib.lines.Line2D([0], [0], color=f'C{i}') for i in range(2)],
    [r'${\rm with\ spin}$', r'${\rm without\ spin}$'],
    loc='center',
    fontsize=20
)


fig.tight_layout()
fig.savefig("full-corner.pdf")
