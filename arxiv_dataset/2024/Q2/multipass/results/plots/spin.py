import sys
sys.path.append("..")
from analyse import *
from scipy.special import gamma

matplotlib.rcParams.update({'font.size': 14})

samples = loadMC('../ppisn-spin/long', 'ppisn+trivial+beta')

n = 5000
CL = 0.68
dcl = 100 * (1-CL)/2
paras = np.random.multivariate_normal(best_fit(samples)[9:], samples.cov()[9:,9:], size=n)

mask = np.all((1<paras, paras < 10), axis=(0,2))
paras = paras[mask]

# al1,be1, al2, be2 = best_fit(samples)[9:]
chi=np.linspace(0,1,500)

def beta(al, be):
    return chi**(al-1) * (1-chi)**(be) * gamma(al+be)/gamma(al)/gamma(be)

spin1 = np.array([beta(*i) for i in paras[:,:2]])
spin2 = np.array([beta(*i) for i in paras[:,2:]])

fig = figure(figsize=(4,3))
fill_between(
    chi,
    *np.percentile(spin1, [dcl, 100-dcl],axis=0),
    alpha=0.2, label='$1g$'
)
plot(
    chi,
    np.percentile(spin1, 50,axis=0),
    'C0'
)
fill_between(
    chi,
    *np.percentile(spin2, [dcl, 100-dcl],axis=0),
    alpha=0.2, label='$2g$'
)
plot(
    chi,
    np.percentile(spin2, 50,axis=0),
    'C1'
)
legend()

gca().xaxis.set_minor_locator(MultipleLocator(0.1))
gca().xaxis.set_major_locator(MultipleLocator(0.2))

xlabel(r"$\chi_{\rm eff}$")
ylabel(r"${\rm d}R/{\rm d}\chi_{\rm eff}$")
xlim(0,1)

fig.savefig("spin.pdf")

# fig, axs = subplots(
#     4,
#     4,
#     gridspec_kw={"hspace": 0, "wspace": 0},
#     # figsize=(1.5 * 7.84, 1.5 * 5.9),
#     # sharex="col",
#     # sharey="row",
# )

# vars = parameters["ppisn+trivial+beta-mass"]

# for i2, (latex, name2) in enumerate(vars):
#     for i1, (_, name1) in enumerate(vars):
#         if i1 == i2:
#             density = samples.get1DDensityGridData(name1).normalize()
#             axs[i1, i2].plot(density.x, density.P)
#             axs[i1, i2].get_yaxis().set_ticks([])

#         elif i1 > i2:
#             # axs[i1, i2]._shared_axes["y"].remove(axs[i1, i1])
#             density = samples.get2DDensityGridData(name2, name1).normalize()
#             axs[i1, i2].contourf(
#                 *np.meshgrid(density.x, density.y),
#                 density.P,
#                 levels=density.getContourLevels((0.95, 0.68, 1e-10))
#             )

#             if i2 > 0:
#                 axs[i1, i2].get_yaxis().set_ticklabels([])
#         else:
#             axs[i1, i2].set_axis_off()


# for i, (latex, _) in enumerate(vars[1:]):
#     axs[i+1,0].set_ylabel(f"${latex}$")

# for i, (latex, _) in enumerate(vars):
#     axs[-1,i].set_xlabel(f"${latex}$")

# lims = [
#     (-.7,11.1),
#     (-.7,11.1),
#     (-.7,11.1),
#     (-.7,11.1),
# ]

# for i in range(4):
#     for j, l in enumerate(lims):
#         axs[i,j].set_xlim(l)

#         if i < j:
#             axs[j,i].set_ylim(l)

# fig.savefig("spin-corner.pdf")
