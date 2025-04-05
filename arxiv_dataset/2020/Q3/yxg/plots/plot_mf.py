# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
####
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as colors
from matplotlib import rc
rc("font", **{"family":"sans-serif", "sans-serif":["Helvetica"]})
rc("text", usetex=True)



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # https://matplotlib.org/api/colors_api.html
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


z = np.linspace(0.01, 0.5, 20)
a = 1/(1+z)


cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665)

hmd_200m = ccl.HMDef200mat()
hmfs=[ccl.MassFuncTinker08(cosmo, mass_def=hmd_200m),
      ccl.MassFuncTinker10(cosmo, mass_def=hmd_200m),      
      ccl.MassFuncDespali16(cosmo, mass_def=hmd_200m)]

M = np.logspace(10, 15, 100)

mfr_t08 = [[]]*len(a)
mfr_boc = [[]]*len(a)
for i, sf in enumerate(a):
    mf = [hmf.get_mass_function(cosmo, M, sf)
          for hmf in hmfs]
    mfr_t08[i] = mf[1]/mf[0]
    mfr_boc[i] = mf[2]/mf[0]

mfr_t08 = np.array(mfr_t08)
mfr_boc = np.array(mfr_boc)

cmap = truncate_colormap(cm.Reds, 0.2, 1.0)
cmap_b = truncate_colormap(cm.Blues, 0.2, 1.0)
col = [cmap(i) for i in np.linspace(0, 1, len(a))]

fig, ax = plt.subplots()
ax.set_xlim(M.min(), M.max())
ax.axhline(y=1, ls="--", color="k")
ax.plot([-1,-1],[-1,-1],'k-',label=r'${\rm Tinker\,\, 2010}$')
ax.plot([-1,-1],[-1,-1],'k--',label=r'${\rm Despali\,\, 2016}$')
[ax.loglog(M, R, c=c) for R, c, red in zip(mfr_t08, col, z)]
[ax.loglog(M, R, '--', c=c) for R, c, red in zip(mfr_boc, col, z)]

ax.set_xlim([M[0],M[-1]])
ax.set_ylim([0.9,1.4])
ax.yaxis.set_major_formatter(FormatStrFormatter("$%.1f$"))
ax.yaxis.set_minor_formatter(FormatStrFormatter("$%.1f$"))

sm = plt.cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=z.min(), vmax=z.max()))
sm._A = []
cbar = fig.colorbar(sm)
ticks = cbar.get_ticks()
cbar.ax.invert_yaxis()
cbar.set_ticks(ticks[::-1])

ax.set_xlabel(r"$M_{200m} \mathrm{/ M_{\odot}}$", fontsize=17)
ax.set_ylabel(r"${\rm Ratio\,\,with\,\,Tinker\,\,2008}$", fontsize=17)
ax.tick_params(which="both", labelsize="large")
ax.set_yscale('linear')

cbar.set_label("$z$", rotation=0, labelpad=15, fontsize=17)
cbar.ax.tick_params(labelsize="large")
plt.legend(loc='upper left', frameon=False, fontsize=15)
plt.savefig("notes/paper/mf_ratio.pdf", bbox_inches="tight")
plt.show()
