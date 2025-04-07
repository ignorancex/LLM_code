"""
Test the two-scale method on a simulation where the sign of helicity flips across z=0. Here, we perform the one-sided shift that was used in earlier works.
"""

import matplotlib.pyplot as plt
import pencil as pc
import numpy as np
import os

import sys
import pathlib
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

from spectrum import signed_loglog_plot
from utils import fig_saver, real
from plot_from_simulations import SpecFromSim

savefig = True #whether to save plots
simpath = "1"
savedir = os.path.join(simpath, "plots") #Where to save plots

save = fig_saver(savefig, savedir)

sim = pc.sim.get(quiet=True, path=simpath)
av = pc.read.aver(quiet=True, datadir=sim.datadir, simdir=sim.path)
grid = sim.grid

spec = SpecFromSim(
	sim,
	double_domain = False,
	shift_onesided = 1,
	var_file="var.h5",
	)

H1av = spec.H1av
E0av = spec.E0av
k = spec.k

fig,axs = plt.subplots(ncols=2)

axs[0].plot(grid.z, av.xy.abmz[-1])
axs[0].set_xlabel(r"$z$")
axs[0].set_ylabel(r"$\left< \vec{A}\cdot\vec{B} \right>_{xy}$")
axs[0].set_xlim(min(grid.z), max(grid.z))
axs[0].axhline(0, ls=':', c='k')

handles = []
handles.extend( signed_loglog_plot(k, k*(-np.imag(H1av)), axs[1], {'label':r"$-\mathrm{Im}(k\,H(k,K_1))$"}) )
handles.extend( axs[1].loglog(k, real(E0av), label="$E(k,0)$") )
axs[1].legend(handles=handles)
axs[1].set_xlabel("k")

fig.set_size_inches(6.4,3)
fig.tight_layout()
save(fig, "check_helspec_calc.pdf")
