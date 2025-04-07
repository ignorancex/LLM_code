"""
Check how the 180 degree azimuth ambiguity affects the helicity spectrum.

The idea is that we compare the helicity spectra in two different 'realizations' of the same magnetograms. These realizations only differ in that the sign of the transverse component of the magnetic field in the weak-field regions (randomly flipped).
"""

import sys
import pathlib
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from spectrum import signed_loglog_plot
from read_FITS import HMIreader_dbl, RandomizeWeakMixin
from utils import fig_saver
from plot_hel_with_err import E0H1_dbl

class HMIreader_dbl_randazi(RandomizeWeakMixin, HMIreader_dbl): pass

if __name__ == "__main__":
	savefig = True
	savedir = root/"plots/azimuth_apj"
	mpl.style.use(root/"kishore_apj.mplstyle")
	
	save = fig_saver(savefig, savedir)
	
	cr_list = list(range(2143, 2153))
	threshold=150
	
	res_list = []
	for seed in range(2):
		read = HMIreader_dbl_randazi(threshold=threshold, seed=seed)
		res_list.append(E0H1_dbl(cr_list, read))
	
	fig,axs = plt.subplots(1, 2, sharex=True, sharey=True)
	
	for res, ax in zip(res_list, axs):
		signed_loglog_plot(
			res.k[1:],
			(res.k*res.nimH1)[1:],
			err = (res.k*res.nimH1_err)[1:],
			ax = ax,
			)
		
	axs[0].legend()
	axs[0].set_ylabel(r"$-\mathrm{Im}(k\,\widetilde{H}(k,K_1))$ (erg cm$^{-3}$)")
	fig.supxlabel("$k$ (Mm$^{-1}$)", size='medium')
	axs[0].margins(x=0)
	
	fig.set_size_inches(6.5,2.7)
	
	save(fig, f"compare_helspec_HMI_randazimuth_cr_{min(cr_list)}-{max(cr_list)}.pdf")
