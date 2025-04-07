"""
Similar to the plot produced by reproduce_singh18_HMI_dbl, but for different CR. THE HMI maps are apodized before calculating the spectra.
"""

import numpy as np

from read_FITS import HMIreader_dbl, ExciseLatitudeMixin, MaskWeakMixin
from utils import fig_saver
from plot_hel_with_err import E0H1_dbl, plot_hel_with_err


class HMIreader_dblexcmsk(ExciseLatitudeMixin, MaskWeakMixin, HMIreader_dbl):
	pass

if __name__ == "__main__":
	savefig = True
	savedir = "plots/hel_with_err_HMIapodmask"
	
	save = fig_saver(savefig, savedir)
	read = HMIreader_dblexcmsk(max_lat=60, threshold=200)
	
	cr_bins = np.arange(2097,2268,10)
	for i in range(len(cr_bins)-1):
		cr_list = [f"{cr}" for cr in range(cr_bins[i], cr_bins[i+1])]
		figname = f"{cr_bins[i]}-{cr_bins[i+1]-1}.pdf"
		
		res = E0H1_dbl(cr_list, read)
		fig = plot_hel_with_err(res)
		fig.suptitle(f"CR: {min(cr_list)}–{max(cr_list)}")
		fig.set_size_inches(4,4)
		fig.tight_layout()
		
		save(fig, figname)
