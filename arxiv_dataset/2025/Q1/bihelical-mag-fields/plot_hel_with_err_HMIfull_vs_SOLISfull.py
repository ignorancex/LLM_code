"""
Similar to the plot produced by reproduce_singh18_HMI_dbl, but for different CR.
"""

import numpy as np
import matplotlib.pyplot as plt

# from spectrum import calc_spec_G2 as calc_spec, signed_loglog_plot
from read_FITS import SOLISreader_noexc, StackLatitudeMixin, HMIreader_dbl
# from utils import jackknife, downsample_half, fig_saver
from utils import fig_saver
from plot_hel_with_err import E0H1_dbl
from plot_hel_with_err_HMIapodmask_vs_SOLIS import plot_hel_with_err_compare

class SOLISreader_dbl_noexc(
	StackLatitudeMixin,
	SOLISreader_noexc,
	): pass

def E0H1_SOLISdbl(cr_list):
	read = SOLISreader_dbl_noexc()
	return E0H1_dbl(cr_list, read)

def E0H1_HMIdbl(cr_list):
	read = HMIreader_dbl()
	return E0H1_dbl(cr_list, read)

if __name__ == "__main__":
	savefig = True
	savedir = "plots/hel_with_err_compare_HMIfull_SOLISfull"
	max_lat = 60 #For HMI apodization
	threshold = 200 #For HMI masking
	
	#Just like Singh 2018, we exclude certain Carrington rotations.
	cr_exclude = [2099, 2107, 2127, 2139, 2152, 2153, 2154, 2155, 2163, 2164, 2166, 2167, 2192, 2196]
	cr_bins = list(np.arange(2097,2196,10))
	
	save = fig_saver(savefig, savedir)
	for i in range(len(cr_bins)-1):
		cr_list = [cr for cr in range(cr_bins[i], cr_bins[i+1]) if cr not in cr_exclude]
		figname = f"{cr_bins[i]:04d}-{cr_bins[i+1]-1:04d}.pdf"
		
		res_h = E0H1_HMIdbl(cr_list)
		res_h.title = "HMI"
		
		res_s = E0H1_SOLISdbl(cr_list)
		res_s.title = "SOLIS"
		
		fig = plot_hel_with_err_compare(res_h, res_s)
		fig.suptitle(f"CR: {min(cr_list):04d}–{max(cr_list):04d}")
		fig.set_size_inches(6.3,4)
		fig.tight_layout()
		
		save(fig, figname)
