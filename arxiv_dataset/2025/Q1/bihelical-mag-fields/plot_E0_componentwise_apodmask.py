"""
Motivated by observation of double-peaked magnetic spectra in the minimum between cycles 24--25, we now examine if all the three components of the magnetic field show the same behaviour even when the HMI maps are apodized and masked.
"""

import numpy as np

from plot_E0_twopeak import HMIreader_dblexcmsk
from plot_E0_componentwise import plot_Erpm_with_err
from utils import fig_saver

if __name__ == "__main__":
	savefig = True
	savedir = "plots/Espec_componentwise_apodmask"
	
	save = fig_saver(savefig, savedir)
	read = HMIreader_dblexcmsk(max_lat=60, threshold=200)
	
	cr_bins = np.arange(2097,2268,10)
	for i in range(len(cr_bins)-1):
		cr_list = [f"{cr}" for cr in range(cr_bins[i], cr_bins[i+1])]
		figname = f"{cr_bins[i]}-{cr_bins[i+1]-1}.pdf"
		save(plot_Erpm_with_err(cr_list, read=read), figname)
