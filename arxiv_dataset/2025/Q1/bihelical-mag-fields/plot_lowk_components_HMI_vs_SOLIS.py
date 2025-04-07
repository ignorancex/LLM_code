"""
Fourier-filter the HMI and SOLIS synoptic maps for a particular CR and see if they look similar.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from spectrum import filter_fourier
from read_FITS import HMIreader, SOLISreader
from utils import fig_saver

if __name__ == "__main__":
	cr = 2180
	k_max = 1e-2
	savefig = True
	savedir=  "plots/filtered_synoptic"
	mpl.style.use("kishore.mplstyle")
	
	save = fig_saver(savefig, savedir)
	read_h = HMIreader()
	read_s = SOLISreader()
	
	Bvec_h = read_h.get_Brtp(read_h.get_fname(cr))
	Bvec_s = read_s.get_Brtp(read_s.get_fname(cr))
	
	Bvec_h_filt = filter_fourier(Bvec_h, 0, k_max, k_axes=[-1,-2], L=[2*np.pi*700, np.pi*700])
	Bvec_s_filt = filter_fourier(Bvec_s, 0, k_max, k_axes=[-1,-2], L=[2*np.pi*700, np.pi*700])
	
	fig = plt.figure()
	gs = mpl.gridspec.GridSpec(1,3, width_ratios=[0.475, 0.475, 0.05], figure=fig)
	ax0 = fig.add_subplot(gs[0])
	ax1 = fig.add_subplot(gs[1], sharex=ax0, sharey=ax0)
	ax_cbar = fig.add_subplot(gs[2])
	
	vmax = max(np.max(np.abs(Bvec_h_filt)), np.max(np.abs(Bvec_s_filt)))
	
	im_kwargs = {
		'origin': 'lower',
		'extent': [0,360,-90,90],
		'vmin': -vmax,
		'vmax': vmax,
		'cmap': 'bwr',
		'aspect': 'auto',
		}
	
	im0 = ax0.imshow(Bvec_h_filt[0], **im_kwargs)
	ax0.set_title("HMI")
	ax0.set_ylabel(r"$\lambda$ (degrees)")
	ax0.set_xlabel(r"$\phi$ (degrees)")
	
	im1 = ax1.imshow(Bvec_s_filt[0], **im_kwargs)
	ax1.set_title("SOLIS")
	plt.setp(ax1.get_yticklabels(), visible=False)
	ax1.set_xlabel(r"$\phi$ (degrees)")
	
	c = fig.colorbar(im1, cax=ax_cbar)
	c.set_label(r"$B_r$ (G)")
	
	assert k_max == 1e-2
	fig.set_size_inches(5.4,2.3)
	
	save(fig, f"synoptic_lowpass_cr_{cr}_kmax_{k_max}.pdf")
