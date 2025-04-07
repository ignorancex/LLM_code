"""
Look at Carrington rotations 2177-2186, and check the robustness of the two-peaked structure in the magnetic energy spectrum.
Main checks:
1. Is it also visible in SOLIS data?
2. Does it still persist when the weak-field regions are masked?
3. Does it persist even when the domain is not doubled?
4. Does it persist when the HMI maps are apodized?

Secondary checks:
4. Do we recover the same helicity spectra from SOLIS and HMI?
"""

import numpy as np
import matplotlib.pyplot as plt

from read_FITS import HMIreader, HMIreader_dbl, ExciseLatitudeMixin, MaskWeakMixin, SOLISreader_dbl
from plot_hel_with_err import E0H1_dbl, result
from spectrum import calc_spec_G2 as calc_spec, signed_loglog_plot
from utils import jackknife, downsample_half, fig_saver, real

class HMIreader_dblexc(ExciseLatitudeMixin, HMIreader_dbl):
	pass

class HMIreader_dblmsk(MaskWeakMixin, HMIreader_dbl):
	pass

class HMIreader_dblexcmsk(ExciseLatitudeMixin, MaskWeakMixin, HMIreader_dbl):
	pass

def E0H1_HMIdbl(cr_list):
	read = HMIreader_dbl()
	return E0H1_dbl(cr_list, read)

def E0H1_HMIdblapod(cr_list, max_lat):
	read = HMIreader_dblexc(max_lat=max_lat)
	return E0H1_dbl(cr_list, read)

def E0H1_HMIdblmsk(cr_list, threshold):
	read = HMIreader_dblmsk(threshold=threshold)
	return E0H1_dbl(cr_list, read)

def E0H1_HMIdblapodmsk(cr_list, max_lat, threshold):
	read = HMIreader_dblexcmsk(max_lat=max_lat, threshold=threshold)
	return E0H1_dbl(cr_list, read)

def E0H1_HMIsgl(cr_list):
	L = np.array([2*np.pi*700,np.pi*700])
	
	read = HMIreader()
	E0_list = []
	H1_list = []
	for cr in cr_list:
		B_vec = read(f"images/hmi.b_synoptic_small.rebinned.{cr}")
		k, E0, _ = calc_spec(B_vec, K=np.array([0,0]), L=L)
		_, _, H1 = calc_spec(B_vec, K=np.array([0,1]), L=L)
		
		E0_list.append(E0)
		H1_list.append(H1)
	
	E0_list = np.array(E0_list)
	H1_list = np.array(H1_list)
	
	E0, E0_err = jackknife(E0_list, axis=0)
	nimH1, nimH1_err = jackknife(-np.imag(H1_list), axis=0)
	
	#Avoid some annoying matplotlib warnings
	E0 = real(E0)
	E0_err = real(E0_err)
	nimH1 = real(nimH1)
	nimH1_err = real(nimH1_err)
	
	return result(k, E0, E0_err, nimH1, nimH1_err)

def E0H1_SOLISdbl(cr_list):
	read = SOLISreader_dbl(max_lat=70)
	return E0H1_dbl(cr_list, read)

if __name__ == "__main__":
	cr_list = np.arange(2177,2187)
	max_lat = 60 #Used for HMI apodization
	threshold  = 200 #Used for HMI masking
	#TODO: Interestingly, I see a bihelical spectrum when the threshold is 200, but not when it is 500. Does this suggest that the large-scale signatures are not carried only by active regions?
	savefig = True
	savedir = "plots/test_twopeak"
	
	save = fig_saver(savefig, savedir)
	
	r_h1 = E0H1_HMIsgl(cr_list)
	r_h2 = E0H1_HMIdbl(cr_list)
	r_h2m = E0H1_HMIdblmsk(cr_list, threshold)
	r_s2 = E0H1_SOLISdbl(cr_list)
	r_h2a = E0H1_HMIdblapod(cr_list, max_lat)
	r_h2am = E0H1_HMIdblapodmsk(cr_list, max_lat=max_lat, threshold=threshold)
	
	#Compare HMI with SOLIS
	fig,axs = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'height_ratios': [2,1]})
	
	handles = signed_loglog_plot(r_h2.k, r_h2.k*r_h2.nimH1, axs[0,0], {'label':r"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,0].loglog(r_h2.k, r_h2.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,0].loglog(r_h2.k, np.abs(r_h2.nimH1)/r_h2.nimH1_err, label=r"$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,0].loglog(r_h2.k, r_h2.E0/r_h2.E0_err, label="$E(k,0)$")
	
	handles = signed_loglog_plot(r_s2.k, r_s2.k*r_s2.nimH1, axs[0,1], {'label':r"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,1].loglog(r_s2.k, r_s2.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,1].loglog(r_s2.k, np.abs(r_s2.nimH1)/r_s2.nimH1_err, label=r"$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,1].loglog(r_s2.k, r_s2.E0/r_s2.E0_err, label="$E(k,0)$")
	
	axs[0,0].set_title("HMI")
	axs[0,1].set_title("SOLIS")
	fig.suptitle(f"CR {min(cr_list)}–{max(cr_list)}")
	
	for ax in axs[1]:
		ax.axhline(1, ls=':', c='k')
		ax.set_ylabel("|data/error|")
		ax.set_xlabel("k")
	
	fig.set_size_inches(6,4)
	fig.tight_layout()
	save(fig, "compare_HMI_SOLIS.pdf")
	
	#See if domain-doubling changes the energy spectrum for the HMI data
	fig,ax = plt.subplots()
	ax.loglog(r_h1.k, r_h1.E0, label="undoubled")
	ax.loglog(r_h2.k, r_h2.E0, label="doubled", ls='--')
	ax.loglog(r_h2m.k, r_h2m.E0, label="masked", ls=':')
	ax.legend()
	ax.set_ylabel("$E(k,0)$")
	ax.set_xlabel("$k$")
	fig.suptitle(f"CR {min(cr_list)}–{max(cr_list)}")
	fig.set_size_inches(4,4)
	fig.tight_layout()
	save(fig, "effect_double_mask.pdf")
	
	#Effect of apodization on HMI data
	fig,axs = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'height_ratios': [2,1]})
	
	handles = signed_loglog_plot(r_h2.k, r_h2.k*r_h2.nimH1, axs[0,0], {'label':r"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,0].loglog(r_h2.k, r_h2.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,0].loglog(r_h2.k, np.abs(r_h2.nimH1)/r_h2.nimH1_err, label=r"$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,0].loglog(r_h2.k, r_h2.E0/r_h2.E0_err, label="$E(k,0)$")
	
	handles = signed_loglog_plot(r_h2a.k, r_h2a.k*r_h2a.nimH1, axs[0,1], {'label':r"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,1].loglog(r_h2a.k, r_h2a.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,1].loglog(r_h2a.k, np.abs(r_h2a.nimH1)/r_h2a.nimH1_err, label=r"$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,1].loglog(r_h2a.k, r_h2a.E0/r_h2a.E0_err, label="$E(k,0)$")
	
	axs[0,0].set_title("Full")
	axs[0,1].set_title(rf"$\left|\lambda\right| < {max_lat}^{{\circ}}$")
	fig.suptitle(f"CR {min(cr_list)}–{max(cr_list)}")
	
	for ax in axs[1]:
		ax.axhline(1, ls=':', c='k')
		ax.set_ylabel("|data/error|")
		ax.set_xlabel("k")
	
	fig.set_size_inches(6,4)
	fig.tight_layout()
	save(fig, "effect_HMI_apodization.pdf")
	
	#Effect of masking weak-field regions on HMI data
	fig,axs = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'height_ratios': [2,1]})
	
	handles = signed_loglog_plot(r_h2.k, r_h2.k*r_h2.nimH1, axs[0,0], {'label':r"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,0].loglog(r_h2.k, r_h2.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,0].loglog(r_h2.k, np.abs(r_h2.nimH1)/r_h2.nimH1_err, label=r"$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,0].loglog(r_h2.k, r_h2.E0/r_h2.E0_err, label="$E(k,0)$")
	
	handles = signed_loglog_plot(r_h2m.k, r_h2m.k*r_h2m.nimH1, axs[0,1], {'label':r"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,1].loglog(r_h2m.k, r_h2m.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,1].loglog(r_h2m.k, np.abs(r_h2m.nimH1)/r_h2m.nimH1_err, label=r"$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,1].loglog(r_h2m.k, r_h2m.E0/r_h2m.E0_err, label="$E(k,0)$")
	
	axs[0,0].set_title("Full")
	axs[0,1].set_title(rf"$\left|\vec{{B}}\right| > {threshold}$")
	fig.suptitle(f"CR {min(cr_list)}–{max(cr_list)}")
	
	for ax in axs[1]:
		ax.axhline(1, ls=':', c='k')
		ax.set_ylabel("|data/error|")
		ax.set_xlabel("k")
	
	fig.set_size_inches(6,4)
	fig.tight_layout()
	save(fig, "effect_HMI_mask.pdf")
	
	#Effect of masking weak-field regions along with apodization on HMI data
	fig,axs = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw={'height_ratios': [2,1]})
	
	handles = signed_loglog_plot(r_h2.k, r_h2.k*r_h2.nimH1, axs[0,0], {'label':r"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,0].loglog(r_h2.k, r_h2.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,0].loglog(r_h2.k, np.abs(r_h2.nimH1)/r_h2.nimH1_err, label=r"$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,0].loglog(r_h2.k, r_h2.E0/r_h2.E0_err, label="$E(k,0)$")
	
	handles = signed_loglog_plot(r_h2am.k, r_h2am.k*r_h2am.nimH1, axs[0,1], {'label':r"$-\mathrm{Im}(k\,H(k,K_1))$"})
	h = axs[0,1].loglog(r_h2am.k, r_h2am.E0, label="$E(k,0)$")
	handles.extend(h)
	
	axs[1,1].loglog(r_h2am.k, np.abs(r_h2am.nimH1)/r_h2am.nimH1_err, label=r"$-\mathrm{Im}(k H(k,K_1))$")
	axs[1,1].loglog(r_h2am.k, r_h2am.E0/r_h2am.E0_err, label="$E(k,0)$")
	
	axs[0,0].set_title("Full")
	axs[0,1].set_title(rf"$\left|\lambda\right| < {max_lat}^{{\circ}}, \left|\vec{{B}}\right| > {threshold}$")
	fig.suptitle(f"CR {min(cr_list)}–{max(cr_list)}")
	
	for ax in axs[1]:
		ax.axhline(1, ls=':', c='k')
		ax.set_ylabel("|data/error|")
		ax.set_xlabel("k")
	
	fig.set_size_inches(6,4)
	fig.tight_layout()
	save(fig, "effect_HMI_apod_mask.pdf")
