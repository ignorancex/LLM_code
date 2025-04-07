"""
Motivated by observation of double-peaked magnetic spectra in the minimum between cycles 24--25, we now examine if all the three components of the magnetic field show the same behaviour in HMI maps.
"""

import numpy as np
import matplotlib.pyplot as plt

from read_FITS import HMIreader
from spectrum import calc_spec_G2 as calc_spec
from utils import jackknife, fig_saver, real

def E0rpm_list_from_CR_list(cr_list, read=None):
	L = np.array([2*np.pi*700,np.pi*700])
	if read is None:
		read = HMIreader()
	
	E0r_list = []
	E0p_list = []
	E0mu_list = []
	for cr in cr_list:
		#No need to double domain here, since we are only interested in the K=0 mode.
		B_vec = read(read.get_fname(cr))
		
		B_r = B_vec.copy()
		B_p = B_vec.copy()
		B_mu = B_vec.copy()
		
		B_r[[1,2]] = 0
		B_p[[0,2]] = 0
		B_mu[[0,1]] = 0
		
		k, E0r, _ = calc_spec(B_r, K=np.array([0,0]), L=L)
		E0r_list.append(E0r)
		
		k, E0p, _ = calc_spec(B_p, K=np.array([0,0]), L=L)
		E0p_list.append(E0p)
		
		k, E0mu, _ = calc_spec(B_mu, K=np.array([0,0]), L=L)
		E0mu_list.append(E0mu)
	
	E0r_list = np.array(E0r_list)
	E0p_list = np.array(E0p_list)
	E0mu_list = np.array(E0mu_list)
	
	return k, E0r_list, E0p_list, E0mu_list

def plot_Erpm_with_err(cr_list, read=None):
	k, E0r_list, E0p_list, E0mu_list = E0rpm_list_from_CR_list(cr_list, read=read)
	
	E0r, E0r_err = jackknife(E0r_list, axis=0)
	E0p, E0p_err = jackknife(E0p_list, axis=0)
	E0mu, E0mu_err = jackknife(E0mu_list, axis=0)
	
	#Avoid some annoying matplotlib warnings
	E0r = real(E0r)
	E0r_err = real(E0r_err)
	E0p = real(E0p)
	E0p_err = real(E0p_err)
	E0mu = real(E0mu)
	E0mu_err = real(E0mu_err)
	
	fig,axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2,1]})
	
	axs[0].loglog(k, E0r, label="$E_r$")
	axs[0].loglog(k, E0p, label=r"$E_\phi$")
	axs[0].loglog(k, E0mu, label=r"$E_\mu$")
	axs[0].legend()
	
	axs[1].loglog(k, E0r/E0r_err, label="$E_r$")
	axs[1].loglog(k, E0p/E0p_err, label=r"$E_\phi$")
	axs[1].loglog(k, E0mu/E0mu_err, label=r"$E_\mu$")
	axs[1].axhline(1, ls=':', c='k')
	axs[1].set_ylabel("|data/error|")
	axs[1].set_xlabel("k")
	
	cr_list_int = [int(cr) for cr in cr_list]
	fig.suptitle(f"CR: {min(cr_list_int)}–{max(cr_list_int)}")
	fig.set_size_inches(4,4)
	fig.tight_layout()
	
	return fig

if __name__ == "__main__":
	savefig = True
	savedir = "plots/Espec_componentwise"
	
	save = fig_saver(savefig, savedir)
	
	cr_bins = np.arange(2097,2268,10)
	for i in range(len(cr_bins)-1):
		cr_list = [f"{cr}" for cr in range(cr_bins[i], cr_bins[i+1])]
		figname = f"{cr_bins[i]}-{cr_bins[i+1]-1}.pdf"
		save(plot_Erpm_with_err(cr_list), figname)
