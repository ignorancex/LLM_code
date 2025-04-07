"""
Statistical measures of how well the signs of the helicity spectra calculated from HMI and SOLIS synoptic magnetograms are correlated.

1. signed correlation coefficient (like Spearman; 1 if same sign at a particular wavenumber; else 0)

2. chi-squared estimator, which is then used to calculate a p value

Also computes the above in restricted wavenumber bands, to check if perhaps, e.g., the helicity spectrum at large wavenumbers is more reliable.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
from dataclasses import dataclass


from read_FITS import HMIreader_dbl, SOLISreader_dbl as SOLISreader_dbl_exc, ExciseLatitudeMixin
from utils import fig_saver, smooth_boxcar
from plot_hel_with_err import E0H1_dbl

class HMIreader_dblexc(ExciseLatitudeMixin, HMIreader_dbl): pass

def calc_frachel(k, H_w_err, E_w_err):
	H, Herr = H_w_err
	E, Eerr = E_w_err
	
	frachel = k*H/E
	frachelerr = np.sqrt((Herr/H)**2 + (Eerr/E)**2)*frachel
	return frachel, frachelerr

def trunc(arr, k, k_min, k_max):
	if arr.ndim != 1: raise ValueError
	if k.ndim != 1: raise ValueError
	if len(arr) != len(k): raise ValueError
	
	ik_min = np.argmin(np.abs(k - k_min))
	ik_max = np.argmin(np.abs(k - k_max))
	return arr[ik_min:ik_max]

def sign_werr(H, Herr):
	"""
	Variant of np.sign that returns 0 if 0 is within the 1 sigma error region of H.
	
	H: np.ndarray.
	Herr: np.ndarray. Same size as H. Error in H.
	"""
	return np.where(Herr < abs(H), np.sign(H), 0)

@dataclass
class res_for_kmax:
	kmin: float
	kmax: float
	corr_sign_vs_cr: list
	corr_sign_werr_vs_cr: list
	chi2_vs_cr: list
	chi2r_vs_cr: list
	pval_vs_cr: list
	cr_labels: list

def calc_stats_for_kmax(k_bounds, cr_bins, read_HMI, read_SOLIS):
	"""
	k_bounds: tuple of float: (k,min, kmax)
	cr_bins: iterable of tuples; outer iterable indexes the bins, while the inner tuple is the list of CRs in each bin
	"""
	kmin, kmax = k_bounds
	
	corr_sign_list = []
	corr_sign_werr_list = []
	chi2_list = []
	chi2r_list = []
	pval_list = []
	cr_labels = []
	for cr_list in cr_bins:
		cr_labels.append(np.average(cr_list))
		
		res_h = E0H1_dbl(cr_list, read_HMI)
		res_s = E0H1_dbl(cr_list, read_SOLIS)
		
		k_min = max(res_h.k[1], res_s.k[1], kmin) #k[0]==0, so we ignore that
		k_max = min(res_h.k[-1], res_s.k[-1], kmax)
		
		trunc_h = lambda arr: trunc(arr, res_h.k[1:], k_min, k_max)
		trunc_s = lambda arr: trunc(arr, res_s.k[1:], k_min, k_max)
		
		E_s = trunc_s(res_s.E0[1:])
		Eerr_s = trunc_s(res_s.E0_err[1:])
		H_s = trunc_s(res_s.nimH1[1:])
		Herr_s = trunc_s(res_s.nimH1_err[1:])
		k_s = trunc_s(res_s.k[1:])
		
		E_h = trunc_h(res_h.E0[1:])
		Eerr_h = trunc_h(res_h.E0_err[1:])
		H_h = trunc_h(res_h.nimH1[1:])
		Herr_h = trunc_h(res_h.nimH1_err[1:])
		k_h = trunc_h(res_h.k[1:])
		
		if not np.all(np.isclose(k_s, k_h)):
			raise RuntimeError("Wavenumbers are unequal")
		
		corr_sign_list.append(
			np.average(np.sign(H_s)*np.sign(H_h)),
			)
		
		corr_sign_werr_list.append(
			np.average(sign_werr(H_s, Herr_s)*sign_werr(H_h, Herr_h)),
			)
		
		frachel_h, frachelerr_h = calc_frachel(k_h, (H_h, Herr_h), (E_h, Eerr_h))
		frachel_s, frachelerr_s = calc_frachel(k_s, (H_s, Herr_s), (E_s, Eerr_s))
		
		ndof = len(frachel_h)
		rv = scipy.stats.chi2(ndof)
		chi2 = np.sum((frachel_h - frachel_s)**2/(frachelerr_h**2 + frachelerr_s**2))
		
		chi2_list.append(chi2)
		chi2r_list.append(chi2/ndof)
		pval_list.append(1-rv.cdf(chi2))
	
	return res_for_kmax(
		kmin = k_min,
		kmax = k_max,
		corr_sign_vs_cr = corr_sign_list,
		corr_sign_werr_vs_cr = corr_sign_werr_list,
		chi2_vs_cr = chi2_list,
		chi2r_vs_cr = chi2r_list,
		pval_vs_cr = pval_list,
		cr_labels = cr_labels,
		)

if __name__ == "__main__":
	savefig = True
	savedir = "plots/correlation"
	mpl.style.use("kishore.mplstyle")
	
	save = fig_saver(savefig, savedir)
	
	read_HMI = HMIreader_dblexc(max_lat=60)
	read_SOLIS = SOLISreader_dbl_exc(max_lat=60)
	
	k_bounds_list = [(0,np.inf), (0,1e-1), (0,2e-2)]
	#Just like Singh 2018, we exclude certain Carrington rotations.
	cr_exclude = [2099, 2107, 2127, 2139, 2152, 2153, 2154, 2155, 2163, 2164, 2166, 2167, 2192, 2196]
	
	cr_bins = [tuple(cr for cr in range(cr_ini, cr_ini+10) if cr not in cr_exclude) for cr_ini in range(2097,2186)]
	
	res_list = [calc_stats_for_kmax(k_bounds, cr_bins, read_HMI, read_SOLIS) for k_bounds in k_bounds_list]
	c_list = mpl.cm.copper(np.linspace(0,1,len(res_list)))
	kwargs = {
		'marker': 'o',
		'markersize': 3,
		}
	
	fig, ax = plt.subplots()
	for res, c in zip(res_list, c_list):
		ax.plot(
			res.cr_labels,
			res.corr_sign_vs_cr,
			label=f"$k < {res.kmax:.2f}$ Mm$^{{-1}}$",
			color=c,
			**kwargs,
			)
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\sigma_\mathrm{sign}$")
	ax.set_ylim(-1,1)
	ax.legend()
	save(fig, "corr_sign.pdf")
	
	fig, ax = plt.subplots()
	for res, c in zip(res_list, c_list):
		ax.plot(
			res.cr_labels,
			res.corr_sign_werr_vs_cr,
			label=f"$k < {res.kmax:.2f}$ Mm$^{{-1}}$",
			color=c,
			**kwargs,
			)
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\sigma_\mathrm{sign}$")
	ax.set_ylim(-1,1)
	ax.legend()
	save(fig, "corr_sign_werr.pdf")
	
	fig, ax = plt.subplots()
	sm_hw = 4
	for res, c in zip(res_list, c_list):
		corr_sign_werr_vs_cr = smooth_boxcar(np.array(res.corr_sign_werr_vs_cr), sm_hw)[sm_hw:-sm_hw]
		cr_labels = res.cr_labels[sm_hw:-sm_hw]
		
		ax.plot(
			cr_labels,
			corr_sign_werr_vs_cr,
			label=f"$k < {res.kmax:.2f}$ Mm$^{{-1}}$",
			color=c,
			**kwargs,
			)
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\sigma_\mathrm{sign}$ (smoothed)")
	ax.set_ylim(-1,1)
	ax.legend()
	save(fig, "corr_sign_werr_sm.pdf")
	
	fig, ax = plt.subplots()
	for res, c in zip(res_list, c_list):
		ax.plot(
			res.cr_labels,
			res.chi2_vs_cr,
			label=f"$k < {res.kmax:.2f}$ Mm$^{{-1}}$",
			color=c,
			**kwargs,
			)
	ax.set_ylim(bottom=0)
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\chi^2$")
	ax.legend()
	save(fig, "chi2.pdf")
	
	fig, ax = plt.subplots()
	for res, c in zip(res_list, c_list):
		ax.plot(
			res.cr_labels,
			res.chi2r_vs_cr,
			label=f"$k < {res.kmax:.2f}$ Mm$^{{-1}}$",
			color=c,
			**kwargs,
			)
	ax.axhline(1, ls=':', c='k')
	ax.set_ylim(bottom=0)
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$\chi^2/n$")
	ax.legend()
	save(fig, "chi2r.pdf")
	
	ax.set_yscale('log')
	ax.autoscale(enable=True, axis='y')
	save(fig, "chi2r_log.pdf")
	
	fig, ax = plt.subplots()
	for res, c in zip(res_list, c_list):
		ax.plot(
			res.cr_labels,
			res.pval_vs_cr,
			label=f"$k < {res.kmax:.2f}$ Mm$^{{-1}}$",
			color=c,
			**kwargs,
			)
	# ax.set_yscale('log')
	ax.set_xlabel("Carrington rotation")
	ax.set_ylabel(r"$p$")
	ax.legend()
	save(fig, "pval.pdf")
