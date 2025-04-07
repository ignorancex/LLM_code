"""
Statistical measures of how well the signs of the helicity spectra calculated from HMI and SOLIS synoptic magnetograms are correlated.

1. signed correlation coefficient (like Spearman; 1 if same sign at a particular wavenumber; else 0)

2. chi-squared estimator, which is then used to calculate a p value

Also computes the above in restricted wavenumber bands, to check if perhaps, e.g., the helicity spectrum at large wavenumbers is more reliable.
"""

import sys
import pathlib
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
from dataclasses import dataclass
import sunpy.coordinates

from read_FITS import HMIreader_dbl, SOLISreader_dbl as SOLISreader_dbl_exc, ExciseLatitudeMixin
from utils import fig_saver, smooth_boxcar
from plot_hel_with_err import E0H1_dbl
from plot_correlation_HMI_SOLIS import calc_frachel, trunc, sign_werr, calc_stats_for_kmax

class HMIreader_dblexc(ExciseLatitudeMixin, HMIreader_dbl): pass

def add_time_axis_from_cr(ax):
	# https://stackoverflow.com/questions/39920252/second-plot-axis-with-different-units-on-same-data-in-matplotlib/64922723#64922723
	ax_date = ax.twiny()
	
	#Reduce the maximum number of ticks to avoid overlapping labels.
	loc = mpl.dates.AutoDateLocator(minticks=3, maxticks=5)
	ax_date.xaxis.set_major_locator(loc)
	
	def date_xlim_from_cr(ax):
		time_for_cr = lambda cr: sunpy.coordinates.sun.carrington_rotation_time(cr).to_datetime()
		xmin, xmax = ax.get_xlim()
		ax_date.set_xlim(time_for_cr(xmin), time_for_cr(xmax))
	
	ax.callbacks.connect('xlim_changed', date_xlim_from_cr)
	
	return ax_date

if __name__ == "__main__":
	savefig = True
	savedir = root/"plots/correlation_apj"
	mpl.style.use(root/"kishore_apj.mplstyle")
	mpl.rcParams['figure.figsize'] = (3.3,2.8) #A bit more height since we have two x-axes
	
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
	ax_date = add_time_axis_from_cr(ax)
	
	sm_hw = 4
	for res, c in zip(res_list, c_list):
		corr_sign_werr_vs_cr = smooth_boxcar(np.array(res.corr_sign_werr_vs_cr), sm_hw)[sm_hw:-sm_hw]
		cr_labels = res.cr_labels[sm_hw:-sm_hw]
		
		ax.plot(
			cr_labels,
			corr_sign_werr_vs_cr,
			label=f"{res.kmax:.2f}",
			color=c,
			**kwargs,
			)
	ax.set_xlabel("Carrington rotation")
	ax_date.set_xlabel("Year")
	ax.set_ylabel(r"$\sigma_\mathrm{sign}$ (smoothed)")
	ax.set_ylim(-1,1)
	ax.legend(title=r"$k_\mathrm{max}$ (Mm$^{{-1}}$)")
	save(fig, "corr_sign_werr_sm.pdf")
	
	fig, ax = plt.subplots()
	ax_date = add_time_axis_from_cr(ax)
	
	for res, c in zip(res_list, c_list):
		ax.plot(
			res.cr_labels,
			res.chi2r_vs_cr,
			label=f"{res.kmax:.2f}",
			color=c,
			**kwargs,
			)
	ax.axhline(1, ls=':', c='k')
	
	ax.set_yscale('log')
	ax.autoscale(enable=True, axis='y')
	
	ax.set_xlabel("Carrington rotation")
	ax_date.set_xlabel("Year")
	ax.set_ylabel(r"$\chi^2/n$")
	ax.legend(title=r"$k_\mathrm{max}$ (Mm$^{{-1}}$)")
	save(fig, "chi2r_log.pdf")
