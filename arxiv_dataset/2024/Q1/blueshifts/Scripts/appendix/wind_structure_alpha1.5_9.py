import astropy.io.ascii as io 
import astropy
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from astropy import constants as const
from astropy import units as u
import py_plot_util as util
import sys, os
import matplotlib
import py_read_output as rd 
from line_util import *
import jm_util
import string
import blueshift_util

jm_util.set_mod_defaults()

Alpha, AlphaString, data_dir, speclist = blueshift_util.get_filenames_for_alpha(sys.argv)

cmap_names = blueshift_util.cmap_dict

plt.figure()
angles_all = np.arange(5,90,5)
iplot = 0

plt.figure(figsize=(6,7))

fnames = [f for f in speclist]

rho2nh, _ = blueshift_util.rho2nh()

v90_all = np.zeros(len(fnames))
ia1 = 0
ia2 = 9
for f in fnames:
		full_fname = "{}/{}".format(data_dir, f[:-5])
		#print (full_fname)
		#os.system("windsave2table {}".format(full_fname))
		d = io.read(full_fname + ".master.txt")

		run_name = "Run\n{:d}".format(int(f[3:6]))

		#util.run_py_wind(full_fname, cmds=["1", "L", "0", "3", "s", "q"])
		_,_,lc4 = rd.read_pywind("{}.lineC4.dat".format(full_fname))
		_,_,vol = rd.read_pywind("{}.vol.dat".format(full_fname))


		plt.subplot(3,3,iplot+1)
		x, z, rho, _ = util.wind_to_masked(d, "rho", return_inwind=True, ignore_partial=True)
		x, z, c4, _ = util.wind_to_masked(d, "c4", return_inwind=True, ignore_partial=True)
		x, z, ne, _ = util.wind_to_masked(d, "ne", return_inwind=True, ignore_partial=True)
		x, z, vz, _ = util.wind_to_masked(d, "v_z", return_inwind=True, ignore_partial=True)
		x, z, vx, _ = util.wind_to_masked(d, "v_x", return_inwind=True, ignore_partial=True)
		#vz = np.sqrt(vz*vz + vx *vx)
		#plt.pcolormesh(x,z,np.log10(c4), vmax=0, vmin=-4)
		mu = 1.41
		print (rho2nh)
		ne = rho * rho2nh
		mappable = plt.scatter(np.log10(ne), np.log10(vz/1e5), c=np.log10(lc4), s=5, vmin=37.5, vmax=42.5, cmap="viridis")

		vz_flt = vz.flatten()
		iargs = np.argsort(vz)
		lc4_flt = lc4.flatten()
		lcf_cdf = np.cumsum(lc4_flt) / np.sum(lc4_flt)
		v90 = vz_flt[np.argmin(np.fabs(lcf_cdf-0.9))]/1e5
		print (string.ascii_lowercase[iplot], v90)
		v90_all[iplot] = v90
		#print (np.max(lc4), np.min(lc4), np.median(lc4))
		#plt.loglog()
		plt.grid(ls=":")
		if iplot not in [0,3,6]:
			plt.gca().set_yticklabels([])
		if iplot < 6:
			plt.gca().set_xticklabels([])
		if iplot == 6:
			plt.xlabel(r"$\log[n_H~({\rm cm}^{-3})]$", fontsize=20)
			plt.ylabel(r"$\log[v_z~({\rm km~s}^{-1})]$", fontsize=20)
		if iplot == 2 and AlphaString == "combo":
			plt.text(10.86,4.6,r"$\leftarrow \alpha = 0.5 | \alpha = 1 \rightarrow$", ha="center")
	

		plt.text(9, 3.7,run_name)
		if iplot == 1:
			plt.title(r"$\alpha = 1.5$")


	
		iplot+=1
		plt.xlim(5.2,11)
		plt.ylim(1.9,4.5)



#ax = plt.gcf().add_axes([0.62,0.05,0.2,0.03])
#cbar = plt.colorbar(mappable=mappable, cax = ax, orientation="horizontal", extend="both")
#plt.text(1.07,0.1,r"log[$L_{1550}$~(erg~s$^{-1}$)]", transform=ax.transAxes, fontsize=16)
plt.subplots_adjust(top=0.96, left=0.1, bottom=0.1, right=0.98, wspace=0.05, hspace=0.05)
figure_dir = os.path.abspath(os.path.join(os.path.dirname(__file__ ), '..', 'Figures'))

if Alpha == 0.5:
	plt.savefig("{}/wind_structure.pdf".format(figure_dir), dpi=300)
else:
	plt.savefig("{}/wind_structure_{}.pdf".format(figure_dir, AlphaString), dpi=300)

print (np.mean(v90_all), np.median(v90_all))