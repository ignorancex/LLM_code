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

# CHOOSE YOUR FIGHTER
# by which I mean, choose which value of alpha to plot 
Alpha, AlphaString, data_dir, speclist = blueshift_util.get_filenames_for_alpha(sys.argv)

jm_util.set_mod_defaults()

cmap_names = blueshift_util.cmap_dict
angles_all = np.arange(5,90,5)
iplot = 0

plt.figure(figsize=(6,7))
ia1 = 0
ia2 = 9
for itb, tb in enumerate(["thmin20", "thmin45", "thmin70"]):
	#print (itb)

	norm, mappable = blueshift_util.get_norm_mappable(cmap=cmap_names[tb])

	#fnames = [f for f in os.listdir(data_dir) if ".spec" in f[-6:] and tb in f]
	fnames = [f for f in speclist if ".spec" in f[-6:] and tb in f]
	#print (fnames)
	for j,f in enumerate(fnames[:6]):


		file_present = True
		plt.subplot(3,3,iplot+1)
		
		run_name = "Run\n{:d}".format(int(f[3:6]))
		print (string.ascii_lowercase[iplot], f)

		try:
			print ("{}/{}".format(data_dir, f))
			s = io.read("{}/{}".format(data_dir, f))
		except FileNotFoundError:
			file_present = False
			print ("Error for", f)

		if file_present:
			pf_root = f[:-5]
			pf = rd.read_pf("{}/{}".format(data_dir, pf_root))
			theta1 = float(pf["SV.thetamin(deg)"])
			theta2 = float(pf["SV.thetamax(deg)"])
			theta_b = 0.5 * (theta1 + theta2)

			angles = np.arange(5,theta1,5)

			wavelength = s["Lambda"]
			velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs 
			velocity = velocity.to(u.km/u.s)


			#print (angles, string.ascii_lowercase[iplot])
			fmax = 0

			for i in range(len(angles)):
				colname = "A{:02d}P0.50".format(int(angles[i]))

				fcont = blueshift_util.fit_continuum(wavelength, s[colname])
				flux = savgol_filter(s[colname]/fcont, 15, 3)

				plt.plot(velocity/1e3, flux, c=mappable.to_rgba(angles[i]), lw=3)
				plt.xlim(-9.9,9.9)
				# plt.ylim(-0.5, 4)
				plt.ylim(0.8, 5)
				#plt.ylim(-0.05,0.6)
				# plt.semilogy()


		if iplot not in [0,3,6]:
			plt.gca().set_yticklabels([])
		if iplot < 6:
			plt.gca().set_xticklabels([])
		if iplot == 6:
			plt.xlabel("Velocity~($10^3$~km~s$^{-1}$)", fontsize=20)
			plt.ylabel(r"Normalised $F_\lambda$", fontsize=20)
		#if iplot == 2 and AlphaString == "combo":
		#	plt.text(10.86,4.6,r"$\leftarrow \alpha = 0.5 | \alpha = 1 \rightarrow$", ha="center")
	

		if iplot == 1:
			plt.title(r"$\alpha = 1.5$")
		#elif iplot == 3:
		#	plt.text(-8,5.2,r"$\alpha = 0.5 \rightarrow$")

		plt.gca().axvline(0, ls="--", alpha=0.5, color="k")
		iplot+=1

		plt.text(-9,3.5,run_name)


plt.subplots_adjust(top=0.96, left=0.1, bottom=0.1, right=0.98, wspace=0.05, hspace=0.05)
figure_dir = os.path.abspath(os.path.join(os.path.dirname(__file__ ), '..', 'Figures'))

if Alpha == 0.5 or Alpha == -1:
	plt.savefig("{}/fig3.pdf".format(figure_dir), dpi=300)
else:
	plt.savefig("{}/fig3_{}.pdf".format(figure_dir, AlphaString), dpi=300)