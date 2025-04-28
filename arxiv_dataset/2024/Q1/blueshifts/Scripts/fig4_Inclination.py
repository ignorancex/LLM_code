import astropy.io.ascii as io 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from astropy import constants as const
from astropy import units as u
import os
from tqdm import tqdm
import blueshift_util


def make_figure(alphas = ["0.5", "1", "1.5"]):

	print ("Making figure 4...")
	cmap_names = blueshift_util.cmap_dict 
	th_label = [70, 45, 20]


	fig_cdf, ax_cdf = plt.subplots(figsize=(6,5), nrows=1, ncols=1)

	colors = ["C0", "C1", "C2"]
	for ialpha,alpha_str in enumerate(alphas):
		Alpha, AlphaString, data_dir, speclist = blueshift_util.get_filenames_for_alpha(override=alpha_str)
		print (data_dir, alpha_str)

		fig, ax = plt.subplots(figsize=(6,5), nrows=1, ncols=1)
		all_blueshifts = []
		all_angles = []

		for itb, tb in enumerate(["thmin70", "thmin45", "thmin20"]):


			fnames = [f for f in os.listdir(data_dir) if ".spec" in f[-6:] and tb in f and "tau" not in f]
			norm, mappable = blueshift_util.get_norm_mappable(cmap=cmap_names[tb])
			ilabel = 0

			print (tb)

			for j,f in enumerate(tqdm(fnames)):

				try:
					s = io.read("{}/{}".format(data_dir, f))
				except io.ascii.core.InconsistentTableError:
					print ("Error for", f)

				wavelength = s["Lambda"]
				velocity = (wavelength - 1550.0) / 1550.0 * const.c.cgs 
				velocity = velocity.to(u.km/u.s)

				# read parameter file to get wind opening angles
				pf_root = f[:-5]
				pf = blueshift_util.read_pf("{}/{}".format(data_dir, pf_root))
				theta1 = float(pf["SV.thetamin(deg)"])
				theta2 = float(pf["SV.thetamax(deg)"])
				theta_b = 0.5 * (theta1 + theta2)
				launch = float(pf["SV.diskmin(units_of_rstar)"])

				if launch == 50:
					edgecolor="k"
				else:
					edgecolor="None"

				angles = np.arange(5,int(theta1)+5,5)

				blueshifts = np.zeros_like(angles, dtype=float)
				ews = np.zeros_like(angles, dtype=float)

				for i in range(len(angles)):
					colname = "A{:02d}P0.50".format(angles[i])

					fcont = blueshift_util.fit_continuum(wavelength, s[colname])

					# get blueshift 
					blueshifts[i] =  blueshift_util.get_blueshift(velocity/(u.km / u.s), s[colname], fcont)
					ews[i] = blueshift_util.get_ew(wavelength, velocity/(u.km / u.s), s[colname], fcont)


				if np.all(ews > 5):
					if ilabel == 0:
						label = r"$\theta_{{\rm min}} = {:.0f}^\circ$".format(th_label[itb])
						ilabel+=1
					else:
						label = None
					ax.plot(blueshifts, angles, color=mappable.to_rgba(angles[3]), label=label)
					ax.scatter(blueshifts, angles, facecolors=mappable.to_rgba(angles), edgecolors=edgecolor, s=80, alpha=1, zorder=3)

					for i,b in enumerate(blueshifts):
						all_blueshifts.append(b)
						all_angles.append(angles[i])

		ax.legend(frameon=False)
		ax.grid(ls=":")
		ax.set_xlabel(r"C~\textsc{iv}~Blueshift~(km~s$^{-1}$)", fontsize=16)
		ax.set_ylabel(r"$\theta_i~(^\circ)$", fontsize=16)
		ax.set_xlim(-2000,3000)
		figure_dir = os.path.abspath(os.path.join(os.path.dirname(__file__ ), '..', 'Figures'))
		fig.tight_layout(pad=0.05)
		if Alpha == 0.5:
			blueshift_util.save_paper_figure("fig4.pdf", fig=fig)
		else:
			blueshift_util.save_paper_figure("fig4_{}.pdf".format(AlphaString), fig=fig)

		angles_crit = np.arange(0,90,5)
		count = np.zeros_like(angles_crit)
		all_angles = np.array(all_angles)
		all_blueshifts = np.array(all_blueshifts)

		for i,a in enumerate(angles_crit):
			select = (all_angles <= a)
			count[i] = np.sum(all_blueshifts[select] > 500)

		ax_cdf.plot(angles_crit, count/count[-1], "-o", lw=4, c=colors[ialpha], label=r"$\alpha = {}$".format(alpha_str))
		ax_cdf.set_xlabel(r"$\theta~(^\circ)$", fontsize=20, labelpad=-1)
		ax_cdf.set_ylabel(r"Fraction of All Blueshifts at $\theta_i<\theta$", fontsize=18)
		fig_cdf.tight_layout(pad=0.05)
		ax_cdf.grid(ls=":")
		ax_cdf.set_xlim(0,70)
		ax_cdf.set_ylim(0,1.05)

	ax_cdf.legend(frameon=False)
	blueshift_util.save_paper_figure("fig4b.pdf", fig=fig_cdf)
	print ("Done.")

if __name__ == "__main__":
    blueshift_util.set_plot_defaults()
    make_figure()