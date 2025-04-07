"""
Calibration keywords of the SOLIS maps as a function of Carrington rotation.
"""

import pathlib
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib as mpl

from read_FITS import m_get_fname_SOLIS

from utils import fig_saver

class CalibrationInfoSOLIS:
	def __init__(self, filename):
		with fits.open(filename) as f:
			hdu = f[0]
			header = hdu.header
		
		self.ver0 = header['PROVER0']
		self.idlver_lev2 = header['PROVER1A']
		self.idlver_cameragap = header['PROVER1B']
		self.idlver_distortion = header['PROVER1C']
		self.idlver_normalize = header['PROVER1D']
		self.data_version = header['VERSION']
		self.camtype = header['CAMTYPE']

if __name__ == "__main__":
	savefig = True
	savedir = "plots/calibration_info/SOLIS"
	mpl.style.use("kishore.mplstyle")
	cr_list = list(range(2097,2196))
	
	name_getter = m_get_fname_SOLIS()
	
	save = fig_saver(savefig, savedir)
	
	crs = []
	calver_list = []
	
	for cr in cr_list:
		try:
			filename = name_getter.get_fname(cr)
		except RuntimeError:
			#Could not get the filename (there is probably no corresponding file)
			continue
		crs.append(cr)
		calver_list.append(CalibrationInfoSOLIS(filename))
	
	fig,ax = plt.subplots()
	ax.plot(crs, [str(c.camtype) for c in calver_list])
	
	plt.show()
