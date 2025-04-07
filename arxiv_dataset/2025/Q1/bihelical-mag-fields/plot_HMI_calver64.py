"""
How the calibration version used in the HMI synoptic maps files changes with Carrington rotation.
Documented at http://jsoc.stanford.edu/jsocwiki/CalibrationVersions
"""

import pathlib
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib as mpl

from utils import fig_saver

class Calver64:
	def __init__(self, calver):
		"""
		Height-of-formation code version used to find disk center. Vers 2 for all data to date.
		"""
		self.hfcorrvr = (calver &  0x0F)
		
		"""
		Version of CROTA2 in the Master pointing table. Vers 0 prior to Transit of Venus, then 1.
		"""
		self.crota2vr = (calver & 0xF0)//2**4
		
		"""
		If > 0, then smooth look-up tables were used to produce observables 
		"""
		self.lookup = (calver & 0xF00)//2**8
		
		"""
		If > 0, then a correction for non-linearity of the CCDs was applied.
		If 0x1000, the polynomial coefficients used for this correction are: -8.2799134,0.017660396,-3.7157499e-06,9.0137137e-11 (for the side camera) and -11.081771,0.017383740,-2.7165221e-06,6.9233459e-11 (for the front camera).
		If 0x2000, the polynomial coefficients used for this correction are: 0.0,0.025409177,-4.0088672e-06,1.0615198e-10 (side camera) and 0.0,0.020677687,-3.1873243e-06,8.7536678e-11 (front camera) 
		"""
		self.nonlin = (calver & 0xF000)//2**12
		
		"""
		If 0, then observing sequence was 'mod C'; mod C was the standard observing sequence before 2016.04.13_19:12:55.11_UTC, FSN=104683793. 
		If 2 then original deprecated Mod L processing 
		If 3 then mod-L misalignment corrected 
		If 4 then misalignment and filtergram selection correct 
		"""
		self.obs_seq = (calver & 0xF0000)//2**16
		
		"""
		If > 0 then PSF/scattered light deconvolution has been done, 1==CUDA version, 2== C version 
		"""
		self.psf = (calver & 0xF00000)//2**20
		
		"""
		If > 0, then a rotational flat field was used. The code in HMI_observables and HMI_IQUVaveraging checks only on the bit 0x1000000 (originally field 4 was used for this, changed in Nov 2016) 
		"""
		self.flat_field = (calver &  0xF000000)//2**24
		
		"""
		If > 0, then the observer location keywords have been updated. 0x1, bit 28, means CRLN_OBS and HGLN_OBS if present have been corrected.
		"""
		self.obs = (calver & 0xF0000000)//2**28

if __name__ == "__main__":
	savefig = True
	savedir = "plots/calibration_info/HMI"
	mpl.style.use("kishore.mplstyle")
	img_loc = pathlib.Path("images")
	cr_list = list(range(2097,2196))
	
	save = fig_saver(savefig, savedir)
	
	calver_list = []
	for cr in cr_list:
		fname = img_loc/f"hmi.b_synoptic_small.rebinned.{cr}.Br.fits"
		with fits.open(fname) as f:
			hdu = f[0]
			calver_list.append(
				Calver64(hdu.header['CALVER64']),
				)
	
	attr_names = ["nonlin", "lookup", "psf", "flat_field"]
	fig,axs = plt.subplots(nrows=len(attr_names), sharex=True)
	for i, name in enumerate(attr_names):
		axs[i].plot(cr_list, [getattr(c, name) for c in calver_list])
		axs[i].set_ylabel(name)
	
	plt.show()
