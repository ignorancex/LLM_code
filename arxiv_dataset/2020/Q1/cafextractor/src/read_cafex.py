import numpy as np
from astropy.io import fits
import os
import sys
import glob



class store_results:
  def __init__(self, obj, MYPATH=None):
	if MYPATH is not None:
		wpath = MYPATH
	else:
		wpath = os.getcwd()
	files = glob.glob(wpath+'/'+obj+'*.fits')
# 	hjd, snr, rv, erv, rvcorr, berv, texp = 7*[np.empty(len(files))*np.nan] #[], [], [], [], [], [], []
# 	telfocus, telfocus_scale = 2*[np.empty(len(files))*np.nan] #[],[]
	hjd, snr, rv, erv, rvcorr, berv, texp, telfocus, telfocus_scale, erv2 = np.zeros((10, len(files))).tolist() 
	fwhm, press = np.zeros(len(files)), np.zeros(len(files))
	
	for ii,file in enumerate(glob.glob(wpath+'/'+obj+'*.fits')):
		a = fits.open(file)
		hdr = a[0].header
		hjd[ii] 	= np.float(hdr["HIERARCH CAFEX HJD"])				if "CAFEX HJD" 	in hdr.keys() else np.nan
		erv[ii] 	= np.float(hdr["HIERARCH CAFEX ERV"]) 			if "CAFEX ERV" 	in hdr.keys() else np.nan
		erv2[ii] 	= np.float(hdr["HIERARCH CAFEX ERV2"]) 			if "CAFEX ERV2" 	in hdr.keys() else np.nan
		rv[ii] 		= np.float(hdr["HIERARCH CAFEX RV"] )				if "CAFEX RV" 	in hdr.keys() else np.nan
		snr[ii] 	= np.float(hdr["HIERARCH CAFEX SNR"]) 			if "CAFEX SNR" 	in hdr.keys() else np.nan
		berv[ii] 	= np.float(hdr["HIERARCH CAFEX BERV"]) 	if "CAFEX BERV" in hdr.keys() else np.nan
		texp[ii] 	= np.float(hdr["EXPTIME"]) 				if "EXPTIME" 	in hdr.keys() else np.nan
		rvcorr[ii] 	= np.float(hdr["HIERARCH CAFEX RVCORR"]) if "CAFEX RVCORR" in hdr.keys() else np.nan
		telfocus[ii] = np.float(hdr["HIERARCH CAHA TEL FOCU VALUE"]) if "CAHA TEL FOCU VALUE" in hdr.keys() else np.nan
		telfocus_scale[ii] = np.float(hdr["HIERARCH CAHA TEL FOCU SCALE"]) if "CAHA TEL FOCU SCALE" in hdr.keys() else np.nan
		fwhm[ii] 	= np.float(hdr["HIERARCH CAFEX CCF FWHM"]) 	if "CAFEX CCF FWHM" in hdr.keys() else np.nan
		press[ii] 	= np.float(hdr["PRESS1"]) 	if "PRESS1" in hdr.keys() else np.nan
	
	self.hjd 	= np.transpose(hjd)
	self.rv		= np.transpose(rv)		
	self.erv	= np.transpose(erv)	
	self.erv2	= np.transpose(erv2)	
	self.snr	= np.transpose(snr)	
	self.rvcorr	= np.transpose(rvcorr)	
	self.berv	= np.transpose(berv	)
	self.fwhm	= np.transpose(fwhm	)
	self.press	= np.transpose(press)
	self.texp	= np.transpose(texp	)
	self.telfocus	= np.transpose(telfocus	)
	self.telfocus_scale	= np.transpose(telfocus_scale)	

class spectrum(file):
  	def __init__(self, file, FULL_PATH=False):
		if FULL_PATH:
			mypath = file
		else:
			wpath = os.getcwd()
			mypath = wpath+'/'+file
		a = fits.open(mypath)
		
		self.wave	= a[2].data
		self.flux	= a[1].data	
		self.eflux	= a[3].data
		self.fnorm	= a[4].data
		self.efnorm	= a[5].data	
		self.wmerge	= a[6].data
		self.fmerge	= a[7].data
		#self.fmerge0= a[11].data
		self.dvel	= a[8].data
		self.ccf	= a[9].data
		self.eccf	= a[10].data
		self.head	= a[0].header
	
	def ploto(order):
		plt.plot(self.wave[order,:],self.flux[order,:])
		plt.show()
		plt.close()
	

def read_obj(obj,MYPATH=None):
	tmp = store_results(obj, MYPATH=MYPATH)
	return tmp

def read_spec(file,FULL_PATH=False):
	tmp = spectrum(file,FULL_PATH=FULL_PATH)
	return tmp
	

	

