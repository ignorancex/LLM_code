import os
import sys

import scipy
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks_cwt
from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import euclidean_distances#pairwise_distances
from astroML.stats import sigmaG
from astropy import constants as c
from astropy.table import Table, Column
from astropy.io import ascii
import matplotlib.gridspec as gridspec # GRIDSPEC !
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import GLOBALutils
import CAFEutilities
import CAFEx_SetupFile as CS
import rvtoolbox_arcs as rvtbx


# ========================================================================================
# 					Create ARC header
# ========================================================================================

def create_ArcHeader(raw_names, w_frames, arcs_cl, cv, type):
	tab  = np.load(cv.aux_dir+'arc_RVs.npz',allow_pickle=True)
	tabM = np.load(cv.aux_dir+'MasterArc_RVs.npz',allow_pickle=True)
	arcRVs 		= tab['arcRVs']
	MasterRVs 	= tabM['MasterRVs']
	Selected_MasterARC_type = CS.Selected_MasterARC
	Selected_MasterARC = Selected_MasterARC_type #arcs_cl[Selected_MasterARC_type+"ID"][0]

	headers = []

	for i,w_frame in enumerate(w_frames):
		filename, file_extension = os.path.splitext(raw_names[i])
		
		if type == "ARC":
			hdr = fits.getheader(cv.path_red+cv.dir_red+'/'+raw_names[i])
		elif type == "MasterARC":
			primary_hdu = fits.PrimaryHDU()
			hdr = primary_hdu.header
		
		npz = np.load(cv.aux_dir+'WC_'+filename+'.npz',allow_pickle=True)
		coeff, cov, WCdict = npz["coeff"], npz["cov"], npz["WCdict"].tolist()
		norders, npolyorder = np.shape(coeff)
		
		hdr["CAFEX NORDERS"] = (norders , "Order of polynomial")
		hdr["CAFEX WC NPOLY"] = (npolyorder , "Order of polynomial")
		for j in range(norders):
			value = '' 
			for aa,nn in enumerate(reversed(range(npolyorder))):
				value = value+str(coeff[j,nn])+' '
				hdr["CAFEX WCCOEFF ORDER"+str(j)+" A"+str(aa)] = (coeff[j,nn] , "Poly. coeff")

		for j in range(norders):
			hdr["CAFEX WCSTATS NLINES ORDER"+str(j)] = (WCdict["Nlines_used_order"][j], "Number of ThAr lines used in order") if np.isnan(WCdict["Nlines_used_order"][j]) == False else (' ', "Number of ThAr lines used in order")

		for j in range(norders):
			hdr["CAFEX WCSTATS LRESID ORDER"+str(j)] = (round(WCdict["lam_residuals_order"][j]*1.e3,2), "Residuals of WC in order [mA]")  if np.isnan(WCdict["lam_residuals_order"][j]) == False else (' ', "Residuals of WC in order [mA]")

		for j in range(norders):
			hdr["CAFEX WCSTATS VALID ORDER"+str(j)] = (WCdict["WC_validity"][j], "Validity range in order [A]")

		hdr["CAFEX WCSTATS LRESID"] = (WCdict['lam_residuals'], "Residuals disp. of WC [mA]")
		hdr["CAFEX WCSTATS NLINES"] = (WCdict['Nlines_total'], "Number of ThAr lines used")
		hdr["CAFEX WCSTATS RVBUDGET"] = (WCdict['RV_error_budget'], "RV error budget from WC [m/s]")
	
		if type != "MasterARC":
			hdr["CAFEX ARCRV"] = (arcRVs[0][i], "RV of ThAr frame [km/s]") 
			hdr["CAFEX ARCRV DIFF"] = (arcRVs[0][i]-MasterRVs[0][CS.Selected_MasterARC], "RV diff. with MasterArc [km/s]") 
		else:
			try:
				hdr["CAFEX ARCRV"] = (MasterRVs[0][i], "RV of ThAr frame [km/s]") 
			except:
				hdr["CAFEX ARCRV"] = ('nan', "RV of ThAr frame [km/s]") 	
	
		headers.append(hdr)
	

	
	return headers


# ========================================================================================
# 					Create SCIENCE header
# ========================================================================================


def create_SciHeader(raw_names, w_frames, WCdicts_sci, RVdicts, NORMdicts, cv):
	
	tab  = np.load(cv.aux_dir+'arc_RVs.npz',allow_pickle=True)
	tabM = np.load(cv.aux_dir+'MasterArc_RVs.npz',allow_pickle=True)
	arcRVs 		= tab['arcRVs']
	MasterRVs 	= tabM['MasterRVs']

	CS.var.set_OrderProp(CAFEutilities.jdnight(cv.night))
	headers = []

	for i,w_frame in enumerate(w_frames):
		filename, file_extension = os.path.splitext(raw_names[i])
		WCdict = WCdicts_sci[i]
		NORMdict = NORMdicts[i]
		RVdict = RVdicts[i]
		norders = CS.var.Nominal_Nord
		
		hdr = fits.getheader(cv.path_red+cv.dir_red+'/'+raw_names[i])
		
# 		npz = np.load(cv.aux_dir+'WC_'+filename+'.npz')
# 		coeff, cov, WCdict = npz["coeff"], npz["cov"], npz["WCdict"].tolist()
# 		norders, npolyorder = np.shape(coeff)
# 		
# 		# General properties of the Wavelength calibration
# 		hdr["CAFEX NORDERS"] = (norders , "Order of polynomial")
# 		hdr["CAFEX WC NPOLY"] = (npolyorder , "Order of polynomial")
# 		for j in range(norders):
# 			value = '' 
# 			for aa,nn in enumerate(reversed(range(npolyorder))):
# 				value = value+str(coeff[j,nn])+' '
# 				hdr["CAFEX WCCOEFF ORDER"+str(j)+" A"+str(aa)] = (coeff[j,nn] , "Poly. coeff")

		# Statistics of the Wavelength calibration
		for j in range(norders):
			hdr["CAFEX WCSTATS NLINES ORDER"+str(j)] = (WCdict["Nlines_used_order"][j], "Number of ThAr lines used in order") if np.isnan(WCdict["Nlines_used_order"][j]) == False else (' ', "Number of ThAr lines used in order")

		for j in range(norders):
			hdr["CAFEX WCSTATS LRESID ORDER"+str(j)] = (round(WCdict["lam_residuals_order"][j]*1.e3,2), "Residuals of WC in order [mA]")  if np.isnan(WCdict["lam_residuals_order"][j]) == False else (' ', "Residuals of WC in order [mA]")

		for j in range(norders):
			hdr["CAFEX WCSTATS VALID ORDER"+str(j)] = (WCdict["WC_validity"][j], "Validity range in order [A]")

		for j in range(norders):
			hdr["CAFEX SNR"+str(j)] = (NORMdict["SNR_o"][j], "S/N per pixel of order")

		hdr["CAFEX WCSTATS LRESID"] = (WCdict['lam_residuals'], "Residuals disp. of WC [mA]")
		hdr["CAFEX WCSTATS NLINES"] = (WCdict['Nlines_total'], "Number of ThAr lines used")
		hdr["CAFEX WCSTATS RVBUDGET"] = (WCdict['RV_error_budget'], "RV error budget from WC [m/s]")

		# Radial velocities of the Wavelength calibration and Science
		hdr["CAFEX ARCRV"] = (WCdict["RV_MasterARC"], "RV of ThAr Master frame [km/s]") 
		hdr["CAFEX ARCRV DIFF"] = (WCdict["RVinterpolated"], "ThAr RV at target obs. time [km/s]") 
		hdr["CAFEX ARCRV DRIFT"] = (WCdict["RVinterpolated"]-WCdict["RV_MasterARC"], "RV drift at target obs. time [km/s]") 
		
		# ===== Add header keywords
		berv, hjd, coord_flag = CAFEutilities.get_berv(hdr)
		hdr['CAFEX BERV'] = (berv, 'Barycentric Earth Rad. Vel. [km/s]')
		hdr['CAFEX COORDFLAG'] = (coord_flag, 'How target coordinates were computed for BERV')
		hdr['CAFEX HJD']  = (hjd , 'Heliocentric Julian Date [days]')
		hdr["CAFEX MOONCORR"] = (RVdict["Moon_corr"],'Moon correction for CCF')
		# RV
		if ~np.isnan(RVdict["RV"]):
			hdr['CAFEX RV']  = (RVdict["RV"] , 'Radial velocity (including corrections) [km/s]')
		else:
			hdr['CAFEX RV']  = ('nan' , 'Radial velocity [km/s]')
		
		# eRV
		if ~np.isnan(RVdict["eRV"]):
			hdr["CAFEX ERV"] = (RVdict["eRV"], 'Radial velocity uncertainty [km/s]')
		else:
			hdr["CAFEX ERV"] = ('nan', 'Radial velocity uncertainty [km/s]')
		
		# eRV2
		if ~np.isnan(RVdict["eRV2"]):
			hdr["CAFEX ERV2"] = (RVdict["eRV2"], 'Radial velocity uncertainty [km/s]')
		else:
			hdr["CAFEX ERV2"] = ('nan', 'Radial velocity uncertainty [km/s]')			

		try:
			if ~np.isnan(RVdict["CCFFWHM"]):
				hdr['CAFEX CCF FWHM']  = (RVdict["CCFFWHM"] , 'FWHM of CCF [km/s]')
			else:
				hdr['CAFEX CCF FWHM']  = ('nan' , 'FWHM of CCF [km/s]')

			if ~np.isnan(RVdict["CCFheight"]):
				hdr['CAFEX CCF HEIGHT']  = (RVdict["CCFheight"] , 'Normalised CCF height [-]')
			else:
				hdr['CAFEX CCF HEIGHT']  = ('nan' , 'Normalised CCF height [-]')

		except:
			print "No CCFFWHM found for file "+raw_names[i]
	
		
# 		if ~np.isnan(RVdict["eRV2"]):
# 			hdr["CAFEX ERV2"] = (RVdict["eRV2"], 'Radial velocity uncertainty from Boisse+2012 [km/s]')
# 		else:
# 			hdr["CAFEX ERV2"] = ('nan', 'Radial velocity uncertainty from Boisse+2012 [km/s]')			
		
		hdr["CAFEX SNR"] = (NORMdict["SNR"], 'S/N per pixel at 550nm')
		
		# ===== SNR-corrected RV
		snr = NORMdict["SNR"]
		corr_coeff = np.flip([5.07794635e-01, -1.03076535e-02,  4.72775655e-05],0)
		poly_corr = np.poly1d(corr_coeff)
		corr = poly_corr(snr)
		RVcorr = RVdict["RV"] - corr

		if ~np.isnan(RVcorr):
			hdr["CAFEX RVCORR"] = (RVcorr, 'RV - SNRcorr [km/s]')
		else:
			hdr['CAFEX RVCORR']  = ('nan' , 'RV - SNRcorr [km/s]')

		# ===== ThAr-change RV correction
		# Read ThAr lamp changes for RV correction
		thar_change_path = CS.RefFrames
		t = np.genfromtxt(thar_change_path+'/ThAr_LampChanges.dat',dtype=None)
		thar_jdstart, thar_jdend, ThAr_corr = t['f1'], t['f2'], t['f5']
		this_thar = np.where((thar_jdstart < hjd) & (thar_jdend > hjd))[0]
		corrThAr = ThAr_corr[this_thar]
		if hjd <  thar_jdstart[0]: 
			corrThAr = np.atleast_1d(np.nan)
			print "    --> WARNING!!!! No ThAr RV correction found in ReferenceCalib.dat for this date!!! "

		RVcorrThAr = RVdict["RV"] - corr - corrThAr[0]*1.e-3

		if ~np.isnan(RVcorrThAr):
			hdr["CAFEX RVCORR2"] = (RVcorrThAr, 'RV - SNRcorr - ThArcorr  [km/s]')
			hdr["CAFEX THARCORR"] = (corrThAr[0], 'ThArcorr correction [m/s]')
		else:
			hdr['CAFEX RVCORR2']  = ('nan' , 'RV - SNRcorr - ThArcorr [km/s]')
			hdr["CAFEX THARCORR"] = ('nan', 'ThArcorr correction [m/s]')


		# ===== Add header keywords
		hdr['CAFEX VERSION'] = (cv.version, 'Pipeline version')

	
		headers.append(hdr)
	

	
	return headers




# ==========================================
# 	ARCS - SAVE FINAL REDUCED IMAGE
# ==========================================

def save_ArcFrames(raw_names, w_frames, arcs_cl, cv, type):
	
	headers = create_ArcHeader(raw_names, w_frames, arcs_cl, cv, type)
	for i,w_frame in enumerate(w_frames):
		CAFEutilities.save_final_file(raw_names[i], w_frame, cv, type, myHeader=True, Header=headers[i])


def save_SciFrames(raw_names, w_frames, WCdicts, RVdicts, NORMdicts, MERGEdicts, cv):
	
	headers = create_SciHeader(raw_names, w_frames, WCdicts, RVdicts, NORMdicts, cv)

	for i,w_frame in enumerate(w_frames):
		#CAFEutilities.save_final_file(raw_names[i], w_frame, cv, type, myHeader=True, Header=headers[i])
		
		hdr = headers[i]
		RVdict = RVdicts[i]
		NORMdict = NORMdicts[i]
		MERGEdict = MERGEdicts[i]
		
		# Remove bad column 234
		w_frame[1,:,234] = np.nan
		w_frame[2,:,234] = np.nan
		NORMdict["fnorm"][:,234] = np.nan
		NORMdict["efnorm"][:,234] = np.nan
		
		# ===== Read wavelength file or wavelength matrix
		primary_hdu = fits.PrimaryHDU(header=hdr)
		flux  = fits.ImageHDU(data=w_frame[1,:,:], name="FLUX")
		eflux = fits.ImageHDU(data=w_frame[2,:,:], name="eFLUX")
		wave  = fits.ImageHDU(data=w_frame[3,:,:], name="WAVELENGTH")

		# ===== Additional extensions
		dvel  = fits.ImageHDU(data=RVdict["dvel"], name="CCF_vel")
		CCF   = fits.ImageHDU(data=RVdict["CCF"] , name="CCF")
		eCCF  = fits.ImageHDU(data=RVdict["eCCF"], name="eCCF")
		flux_norm  = fits.ImageHDU(data=NORMdict["fnorm"], name="FNORM")
		eflux_norm = fits.ImageHDU(data=NORMdict["efnorm"], name="eFNORM")
		wave_merge = fits.ImageHDU(data=MERGEdict["wmerge"], name="WMERGE1D")
		flux_merge  = fits.ImageHDU(data=MERGEdict["fmerge"], name="FMERGE1D")
		flux_merge0 = fits.ImageHDU(data=MERGEdict["fmerge0"], name="FMERGE1D0")

		# ===== Write file
		# Data
		hdul = fits.HDUList([primary_hdu, flux, wave, eflux, flux_norm, eflux_norm, wave_merge, flux_merge, dvel, CCF, eCCF, flux_merge0])

		# Filename
		rawfilename, file_extension = os.path.splitext(raw_names[i])
		filename = rawfilename+'_red.fits'
		hdul.writeto(cv.redfiles_dir+filename, overwrite=True)
		
		sci_S = w_frame.copy()
		sci_S[2,:,:] = sci_S[1,:,:]
		sci_S[1,:,:] = sci_S[3,:,:]
		sci_S = sci_S[:3,:,:]
		sci_S = np.einsum('kli->ikl', sci_S)
		hdu = pyfits.PrimaryHDU(sci_S)
		filename2 = rawfilename+'_rediraf.fits'
		# hdu.writeto(cv.redfiles_dir+filename2, overwrite=True)





