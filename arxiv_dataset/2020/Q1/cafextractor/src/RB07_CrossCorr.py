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
import ntpath

import GLOBALutils
import CAFEutilities
import CAFEx_SetupFile as CS
import rvtoolbox_arcs as rvtbx


# ========================================================================================
# 										GET RV
# ========================================================================================

def get_RV(sp,inst, wmask, sel_orders=-10, guessRV=True, with_Moon = False, plot_name='tmp.pdf'):

	# Orders exclude (show no relevant information and possible blended intense lines)
	#exclude_orders = [27,29,30,31,50,52,55,61,69,72,74,83,   44,28,41,46,34,79,47,39,26,25,75,42,45]
	exclude_orders = np.array([27,29,30,32,40,41,44,46,50,53,54,71])-2

	# Data properties
	nexten,norders,npix = np.shape(sp)
	sel_orders = np.arange(norders-23)+23
	
	_sel_orders_list = list(sel_orders)
	for i in exclude_orders: _sel_orders_list.remove(i)
	sel_orders = np.array(_sel_orders_list)
		
	
	wave = sp[3,:,300:-300]
	flux = sp[1,:,300:-300]
	eflux = sp[2,:,300:-300]

	# Mask
	fmask = wmask*0.0 + 1.0
	fwhm_mask = wmask*0.0 + 4.0

	# ==============================
	# 	RV first guess
	# ==============================
	if guessRV == True:
		dvel, vwidth = rvtbx.create_dvel(inst,wave,RVguess=0.0,RVampl=200.,verbose=True)
		if norders > 1:
			test_order = 36
			w, f, ef 	= wave[test_order,:], flux[test_order,:], eflux[test_order,:]
		else:
			w, f, ef 	= wave, flux, eflux 

		CCFo,eCCFo = rvtbx.CCF(w,f,ef,dvel,vwidth, wmask, fmask)
		popt, perr = rvtbx.fit_CCF(dvel,CCFo,eCCFo,guessRV=False, with_Moon = False)
		RVguess = popt[1]
	else:
		RVguess = 0.0
	print "Estimated RV (no BERV corrected) = ",np.str(round(RVguess,3))," km/s"

	# ==============================
	# 	Cross-Correlation Funtion
	# ==============================

	CCFall, eCCFall 		= [], []
	RVall, eRVall 			= [], []
	FWHMall, eFWHMall 		= [], []
	Heightall, eHeightall 	= [], []
	snrall 					= []
	dvel, vwidth = rvtbx.create_dvel(inst,wave,RVguess=RVguess,RVampl=15.)

	print "Calculating CCF..."

	# Calculate CCF per order
	# pbar = ProgressBar()
	for jj,oo in enumerate(sel_orders): 	#pbar(sel_orders):

		if jj == 0:
			wcut_end = (wave[oo,-1]+wave[oo+1,0])/2.
			no_overlap = np.where(wave[oo,:] > wcut_end)[0]
		elif jj == len(sel_orders)-1:
			wcut_start = (wave[oo-1,-1]+wave[oo,0])/2.
			no_overlap = np.where(wave[oo,:] < wcut_start)[0]
		else:
			wcut_end = (wave[oo-1,-1]+wave[oo,0])/2.
			wcut_start = (wave[oo,-1]+wave[oo+1,0])/2.
			no_overlap = np.where((wave[oo,:] > wcut_start) & (wave[oo,:] < wcut_end) )[0]

		w,f, ef = wave[oo,no_overlap], flux[oo,no_overlap], eflux[oo,no_overlap]
		
		if ((np.count_nonzero(f) !=0) & (np.count_nonzero(~np.isnan(f)) != 0)):
			CCFo,eCCFo   = rvtbx.CCF(w,f,ef,dvel,vwidth, wmask, fmask, CRAYS=False)
			popto, perro = rvtbx.fit_CCF(dvel,CCFo,eCCFo, guessRV=guessRV, with_Moon = with_Moon)
		else:
			CCFo,eCCFo = np.zeros(len(dvel)), np.zeros(len(dvel))
			popto, perro = np.zeros(4), np.zeros(4)
		
		# Append results
		CCFall.append(CCFo)	
		eCCFall.append(eCCFo)	
		RVall.append(popto[1])	
		eRVall.append(perro[1])	
		FWHMall.append(popto[2]*2.*np.sqrt(2.*np.log(2.)))	
		eFWHMall.append(perro[2]*2.*np.sqrt(2.*np.log(2.)))	
		Heightall.append(popto[0])	
		eHeightall.append(perro[0])	

	
	
	CCFall, eCCFall = np.array(CCFall), np.array(eCCFall)
	RVall, eRVall 	= np.array(RVall), np.array(eRVall)
	#for ttt in range(len(sel_orders)): print sel_orders[ttt],RVall[ttt],eRVall[ttt] 

	# ===== CCF stacking of all orders
	CCF  = np.nansum(CCFall,axis=0)
	eCCF = np.sqrt(np.nansum(eCCFall**2,axis=0))

	# ===== Gaussian fit to the stacked CCF
	print "Measuring RV from CCF..."
	popt, perr = rvtbx.fit_CCF(dvel,CCF,eCCF, with_Moon = with_Moon)

	# ===== RV results and corrections

	# RV from CCF
	RV 		= popt[1]
	eRV 	= perr[1]

#	print RV, eRV
# 	plt.plot(dvel,CCF)
# 	plt.plot(dvel,rvtbx.gaussfit(dvel,*popt),ls='--',c='red')
# 	for i in range(len(sel_orders)): plt.plot(dvel,CCFall[i,:])
# 	np.savez('tmp_ccf',dvel=dvel,CCFall=CCFall,CCF=CCF)
# 	plt.show()
# 	sys.exit()

	# ===== RV uncertainty from Boisse et al. (2010)
	deriva = np.gradient(CCF,dvel)
	Nscale = 0.25 # pix # np.sqrt(np.nanmean(np.diff(dvel))   )
	Q_CCF = np.sqrt(np.nansum(deriva**2/CCF)) / np.sqrt(np.nansum(CCF)) * np.sqrt(Nscale)
	eRV2 = 1./(Q_CCF*np.sqrt(np.nansum(CCF)))


	RVdict =  { 'CCF':CCF,
				'eCCF':eCCF,
				'dvel':dvel,
				'Moon_corr':'no',
				'RVall':RVall,
				'eRVall':eRVall,
				'FWHMall':FWHMall,
				'eFWHMall':eFWHMall,
				'Heightall':Heightall,
				'eHeightall':eHeightall,
				'RV':RV,
				'eRV':eRV,
				'eRV2':eRV2
				}


	try:
		fig = plt.figure(figsize=(12,8))
		gs = gridspec.GridSpec(3,2, height_ratios=[1.,1.,1.], width_ratios=[1,1])
		gs.update(left=0.1, right=0.95, bottom=0.08, top=0.93, wspace=0.08, hspace=0.08)
		# CCF
		ax1 = plt.subplot(gs[:,0]) 
		plt.errorbar(dvel, CCF/popt[3], eCCF/popt[3])	
		if with_Moon == False:
			plt.plot(dvel,rvtbx.gaussfit(dvel,*popt)/popt[3],c='red')
		else:
			plt.plot(dvel,rvtbx.gaussfit_Moon(dvel,*popt)/popt[3],c='red')
		plt.axvline(0.0,ls=':',c='gray',alpha=0.5)
		plt.axvline(RV,ls=':',c='red',alpha=0.8)		
		plt.xlabel('Radial velocity (km/s)')
		plt.ylabel('sum(CCF_o*S/N_o)')
		# RV per order
		ax2 = plt.subplot(gs[0,1])
		plt.errorbar(sel_orders,RVall,yerr=eRVall,fmt='o',c='Dodgerblue')
		plt.axhline(RV,ls=':',c='k')
		plt.ylabel('RV (km/s)')
		# CCF height
		ax3 = plt.subplot(gs[1,1])
		plt.errorbar(sel_orders,Heightall,yerr=eHeightall,fmt='o',c='Tomato')
		plt.axhline(popt[0]/popt[3],ls=':',c='k')
		plt.ylabel('CCF height (normalized)')
		# CCF FWHM
		ax4 = plt.subplot(gs[2,1])
		plt.errorbar(sel_orders,FWHMall,yerr=eFWHMall,fmt='o',c='green')
		plt.axhline(popt[2]*2.*np.sqrt(2.*np.log(2.)),ls=':',c='k')
		plt.ylabel('FWHM (km/s)')

		plt.savefig(plot_name)
		plt.close()
	except:
		print "No RV plot for this frame..."

	
	return RV, eRV, popt, perr, RVall, eRVall, eRV2, RVdict


# ========================================================================================
# 										GET RV
# ========================================================================================


def ArcRV(frames, cv, frame_names):
	
	# ===== ThAr mask
	
	# From CERES:
	#ThArMask = np.genfromtxt(cv.ref_frames+'ThAr_ReferenceLines/all_ThAr_lines.lis',dtype=None)
	#wmask = ThArMask["f2"]
	
	# From Lovis+2007:
# 	ThArMask = np.genfromtxt(cv.ref_frames+'ThAr_ReferenceLines/ThAr_Lovis07.txt',dtype=None)
# 	wmask_vaccuum = ThArMask["f0"]
# 	s = 1.e4/wmask_vaccuum
# 	n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
# 	wmask = wmask_vaccuum/n
	
	ThArMask = np.genfromtxt('ThAr_for_RV.dat',dtype=None,names=True)
	wmask = ThArMask["wmask"]

	inst = 'CAFE'
	
	RVguess = 0.0	
	
	RVs = []
	eRVs = []

	for i,frame in enumerate(frames):
		RV, eRV, popt, perr, _, _, _, _ = get_RV(frame, inst, wmask, guessRV = False, with_Moon = False, plot_name=cv.aux_dir+'/RV_'+frame_names[i]+'.pdf')
		RVs.append(RV)
		eRVs.append(eRV)
		print RV,eRV
			

	return RVs,eRVs


# ========================================================================================
# 					ARCS: Attach Wavelength Calibration
# ========================================================================================


def AttachWC_arcs(x_arc,arcs,arc_names,					# Individual arc frames
			 w_MasterArc, arcs_cl, MasterRVs, cv):		# Master Arc frames
	
	# ===== Preparing the correlation:
	# ThAr lines mask from Lovis+2007:
# 	ThArMask = np.genfromtxt(cv.ref_frames+'ThAr_ReferenceLines/ThAr_Lovis07.txt',dtype=None)
# 	wmask_vaccuum = ThArMask["f0"]
# 	s = 1.e4/wmask_vaccuum
# 	n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
# 	wmask = wmask_vaccuum/n
	
	ThArMask = np.genfromtxt('ThAr_for_RV.dat',dtype=None,names=True)
	wmask = ThArMask["wmask"]
		
	inst = 'CAFE'
	RVguess = 0.0	

	# ===== Loop for each frame
	SS_frame = []
	WCdicts_arc = []
	
	#tab = np.load(cv.aux_dir+'tmp__w_arcs.npz')
	#WCdict_arc = tab['WCdict_arc'].tolist()
	#RVs = []
	#for dict in WCdict_arc: RVs.append(dict["RVfromMaster"])

	for i,arcFr in enumerate(x_arc):
		
		# ===== Science RV against evening MasterArc:
		Selected_MasterARC = CS.Selected_MasterARC	# For now, I use the first MasterArc
													# (usually the evening combined frame). But this
													# must be improved by selecting the closest or the
													# best quality MasterArc.  
						
		# Read wavelength file
		wtable = fits.open(cv.aux_dir+'/WC_MasterArc_'+str(Selected_MasterARC)+'.fits')
		w = wtable[0].data[3,:,:]

		arcFr = np.einsum('kli->lki', arcFr)
		wnew = np.atleast_3d(w)
		wnew = np.einsum('kli->ikl', wnew)
		w_arc_tmp = np.append(arcFr,wnew,axis=0)
		
		# ===== Measure the RV
		already_done = os.path.isfile(cv.aux_dir+'RVdict_'+arc_names[i]+'.npz')
		if already_done == False:
			RV, eRV, popt, perr, _, _, eRV2, _ = get_RV(w_arc_tmp, inst, wmask, guessRV = False, with_Moon = False, plot_name=cv.aux_dir+'/RV_'+arc_names[i]+'.pdf')
			np.savez(cv.aux_dir+'RVdict_'+arc_names[i],RV=RV,eRV=eRV,eRV2=eRV2)
		else:
			print "    --> RVdict found for "+arc_names[i]+". Loading..."
			load_res = np.load(cv.aux_dir+'RVdict_'+arc_names[i]+'.npz',allow_pickle=True)
			RV = load_res["RV"]
			eRV = load_res["eRV"]
			eRV2 = load_res["eRV2"]
		
		
		
		#RV = RVs[i]
		
		# ===== Attach RV-corrected wavelength solution:
		RVdrift = -(RV-MasterRVs[0][Selected_MasterARC])
		wnew = w*(1.+RVdrift/(c.c.value*1.e-3))
		wnew = np.atleast_3d(wnew)
		wnew = np.einsum('kli->ikl', wnew)
		w_arc = np.append(arcFr,wnew,axis=0)

		SS_frame.append(w_arc)
		
		#CAFEutilities.save_final_file(arc_names[i], w_arc, cv, "ARC")
		
		 
		WCdict_arc={'MasterARC':'MasterArc_'+str(Selected_MasterARC),
					'RV_MasterARC':MasterRVs[0][Selected_MasterARC],
					'RVdrift':RVdrift*1.e3,
					'RVfromMaster':RV,
					'eRVfromMaster':eRV,
					'eRVBoisse':eRV2,
					'jd':arcs["jd"][i]
					}
		WCdicts_arc.append(WCdict_arc)
	

	return SS_frame, WCdicts_arc

# ========================================================================================
# 					SCIENCE: Attach Wavelength Calibration
# ========================================================================================


def AttachWC(x_sci,sci,sci_names,						# Science frames
			 w_arc, arcs, arcRVs, arc_names, 			# Individual arc frames
			 w_MasterArc, arcs_cl, MasterRVs, cv):		# Master Arc frames
	
	# ===== ThAr mask
	SS_frame = []
	WCdicts_sci = []
	for i,sciFr in enumerate(x_sci):
		
		# ===== Science RV against evening MasterArc:
		good = np.where(np.isfinite(arcRVs) == True)[0]
		jd_arc, rv_arc = arcs["jd"][good], arcRVs[0][good]
		sciRV = np.interp(sci["jd"][i], jd_arc, rv_arc)
		Selected_MasterARC_type = CS.Selected_MasterARC	# For now, I use the first MasterArc
														# (usually the evening combined frame). But this
														# must be improved by selecting the closest or the
														# best quality MasterArc.  
		
		Selected_MasterARC = Selected_MasterARC_type #arcs_cl[Selected_MasterARC_type+"ID"][0]
		RV_master = MasterRVs[0][Selected_MasterARC]  
		RVdiff = sciRV-RV_master
				
		# Read wavelength file
		wtable = fits.open(cv.aux_dir+'/WC_MasterArc_'+str(Selected_MasterARC)+'.fits')
		w = wtable[0].data[3,:,:]
		RVcorr =  - (sciRV + RV_master) #- RVdiff - RV_master
		wnew = w*(1.+RVcorr/(c.c.value*1.e-3))

		sciFr = np.einsum('kli->lki', sciFr)
		wnew = np.atleast_3d(wnew)
		wnew = np.einsum('kli->ikl', wnew)
		w_sci = np.append(sciFr,wnew,axis=0)
		SS_frame.append(w_sci)
		
		CAFEutilities.save_final_file(sci_names[i], w_sci, cv, "SCI")

		try:
			arc_before = arc_names[np.argmax(arcs["jd"][arcs["jd"] < sci["jd"][i]])]
		except:
			arc_before = '--'

		
		try:
			arc_after  = arc_names[np.argmax(arcs["jd"][arcs["jd"] < sci["jd"][i]]) + 1]
		except:
			arc_after = '--'
			
		
		# Get info from the used MasterArc:
		t = np.load(cv.aux_dir+'/WC_MasterArc_'+str(Selected_MasterARC)+'.npz',allow_pickle=True)
		coeff, cov, WCdict_MasterArc = t["coeff"], t["cov"], t["WCdict"].tolist()
		 
		WCdict_sci={'MasterARC':'MasterArc_'+str(Selected_MasterARC),
					'RV_MasterARC':RV_master,
					'RVinterpolated':sciRV,
					'Arc_before':arc_before,
					'Arc_after':arc_after,
					}
		# Combine the sci and MasterArc dictionaries
		
		z = WCdict_sci.copy() 
		z.update(WCdict_MasterArc)
		
		# Append dictionary
		WCdicts_sci.append(z)
	
	return SS_frame, WCdicts_sci

# ========================================================================================
# 					ARCS: plot RVs
# ========================================================================================

def plot_ArcRV(sci, arcs, arcRVs, arcs_cl, MasterRVs, cv):

	plt.close()
	fig = plt.figure(figsize=(12,8))
	gs = gridspec.GridSpec(2,2, height_ratios=[1.,1.], width_ratios=[1,1])
	gs.update(left=0.1, right=0.95, bottom=0.08, top=0.93, wspace=0.08, hspace=0.08)
	ax1 = plt.subplot(gs[0,:])
	for ii,i in enumerate(sci["jd"]): 
		plt.axvline(i,ls=':',c='gray',alpha=0.4, zorder=0)
		head, tail = ntpath.split(sci["files"][ii])
		name, _ = tail.split('__')
		plt.text(i,np.min(arcRVs[0]), name, rotation=90, color='gray',alpha=0.4, fontsize=10, verticalalignment='bottom', zorder=1)
	good = np.where(np.isfinite(arcRVs) == True)[0]
	plt.errorbar(arcs["jd"][good],arcRVs[0][good],yerr=arcRVs[1][good],fmt='o',c='red', zorder=5)
	plt.errorbar(arcs_cl["centers"],MasterRVs[0],yerr = MasterRVs[1],fmt='o',c='blue',zorder = 10)
	plt.xlabel('JD')
	plt.ylabel('RV (km/s)')
	
	ax2 = plt.subplot(gs[1,1])
	elem = np.where((arcs_cl["membership"] == 0) & (np.isfinite(arcRVs) == True) )[0]
	plt.errorbar(arcs["jd"][elem],arcRVs[0][elem],yerr=arcRVs[1][elem],fmt='o',c='red', zorder=5)

	ax3 = plt.subplot(gs[1,0])
	elem = np.where((arcs_cl["membership"] == 1) & (np.isfinite(arcRVs) == True))[0]
	plt.errorbar(arcs["jd"][elem],arcRVs[0][elem],yerr=arcRVs[1][elem],fmt='o',c='red', zorder=5)

	plt.savefig(cv.aux_dir+'/arcRVs_'+cv.night+'.pdf',bbox_inches='tight')
	plt.close()





