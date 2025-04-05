import sys
import os

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
from matplotlib.pyplot import cm

import GLOBALutils
import CAFEutilities
import CAFEx_SetupFile as CS
import rvtoolbox as rvtbx

import RB07_CrossCorr as RB07

# ========================================================================================
# 										GET RV
# ========================================================================================

def get_RV(sp,inst, cv, sel_orders=-10, guessRV=True, myRVguess=0.0, with_Moon = False, plot_name='tmp.pdf'):

	
	CS.var.set_OrderProp(CAFEutilities.jdnight(cv.night))
	
	# Orders exclude (CARMENES-wise)
	# exclude_orders = [29, 30, 38, 39, 43, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

	# Data properties
	nexten,norders,npix = np.shape(sp)
	#sel_orders = np.array([11,12,13,16,17,18,20,21,24,25,26,28,29,30,31,32,33,34,35,36,39,40,41,42,43,44,45,46,48,49,51,52,53,54,55,56,57,58,59,60,61,63,64,65,66,67,68,69,70,71,73,74,75,76,77,78,79]	)# np.arange(norders)
	#sel_orders = np.array([25,26,29,30,31,32,33,34,35,36,39,40,41,42,43,44,45,46,48,49,51,52,53,54,55,56,57,58,59,60,61,63,64,65,66,67,68,69]	)# np.arange(norders)
	
	order_offset = CS.var.order0 - 60
	exclude_orders = np.array([27,28,30,32,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83])-order_offset#,79,80,81,82,83])
	
 	sel_orders = np.arange(norders-(24-order_offset))+(24-order_offset)
	_sel_orders_list = list(sel_orders)
	for i in exclude_orders: 
		try:
			_sel_orders_list.remove(i)
		except:
			ddd = 0
	sel_orders = np.array(_sel_orders_list)


# 	sel_orders = np.delete(sel_orders, exclude_orders) 
# 	print sel_orders
# 	sys.exit()

	
	wave = sp[3,:,300:-300]
	flux = sp[1,:,300:-300]
	eflux = sp[2,:,300:-300]
	
	# Mask (combination)
# 	mask = np.genfromtxt('mask_v05.dat',dtype=None,names=True)
# 	wmask = mask["WavelengthA"]
# 	fmask = mask["heightnorm"]
# 	fwhm_mask = mask["FWHMkms"]
# 	no_broad = np.where(fwhm_mask < 10.)
# 	wmask, fmask = wmask[no_broad], fmask[no_broad]
# 	
# 	harps_mask = np.genfromtxt(CS.RefFrames+'G2.mas',dtype=None,names=True)
# 	wmaskH = (harps_mask["W0"]+harps_mask["W1"])/2.
# 	fmaskH = harps_mask["Weight"]
# 	widthH = (wmaskH-harps_mask["W0"])/wmaskH * c.c.value*1.e-3
# 	
# 	me_elem = np.where(wmask > 6798.55)[0]
# 	wmaskJ, fmaskJ = wmask[me_elem], fmask[me_elem]
# 	
# 	me2 = np.where((wmask > 5867.59) & (wmask < 6002.98))
# 	wmaskJ2, fmaskJ2 = wmask[me2], fmask[me2]
# 	
# 	wmask = np.concatenate((wmaskH,wmaskJ,wmaskJ2))
# 	fmask = np.concatenate((fmaskH,fmaskJ,fmaskJ2))
# 	id = np.concatenate((np.array(["H"*len(wmaskH)]),np.array(["J"*len(wmaskJ)]),np.array(["J"*len(wmaskJ)])))

	# Mask (from sci4rv.py)
	mask = np.genfromtxt('SciMask_for_RV.dat',dtype=None,names=True)
	wmask = mask["wmask"]
	fmask = mask["fmask"]
	

	# ==============================
	# 	RV first guess
	# ==============================
	if myRVguess != 0.0:
		RVguess = myRVguess
	else:
		dvel, vwidth = rvtbx.create_dvel(inst,wave,RVguess=0.0,RVampl=200.,verbose=False)
		my_vwidth = wmask*0.0 + vwidth
		#widthJ = wmaskJ*0.0 + vwidth
		#widthJ2 = wmaskJ2*0.0 + vwidth
		#my_vwidth = np.concatenate((widthH,widthJ,widthJ2))
		
		CCFo = dvel*0.0
		if norders > 1:
			test_orders = [9, 25, 41, 51]
			for to in test_orders:
				test_order = to
				w, f, ef 	= wave[test_order,:], flux[test_order,:], eflux[test_order,:]
				CCFo_tmp,eCCFo_tmp = rvtbx.CCF(w,f,ef,dvel,my_vwidth, wmask, fmask)
				CCFo += CCFo_tmp
		else:
			w, f, ef 	= wave, flux, eflux 
			CCFo,eCCFo = rvtbx.CCF(w,f,ef,dvel,my_vwidth, wmask, fmask)

		
		popt, perr = rvtbx.fit_CCF(dvel,CCFo,CCFo*0.0+1., with_Moon = False)
		RVguess = dvel[np.argmin(CCFo)]#popt[1]
		CCFfwhm = popt[2]*2.*np.sqrt(2.*np.log(2.))
		dvel0	= dvel.copy()
		
# 		if ~np.isfinite(RVguess):
# 			RVguess = 0.0
# 			CCFfwhm = 300.0
# 			print CCFo
# 			plt.plot(dvel,CCFo)
# 			plt.show()
# 			sys.exit()
			
		
	
	print "        --> Estimated RV (no BERV corrected) = ",np.str(round(RVguess,3))," km/s"

	# ==============================
	# 	Cross-Correlation Funtion
	# ==============================

	CCFall, eCCFall 		= [], []
	RVall, eRVall 			= [], []
	FWHMall, eFWHMall 		= [], []
	Heightall, eHeightall 	= [], []
	snrall, poptall			= [], []
	if CCFfwhm < 20.0: 
		RVampl =  30.0
# 		if CCFfwhm < 3.0: RVampl =  20.0
# 		if CCFfwhm > 3.0: RVampl =  20.0
	elif CCFfwhm > 200.0:
		RVampl =  200.0
		RVguess = 0.0
	else:
		RVampl =  5.*CCFfwhm
	
	dvel, vwidth = rvtbx.create_dvel(inst,wave,RVguess=RVguess,RVampl=RVampl)
	my_vwidth = wmask*0.0 + vwidth
# 	widthJ = wmaskJ*0.0 + vwidth
# 	widthJ2 = wmaskJ2*0.0 + vwidth
# 	my_vwidth = np.concatenate((widthH,widthJ,widthJ2))

	print "        --> Calculating CCF..."

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
		
		if np.count_nonzero(f) !=0:
			CCFo,eCCFo   = rvtbx.CCF(w,f,ef,dvel,my_vwidth, wmask, fmask)
			popto, perro = rvtbx.fit_CCF(dvel,CCFo,eCCFo, with_Moon = with_Moon)
		else:
			CCFo,eCCFo = np.zeros(len(dvel)), np.zeros(len(dvel))
			popto, perro = np.zeros(4)*np.nan, np.zeros(4)*np.nan
		
		# Append results
		CCFall.append(CCFo)	
		eCCFall.append(eCCFo)	
		RVall.append(popto[1])	
		eRVall.append(perro[1])	
		FWHMall.append(popto[2]*2.*np.sqrt(2.*np.log(2.)))	
		eFWHMall.append(perro[2]*2.*np.sqrt(2.*np.log(2.)))	
		Heightall.append(popto[0]/popto[3])	
		eHeightall.append(perro[0]/popto[3])
		poptall.append(popto)

	CCFall, eCCFall = np.array(CCFall), np.array(eCCFall)
	RVall, eRVall 	= np.array(RVall), np.array(eRVall)
	poptall			= np.array(poptall)


	# ===== CCF stacking of all orders
	CCF  = np.nansum(CCFall,axis=0)
	eCCF = np.sqrt(np.nansum(eCCFall**2,axis=0))
	

	# ===== Gaussian fit to the stacked CCF
	print "        --> Fitting CCF..."
	popt, perr = rvtbx.fit_CCF(dvel,CCF,eCCF, with_Moon = with_Moon)

	# ===== RV results and corrections

	# RV from CCF
	RV 		= popt[1]
	eRV 	= perr[1]
	FWHM	= popt[2]*2.*np.sqrt(2.*np.log(2.))
	Height	= popt[0]

# 	if ~np.isfinite(RV):
# 		print CCF
# 		print CCFall
# 		np.savez('tmpCCF',dvel=dvel,CCF=CCF,eCCF=eCCF, CCFo=CCFo, dvel0=dvel0)
# 		plt.plot(dvel0,CCFo)
# 		plt.show()
# 		sys.exit()


	
	# ===== RV uncertainty from Boisse et al. (2010)
	try:
		deriva = np.gradient(CCF,dvel)
		Nscale = 0.25 # pix # np.sqrt(np.nanmean(np.diff(dvel))   )
		Q_CCF = np.sqrt(np.nansum(deriva**2/CCF)) / np.sqrt(np.nansum(CCF)) * np.sqrt(Nscale)
		eRV2 = 1./(Q_CCF*np.sqrt(np.nansum(CCF)))
	except:
		eRV2 = np.nan
	
	
# 	total = 0.0
# 	for oo in enumerate(sel_orders): 
# 		deriv = np.gradient(f[oo,300:-300],w[oo,300:-300])
# 		total += np.nansum(w[oo,300:-300]**2 * deriv**2/(f[oo,300:-300])+3.3**2)
# 	eRV 	= 1./np.sqrt(total) * c.c.value


	RVdict =  { 'CCF':CCF,
				'eCCF':eCCF,
				'dvel':dvel,
				'Moon_corr':'yes' if with_Moon == True else 'no',
				'RVall':RVall,
				'eRVall':eRVall,
				'FWHMall':FWHMall,
				'eFWHMall':eFWHMall,
				'Heightall':Heightall,
				'eHeightall':eHeightall,
				'RV':RV,
				'eRV':eRV,
				'eRV2':eRV2,
				'CCFFWHM':FWHM,
				'CCFheight':Height
				}


	if 1:
		fig = plt.figure(figsize=(12,8))
		gs = gridspec.GridSpec(3,2, height_ratios=[1.,1.,1.], width_ratios=[1,1])
		gs.update(left=0.1, right=0.95, bottom=0.08, top=0.93, wspace=0.12, hspace=0.08)
		# CCF
		ax1 = plt.subplot(gs[:,0]) 
		color=iter(cm.coolwarm_r(np.linspace(0,1,len(sel_orders))))
		for i in range(len(sel_orders)):
			c = next(color)
			ccf_norm_factor = poptall[i][3] if np.isfinite(poptall[i][3]) else np.nanmedian(CCFall[i,:])
			plt.plot(dvel,CCFall[i,:]/ccf_norm_factor,c=c,alpha=0.3,zorder=0)

		CCF_norm_factor = popt[3] if np.isfinite(popt[3]) else np.nanmedian(CCF)
		plt.errorbar(dvel, CCF/CCF_norm_factor, eCCF/CCF_norm_factor,c='k',lw=2,zorder=5,label='Observed CCF')	
		
		if with_Moon == False:
			plt.plot(dvel,rvtbx.gaussfit(dvel,*popt)/CCF_norm_factor,c='red',lw=2,alpha=0.7,zorder=10,label='Model CCF')
		else:
			plt.plot(dvel,rvtbx.gaussfit_Moon(dvel,*popt)/CCF_norm_factor,c='red',lw=2,alpha=0.7,zorder=10,label='Model CCF w/ Moon')
			
		plt.axvline(0.0,ls=':',c='gray',alpha=0.5)
		plt.axvline(RV,ls=':',c='red',alpha=0.8)		
		plt.xlabel('Radial velocity (km/s)')
		plt.ylabel('sum(CCF_o*S/N_o)')
		plt.legend()
		# RV per order
		ax2 = plt.subplot(gs[0,1])
		color=iter(cm.coolwarm_r(np.linspace(0,1,len(sel_orders))))
		for tt in range(len(sel_orders)):
			c = next(color)
			plt.errorbar(sel_orders[tt],RVall[tt],yerr=eRVall[tt],fmt='o',c=c,ecolor=c)#c='Dodgerblue')
		plt.axhline(RV,ls=':',c='k')
		plt.ylabel('RV (km/s)')
		# CCF height
		ax3 = plt.subplot(gs[1,1])
		color=iter(cm.coolwarm_r(np.linspace(0,1,len(sel_orders))))
		for tt in range(len(sel_orders)):
			c = next(color)
			plt.errorbar(sel_orders[tt],Heightall[tt],yerr=eHeightall[tt],fmt='o',c=c,ecolor=c)#c='Tomato')
		plt.axhline(popt[0]/CCF_norm_factor,ls=':',c='k')
		plt.ylabel('CCF height (normalized)')
		# CCF FWHM
		ax4 = plt.subplot(gs[2,1])
		color=iter(cm.coolwarm_r(np.linspace(0,1,len(sel_orders))))
		for tt in range(len(sel_orders)):
			c = next(color)
			plt.errorbar(sel_orders[tt],FWHMall[tt],yerr=eFWHMall[tt],fmt='o',c=c,ecolor=c)#c='green')
		plt.axhline(popt[2]*2.*np.sqrt(2.*np.log(2.)),ls=':',c='k')
		plt.ylabel('FWHM (km/s)')

		plt.savefig(plot_name)
		plt.close()

	
	return RV, eRV, popt, perr, RVdict


# ========================================================================================
# 										GET RV
# ========================================================================================


def ScienceRV(frames, cv, frame_names):
		
	inst = 'CAFE'
	
	RVguess = 0.0	
	
# 	RVs = []
# 	eRVs = []
# 	hjds = []
	RVdicts = []

	for i,frame in enumerate(frames):
		already_done = os.path.isfile(cv.aux_dir+'RVdict_'+frame_names[i]+'.npz')
		print "    ==> RVs for..."+frame_names[i]
		if already_done == False:
			hdr = fits.getheader(cv.path_red+cv.dir_red+'/'+frame_names[i])
			berv, hjd, coord_flag = CAFEutilities.get_berv(hdr)

			if "ThAr" in frame_names[i]:
				ThArMask = np.genfromtxt('ThAr_for_RV.dat',dtype=None,names=True)
				wmask = ThArMask["wmask"]
				RV, eRV, popt, perr, RVall, eRVall, eRV2, RVdict = RB07.get_RV(frame, inst, wmask, guessRV = False, with_Moon = False, plot_name=cv.aux_dir+'/RV_'+frame_names[i]+'.pdf')

			else:
				RV, eRV, popt, perr,RVdict = get_RV(frame, inst, cv, with_Moon = False, plot_name=cv.aux_dir+'/RV_'+frame_names[i]+'.pdf')
				RV += berv
		
			RVdict['HJD'] = hjd
			RVdict['BERV'] = berv
			RVdict['RV'] = RV
			np.savez(cv.aux_dir+'RVdict_'+frame_names[i],RVdict=RVdict)

		else:
			print "        --> RVdict found for "+frame_names[i]+". Loading..."
			load_dict = np.load(cv.aux_dir+'RVdict_'+frame_names[i]+'.npz',allow_pickle=True)
			RVdict = load_dict["RVdict"].item()

		try:
			print "        --> RV = ",np.str(round(RVdict['RV'],3))+' +/- '+np.str(round(RVdict['eRV'],3))+' km/s'
		except:
			print "        --> RV is not a number..."
	
					
# 		RVs.append(RV)
# 		eRVs.append(eRV)
# 		hjds.append(hjd)
		RVdicts.append(RVdict)

	return RVdicts











