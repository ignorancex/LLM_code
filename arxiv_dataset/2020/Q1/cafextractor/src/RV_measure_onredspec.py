import sys
import os

import scipy
import pyfits
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
import argparse
from astropy.utils.data import get_pkg_data_filename
from astroML.stats import sigmaG

import GLOBALutils
import CAFEutilities
import CAFEx_SetupFile as CS
import rvtoolbox as rvtbx
import read_cafex as rcafex

import RB07_CrossCorr as RB07

"""
	Module to re-extract the CCF of a given spectrum reduced by CAFEx with defined 
	values for the guessed RV and velocity amplitude to calculate the CCF. It creates
	a new plot and, if desired, overwrites the fits file with the new RV value (the 
	latter is to be implemented).
	
	INPUT
	-----
	path		Full path to the reduced spectrum file
	
	OPTIONAL INPUT
	--------------
	RVguess		Guessed RV to create the velocity array [km/s]
	RVampl		Amplitude of the velocity array [km/s]
	UPDATERV	Boolean. Update the corresponding header keywords of the reduced file
	
	
"""


# ========================================================================================

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to data folder")
    parser.add_argument("-rv", "--RVguess", type=float , help="Guessed RV")
    parser.add_argument("-ampl", "--RVampl", type=float , help="Guessed RV")
    parser.add_argument("-U", "--UPDATERV", help="Update header keywords", action="store_true")
    parser.add_argument("-R", "--ROTPROF", help="Rotational profile to fit CCF", action="store_true")
    parser.add_argument("-C", "--COORD", help="Use these coordinates to calculate BERV", default=None)
    args = parser.parse_args()
    return args

# ========================================================================================



# ========================================================================================
# 										GET RV
# ========================================================================================

def get_RV(sp,inst, jdnight, sel_orders=-10, 
			guessRV=True, 
			myRVampl=-20., 
			myRVguess=-99.9, 
			with_Moon = False, 
			rot_profile=False,
			COORD='-99.99',
			plot_name='tmp.pdf'):


	if myRVguess != -99.9: guessRV=False
	CS.var.set_OrderProp(jdnight)
	# Orders exclude (CARMENES-wise)
	# exclude_orders = [29, 30, 38, 39, 43, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]

	# Data properties
	norders,npix = np.shape(sp.flux)
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

	
	wave = sp.wave[:,300:-300]
	flux = sp.flux[:,300:-300]
	eflux = sp.eflux[:,300:-300]

	# Mask (from sci4rv.py)
	mask = np.genfromtxt('SciMask_for_RV.dat',dtype=None,names=True)
	wmask = mask["wmask"]
	fmask = mask["fmask"]
	

	# ==============================
	# 	RV first guess
	# ==============================
	if myRVguess != -99.9:
		RVguess = myRVguess
		CCFfwhm = 30.
	else:
		dvel, vwidth = rvtbx.create_dvel(inst,wave,RVguess=0.0,RVampl=200.,verbose=True)
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
	
	print "Estimated RV (no BERV corrected) = ",np.str(round(RVguess,3))," km/s"

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
	elif CCFfwhm > 200.0:
		RVampl =  200.0
		RVguess = 0.0
	else:
		RVampl =  5.*CCFfwhm
	
	if myRVampl > 0.: RVampl = myRVampl
	
	
	dvel, vwidth = rvtbx.create_dvel(inst,wave,RVguess=RVguess,RVampl=RVampl)
	my_vwidth = wmask*0.0 + vwidth
# 	widthJ = wmaskJ*0.0 + vwidth
# 	widthJ2 = wmaskJ2*0.0 + vwidth
# 	my_vwidth = np.concatenate((widthH,widthJ,widthJ2))

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
		
		if np.count_nonzero(f) !=0:
			CCFo,eCCFo   = rvtbx.CCF(w,f,ef,dvel,my_vwidth, wmask, fmask)
			popto, perro = rvtbx.fit_CCF(dvel,CCFo,eCCFo, RVguess=myRVguess, with_Moon = with_Moon)
		else:
			CCFo,eCCFo = np.zeros(len(dvel))*np.nan, np.zeros(len(dvel))*np.nan
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
	print "Measuring RV from CCF..."
	popt, perr = rvtbx.fit_CCF(dvel,CCF,eCCF, RVguess=myRVguess, AMPLguess=80., with_Moon = with_Moon, rot_profile=rot_profile)
	
	#np.savez('tmp_ccfprofile',dvel=dvel,CCF=CCF,eCCF=eCCF)
	

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
		zeroloc = []
		for i in range(len(sel_orders)):
			c = next(color)
			ccf_norm_factor = poptall[i][3] if np.isfinite(poptall[i][3]) else np.nanmedian(CCFall[i,:])
			plt.plot(dvel,CCFall[i,:]/ccf_norm_factor,c=c,alpha=0.3,zorder=0)
			zeroloc.append(np.nanmedian(CCFall[i,:]/ccf_norm_factor))

		zeroloc = np.array(zeroloc)
		CCF_norm_factor = np.nanmedian(CCF) #popt[3] if np.isfinite(popt[3]) else 
		plt.errorbar(dvel, CCF/CCF_norm_factor, eCCF/CCF_norm_factor,c='k',lw=2,zorder=5,label='Observed CCF')	
				
		if with_Moon == False:
			if rot_profile:
				plt.plot(dvel,rvtbx.rotprofile(dvel,*popt)/CCF_norm_factor,c='red',lw=2,alpha=0.7,zorder=10,label='Model CCF')			
			else:
				plt.plot(dvel,rvtbx.gaussfit(dvel,*popt)/CCF_norm_factor,c='red',lw=2,alpha=0.7,zorder=10,label='Model CCF')
		else:
			plt.plot(dvel,rvtbx.gaussfit_Moon(dvel,*popt)/CCF_norm_factor,c='red',lw=2,alpha=0.7,zorder=10,label='Model CCF w/ Moon')
			
		plt.axvline(0.0,ls=':',c='gray',alpha=0.5)
		plt.axvline(RV,ls=':',c='red',alpha=0.8)		
		plt.xlabel('Radial velocity (km/s)')
		plt.ylabel('sum(CCF_o*S/N_o)')
		
		ymax = np.nanmedian(CCF/CCF_norm_factor) + 3.*sigmaG(np.array(zeroloc[np.isfinite(zeroloc)]))
		ymin = np.nanmin(CCF/CCF_norm_factor) - 3.*sigmaG(np.array(zeroloc[np.isfinite(zeroloc)]))
		plt.ylim(ymin,ymax)
		plt.legend()
		
		# RV per order
		ax2 = plt.subplot(gs[0,1])
		color=iter(cm.coolwarm_r(np.linspace(0,1,len(sel_orders))))
		for tt in range(len(sel_orders)):
			c = next(color)
			plt.errorbar(sel_orders[tt],RVall[tt],yerr=eRVall[tt],fmt='o',c=c,ecolor=c)#c='Dodgerblue')
		plt.axhline(RV,ls=':',c='k')
		plt.ylabel('RV (km/s)')
		ymin, ymax = np.nanmedian(RVall) -5.*np.nanstd(RVall), np.nanmedian(RVall) +5.*np.nanstd(RVall)
		if np.isfinite(ymin) & np.isfinite(ymax): plt.ylim(ymin, ymax)
		
		# CCF height
		ax3 = plt.subplot(gs[1,1])
		color=iter(cm.coolwarm_r(np.linspace(0,1,len(sel_orders))))
		for tt in range(len(sel_orders)):
			c = next(color)
			plt.errorbar(sel_orders[tt],Heightall[tt],yerr=eHeightall[tt],fmt='o',c=c,ecolor=c)#c='Tomato')
		plt.axhline(popt[0]/CCF_norm_factor,ls=':',c='k')
		plt.ylabel('CCF height (normalized)')
		ymin, ymax = np.nanmedian(Heightall) -5.*np.nanstd(Heightall), np.nanmedian(Heightall) +5.*np.nanstd(Heightall)
		if np.isfinite(ymin) & np.isfinite(ymax): plt.ylim(ymin, ymax)

		# CCF FWHM
		ax4 = plt.subplot(gs[2,1])
		color=iter(cm.coolwarm_r(np.linspace(0,1,len(sel_orders))))
		for tt in range(len(sel_orders)):
			c = next(color)
			plt.errorbar(sel_orders[tt],FWHMall[tt],yerr=eFWHMall[tt],fmt='o',c=c,ecolor=c)#c='green')
		plt.axhline(popt[2]*2.*np.sqrt(2.*np.log(2.)),ls=':',c='k')
		plt.ylabel('FWHM (km/s)')
		ymin, ymax = np.nanmedian(FWHMall) -5.*np.nanstd(FWHMall), np.nanmedian(FWHMall) +5.*np.nanstd(FWHMall)
		if np.isfinite(ymin) & np.isfinite(ymax): plt.ylim(ymin, ymax)

		plt.savefig(plot_name)
		plt.close()

	
	return RV, eRV, popt, perr, RVdict


# ========================================================================================

def recalculate_rv(path,args):
	"""
	Fuction to recalculate the RV and update the heder and plots for each frame
	"""
	
	UPDATERV = args.UPDATERV
	sp = rcafex.read_spec(path, FULL_PATH=True)
	inst = 'CAFE'
	jdnight = sp.head["HIERARCH CAFEX HJD"]
	cards = np.array(sp.head.cards)[:,0]

	if args.RVampl is not None:
		myRVampl = args.RVampl
	else:
		myRVampl = -1
	
	if args.RVguess is not None:
		myRVguess = args.RVguess
	else:
		myRVguess = 0.0


	if UPDATERV:
		frame_name = os.path.basename(path)
		plotname = os.path.dirname(path)[:-8]+'/auxiliar/RV_'+frame_name+'.pdf'
	else:
		plotname = 'tmp.pdf'

	RV, eRV, popt, perr,RVdict = get_RV(sp, inst, jdnight, myRVguess=myRVguess, myRVampl=myRVampl, with_Moon = False, rot_profile= args.ROTPROF, plot_name=plotname)

	# ===== BERV correction
	if args.COORD is not None:
		_hdr = sp.header #'a'
		coords = args.COORD
		ra,dec = coords.split(' ')
		BERV = CAFEutilities.get_berv(_hdr,RA=np.float(ra), DEC=np.float(dec))	
	else:
		BERV = sp.head['HIERARCH CAFEX BERV']
	
	RV += BERV

	# ===== SNR-corrected RV
	snr = sp.head["HIERARCH CAFEX SNR"]
	corr_coeff = np.flip([5.07794635e-01, -1.03076535e-02,  4.72775655e-05])
	poly_corr = np.poly1d(corr_coeff)
	corr = poly_corr(snr)
	RVcorr = RV - corr

	# ===== ThAr-change RV correction
	# Read ThAr lamp changes for RV correction
	thar_change_path = CS.RefFrames
	t = np.genfromtxt(thar_change_path+'/ThAr_LampChanges.dat',dtype=None)
	thar_jdstart, thar_jdend, ThAr_corr = t['f1'], t['f2'], t['f5']
	this_thar = np.where((thar_jdstart < jdnight) & (thar_jdend > jdnight))[0]
	tharcorr = ThAr_corr[this_thar]*1.e-3

	RVcorrThAr = RVcorr - tharcorr[0]


	print RVcorr,eRV

	if UPDATERV:
		print "Updating RV header keywords: RV, ERV, FWHM, HEIGHT, RVCORR..."
		fits.setval(path, 'HIERARCH CAFEX RV', value=RV)
		fits.setval(path, 'HIERARCH CAFEX ERV', value=eRV)
		fits.setval(path, 'HIERARCH CAFEX CCF FWHM', value=popt[2]*2.*np.sqrt(2.*np.log(2.)))
		fits.setval(path, 'HIERARCH CAFEX CCF HEIGHT', value=popt[0])
		fits.setval(path, 'HIERARCH CAFEX RVCORR', value=RVcorr)
		fits.setval(path, 'HIERARCH CAFEX RVCORR2', value=RVcorrThAr, comment='RV - SNRcorr - ThArcorr  [km/s]')
		fits.setval(path, 'HIERARCH CAFEX THARCORR', value=tharcorr[0], comment='ThArcorr correction [m/s]')

# ========================================================================================

if __name__ == "__main__":

	args = cli()
	path_to_file = args.path

	if path_to_file.endswith('.fits'):
		path = path_to_file
		recalculate_rv(path,args)
	else:
		print 'This is a list...'
		files = np.genfromtxt(path_to_file,dtype=None)
		for file in files:
			print 'File: ',file
			recalculate_rv(file,args)







