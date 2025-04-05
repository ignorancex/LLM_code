import scipy
import pyfits
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks_cwt
from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import euclidean_distances#pairwise_distances
from astroML.stats import sigmaG
import sys
import argparse

from astropy import constants as c
from astropy.table import Table, Column
from astropy.io import ascii
from astropy.stats import sigma_clip
import rvtoolbox as rvtbx
from astropy.table import Table, Column, MaskedColumn
from astropy.io import ascii
import CAFEx_SetupFile as CS

"""
Purpose
-------
		Determine the ThAr lines to be used in the determination of the RVs of the
		arc frames. We start from the Lovis+20XX line list and remove those lines 
		introducing contaminatino in the CCF through an iterative process. 
Input
-----
		The script needs the path to a wavelength calibrated arc frame (e.g., 
		a MasterARC reduced by CAFExtractor). The wavelength calibration does not 
		need to be extremely good.
Output
------
		The final result is a file containing the ThAr lines to be used: ThAr_for_RV.dat

"""



# ==================
# USEFUL FUNCTIONS
# ==================

def gaussfit(x, a0, a1, a2, a3, a4):
	"""
	Gaussian function
	"""
	z = (x-a1) / a2
	y = a0 * np.exp(-z**2 / 2) + a3 + a4 * x #+ a5 * x**2
	return y


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-V", "--VERBOSE", help="Verbose check", action="store_true")
    args = parser.parse_args()
    return args
	


# ==================
# MAIN
# ==================
args = cli()

# _____________________  PREPARATION _____________________________________________________ 

# ===== Read SCI frame to be used
arc = fits.open('sci4rv/HD109358__190425_0100_red.fits')
hdr = arc[0].header
realRV = hdr["HIERARCH CAFEX RV"]
berv = hdr["HIERARCH CAFEX BERV"]

wave 	= arc["WAVELENGTH"].data[:,300:-300]
flux 	= arc["FLUX"].data[:,300:-300]
eflux 	= arc["EFLUX"].data[:,300:-300]*1.e7


norders, npix 	= np.shape(wave)
inst 			= 'CAFE'
cc 				= c.c.value*1.e-3	# [km/s]

# ===== ThAr lines from current science status
# Mask
mask = np.genfromtxt('mask_v05.dat',dtype=None,names=True)
wmask = mask["WavelengthA"]
fmask = mask["heightnorm"]
fwhm_mask = mask["FWHMkms"]
no_broad = np.where(fwhm_mask < 10.)
wmask, fmask = wmask[no_broad], fmask[no_broad]

harps_mask = np.genfromtxt(CS.RefFrames+'G2.mas',dtype=None,names=True)
wmaskH = (harps_mask["W0"]+harps_mask["W1"])/2.
fmaskH = harps_mask["Weight"]
widthH = (wmaskH-harps_mask["W0"])/wmaskH * c.c.value*1.e-3

me_elem = np.where(wmask > 6798.55)[0]
wmaskJ, fmaskJ = wmask[me_elem], fmask[me_elem]

me2 = np.where((wmask > 5867.59) & (wmask < 6002.98))
wmaskJ2, fmaskJ2 = wmask[me2], fmask[me2]

wmask = np.concatenate((wmaskH,wmaskJ,wmaskJ2))
fmask = np.concatenate((fmaskH,fmaskJ,fmaskJ2))
id = np.concatenate((np.array(["H"*len(wmaskH)]),np.array(["J"*len(wmaskJ)]),np.array(["J"*len(wmaskJ)])))




# ===== Orders to be analyzed
""" Orders < 23 are outlside of the HARPS wavelength range, so are not used here."""
sel_orders = np.arange(norders-2-21)+22


# _____________________  CROSS-CORRELATION _______________________________________________ 


# ===== Velocity array
dvel, vwidth = rvtbx.create_dvel(inst,wave,RVguess=realRV-berv,RVampl=20.)
vmin 	= np.min(dvel)
vmax 	= np.max(dvel)

# Mask
fwhm_mask = wmask*0.0 + 4.0
widthJ  = wmaskJ*0.0 + vwidth
widthJ2 = wmaskJ2*0.0 + vwidth
my_vwidth = np.concatenate((widthH,widthJ,widthJ2))


# ===== Initialize some arrays
order_flag = np.zeros(len(sel_orders))
WMASK = np.array([])
FMASK = np.array([])



# ===== Loop for each order
for jj,oo in enumerate(sel_orders): 	

	#| Avoid overlaping order wavelengths
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

	#| Select mask lines in this wavelength range
	inrange = np.where((wmask*(1.+vmin/cc) > np.min(w)+2. ) & (wmask*(1.+vmax/cc) < np.max(w)-2. ))[0]
	wmaskR = wmask[inrange]
	fmaskR = fmask[inrange]
	my_vwidthR = my_vwidth[inrange]
	orig_nlines = len(wmaskR)
	
	#| Check if already done
	already_done = os.path.isfile('sci4rv/order_'+str(oo).zfill(3)+'_sci4rv.npz')
	
	#| Loop to select the well-behaved ThAr lines 
	"""
	Considerations:	
		- Measure the CCF cumulatively adding one ThAr per step and measuring the 
		  quality of the fit through the reduced chi square parameter.
		- Loop break conditions:
			(a)		No more lines to remove
			(b)		Maximum of 20 iterations. In this case, the order should not be used
			(c)		Uncertainty in center below 100 m/s
		- Remove lines that: 
			(i) 	Induced a NaN fit of the CCF, 
			(ii)	Produce the CCF center to shift more than 5 km/s away from 0, 
			(iii) 	Produce an increase of the reduced chi square above 100.  
			(iv)	Produce an increase in the RV uncertainty above the 50%
	"""
	
	if already_done == False:
		iter = 0
		while True:
		
			chi2 	= np.zeros(len(wmaskR))
			rverr 	= np.zeros(len(wmaskR))
		
			#| Iteration per line
			for m in np.arange(len(wmaskR)):
			
				#| Select only lines from 0 to m 
				wmaskR2 = wmaskR[:m]
				fmaskR2 = -1.*fmaskR[:m]
				my_vwidthR2 = my_vwidthR[:m]
			
				#| Calculate and fit CCF
				CCFo,eCCFo   = rvtbx.CCF(w,f,ef,dvel,my_vwidthR2, wmaskR2, fmaskR2, CRAYS=False)
				popto, perro = rvtbx.fit_CCF(dvel,CCFo,eCCFo, guessRV=True, with_Moon = False)
				rverr[m] = perro[1]
# 				if m > 0:
# 					plt.plot(dvel,CCFo)
# 					plt.show()
# 					sys.exit()
			
				#| Chi-square determination
				if ~np.isfinite(popto[0]):
					chi2[m] = 0.0
				elif np.abs(popto[1]+(realRV-berv)) > 5.: 
					chi2[m] =  0.0
				else:
					CCFmodel = rvtbx.gaussfit(dvel,*popto)
					chi2[m] = np.sum((CCFo-CCFmodel)**2/eCCFo**2) / (1.*len(dvel))

				
				if args.VERBOSE:
					if m !=0: print m,chi2[m],rverr[m],(rverr[m]-rverr[m-1])/rverr[m-1]
					plt.plot(dvel,CCFo-popto[3])
					plt.plot(dvel,rvtbx.gaussfit(dvel,*popto)-popto[3],c='k',lw=1,ls='--')
					if (np.isfinite(popto[1]) & np.isfinite(popto[0])): 
						plt.text(popto[1],popto[0],str(m))
				
			
			#| Conditions for removal
			chi2_grad = chi2[1:]-chi2[0:-1]
			rverr_grad = (rverr[1:]-rverr[0:-1])/rverr[0:-1]
			
			#| Remove condition (i)
			remove1 = np.where((chi2_grad + chi2[0:-1] == 0) & (chi2[0:-1] > 0.0))[0]
			#| Remove condition (iii)
			remove3 = np.where(chi2_grad > 100)[0]
			#| Remove condition (iv)
			remove4 = np.where(rverr_grad > 0.3)[0]
			if perro[1] < 0.030: remove4 = []
			
			
			remove	= np.concatenate((remove1,remove3,remove4), axis=None)
			remove = np.unique(remove)

			if ((iter < 10) & (~np.isfinite(popto[1])) & (len(remove) == 0)  ):
				remove = np.arange(int(0.1*len(wmaskR)))
	
			if args.VERBOSE:
				print remove1
				print remove3
				print remove4
				print remove
				plt.plot(dvel,rvtbx.gaussfit(dvel,*popto)-popto[3],c='k',lw=3)
				plt.show()
				#sys.exit()
				plt.close()
	
			#| Break condition (a)
			if len(remove) == 0:
				if ~np.isfinite(popto[1]): 
					order_flag[jj] = 0
					break
				else:
					if perro[1] < 0.1:
						order_flag[jj] = 1
						break
					else:
						remove = np.argmax(rverr_grad)+1
					
		
			#| Remove bad lines
			wmaskR = np.delete(wmaskR, remove) 
			fmaskR = np.delete(fmaskR, remove) 
			iter += 1

			#| Break condition (b)
			if iter > 30:
				if ((np.abs(popto[1]-21.)<5.) & (perro[1]<0.2)):
					order_flag[jj] = 1
				else:
					order_flag[jj] = 0
				break


		if args.VERBOSE:
			plt.plot(dvel,CCFo)
			plt.plot(dvel,rvtbx.gaussfit(dvel,*popto))
			print popto
			plt.show()
			sys.exit()

		np.savez('sci4rv/order_'+str(oo).zfill(3)+'_sci4rv', iter=iter, wmaskR=wmaskR, 
				 fmaskR=fmaskR, popto=popto, perro=perro, oflag=order_flag[jj],dvel=dvel,CCFo=CCFo)

		
	
	else:	 
		res = np.load('sci4rv/order_'+str(oo).zfill(3)+'_sci4rv.npz')
		iter 	= res["iter"]
		wmaskR 	= res["wmaskR"]
		fmaskR 	= res["fmaskR"]
		popto	= res["popto"]
		perro	= res["perro"]
		oflag	= res["oflag"]
		order_flag[jj] = oflag
		#dvel 	= res["dvel"]
		#CCFo 	= res["CCFo"]
		
		if ((oflag == 0) & (popto[1] < 0.5) & (perro[1]<0.4)): order_flag[jj] = 1
			
	print oo, iter, order_flag[jj], orig_nlines, len(wmaskR), popto[1], perro[1]
#	if order_flag[jj] == 0: print oo
	
	
	#| Save good lines
	if order_flag[jj] == 1:
		WMASK = np.concatenate((WMASK, wmaskR),axis=None)
		FMASK = np.concatenate((FMASK, fmaskR),axis=None)
# 	else:
# 		print oo

#| Save file
data = Table([WMASK, FMASK], names=['wmask', 'fmask'])
ascii.write(data, 'SciMask_for_RV.dat')


















