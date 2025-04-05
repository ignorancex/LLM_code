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
from astropy import constants as c
from astropy.table import Table, Column
from astropy.io import ascii
from astropy.stats import sigma_clip
import rvtoolbox_arcs as rvtbx
from astropy.table import Table, Column, MaskedColumn
from astropy.io import ascii

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


# ==================
# MAIN
# ==================


# _____________________  PREPARATION _____________________________________________________ 

# ===== Read arc to be used
arc = fits.open('thar4rv/MasterArc_0_red.fits')

wave 	= arc["WAVELENGTH"].data[:,300:-300]
flux 	= arc["FLUX"].data[:,300:-300]
eflux 	= arc["EFLUX"].data[:,300:-300]*1.e4

norders, npix 	= np.shape(wave)
inst 			= 'CAFE'
cc 				= c.c.value*1.e-3	# [km/s]

# ===== ThAr lines mask from Lovis+2007
ThArMask = np.genfromtxt('/Users/lillo_box/00_Instrumentation/CAFE/CAFExtractor/cafextractor/ReferenceFrames/ThAr_ReferenceLines/ThAr_Lovis07.txt',dtype=None)
wmask_vaccuum = ThArMask["f0"]
s = 1.e4/wmask_vaccuum
n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
wmask = wmask_vaccuum/n


# Mask
fmask = wmask*0.0 - 1.0
fwhm_mask = wmask*0.0 + 4.0

# ===== Orders to be analyzed
""" Orders < 23 are outlside of the HARPS wavelength range, so are not used here."""
sel_orders = np.arange(norders-2-21)+22


# _____________________  CROSS-CORRELATION _______________________________________________ 


# ===== Velocity array
dvel, vwidth = rvtbx.create_dvel(inst,wave,RVguess=0.0,RVampl=20.)
vmin 	= np.min(dvel)
vmax 	= np.max(dvel)
dlogLambda 	= vwidth/cc

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
	orig_nlines = len(wmaskR)
	
	#| Check if already done
	already_done = os.path.isfile('thar4rv/order_'+str(oo).zfill(3)+'_thar4rv.npz')
	
	#| Loop to select the well-behaved ThAr lines 
	"""
	Considerations:	
		- Measure the CCF cumulatively adding one ThAr per step and measureing the 
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
			for m in range(len(wmaskR)):
			
				#| Select only lines from 0 to m 
				wmaskR2 = wmaskR[0:m]
				fmaskR2 = -1.*fmaskR[0:m]
			
				#| Calculate and fit CCF
				CCFo,eCCFo   = rvtbx.CCF(w,f,ef,dvel,vwidth, wmaskR2, fmaskR2, CRAYS=False)
				popto, perro = rvtbx.fit_CCF(dvel,CCFo,eCCFo, guessRV=True, with_Moon = False)
				rverr[m] = perro[1]
			
				#| Chi-square determination
				if ~np.isfinite(popto[0]):
					chi2[m] = 0.0
				elif np.abs(popto[1]) > 5.: 
					chi2[m] =  0.0
				else:
					CCFmodel = rvtbx.gaussfit(dvel,*popto)
					chi2[m] = np.sum((CCFo-CCFmodel)**2/eCCFo**2) / (1.*len(dvel))

				
# 				if m !=0: print m,chi2[m],rverr[m],(rverr[m]-rverr[m-1])/rverr[m-1]
# 				plt.plot(dvel,CCFo)
# 				
# 			plt.plot(dvel,rvtbx.gaussfit(dvel,*popto),c='k',lw=3)
# 			plt.show()
# 			plt.close()
			
			#| Conditions for removal
			chi2_grad = chi2[1:]-chi2[0:-1]
			rverr_grad = (rverr[1:]-rverr[0:-1])/rverr[0:-1]
			
			#| Remove condition (i)
			remove1 = np.where((chi2_grad + chi2[0:-1] == 0) & (chi2[0:-1] > 0.0))[0]
			#| Remove condition (iii)
			remove3 = np.where(chi2_grad > 100)[0]
			#| Remove condition (iv)
			remove4 = np.where(rverr_grad > 0.5)[0]
			
			
			remove	= np.concatenate((remove1,remove3,remove4), axis=None)

			if ((iter < 10) & (~np.isfinite(popto[1])) & (len(remove) == 0)  ):
				remove = np.arange(int(0.1*len(wmaskR)))
	
# 			print remove1
# 			print remove3
# 			print remove4
# 			print remove
	
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
				if ((np.abs(popto[1])<0.2) & (perro[1]<0.1)):
					order_flag[jj] = 1
				else:
					order_flag[jj] = 0
				break


		np.savez('thar4rv/order_'+str(oo).zfill(3)+'_thar4rv', iter=iter, wmaskR=wmaskR, 
				 fmaskR=fmaskR, popto=popto, perro=perro, oflag=order_flag[jj],dvel=dvel,CCFo=CCFo)

# 		plt.plot(dvel,CCFo)
# 		print popto
# 		plt.show()
# 		sys.exit()
		
	
	else:	 
		res = np.load('thar4rv/order_'+str(oo).zfill(3)+'_thar4rv.npz')
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
			
	#print oo, iter, order_flag[jj], orig_nlines, len(wmaskR), popto[1], perro[1]
	if order_flag[jj] == 0: print oo
	
	
	#| Save good lines
	if order_flag[jj] == 1:
		WMASK = np.concatenate((WMASK, wmaskR),axis=None)
		FMASK = np.concatenate((FMASK, fmaskR),axis=None)
# 	else:
# 		print oo

#| Save file
data = Table([WMASK, FMASK], names=['wmask', 'fmask'])
ascii.write(data, 'ThAr_for_RV.dat')


















