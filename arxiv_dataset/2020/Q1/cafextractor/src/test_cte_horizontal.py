import GLOBALutils
import scipy
import pyfits
import numpy as np
import os
import matplotlib.pyplot as plt
import CAFEutilities
import CAFEx_SetupFile as CS
from astropy.io import fits
from scipy.signal import find_peaks_cwt
from scipy.optimize import curve_fit
from sklearn.metrics.pairwise import euclidean_distances#pairwise_distances
from astroML.stats import sigmaG
import sys
from astropy import constants as c
from astropy.table import Table, Column
from astropy.io import ascii
import sys
import rvtoolbox as rvtbx
import matplotlib.gridspec as gridspec # GRIDSPEC !
import jlillo_pypref

import RB08_RadialVelocity as RB08


files = np.genfromtxt('test_files.lis',dtype=None)

RV_L = np.zeros(len(files))
eRV_L = np.zeros(len(files))
RV_R = np.zeros(len(files))
eRV_R = np.zeros(len(files))
RV_C = np.zeros(len(files))
eRV_C = np.zeros(len(files))
JD = np.zeros(len(files))
SNR = np.zeros(len(files))


for ii,file in enumerate(files):
	hdr = fits.getheader(file)
	SNR[ii] = hdr["HIERARCH CAFEX SNR"]

if 0:

	for ii,file in enumerate(files):

		print file
		hdr = fits.getheader(file)
		berv, hjd, coord_flag = CAFEutilities.get_berv(hdr)
		frame = fits.open(file)
		flux = frame[1].data
		JD[ii] = hdr['JUL-DATE']

		# for i in range(82): 
		# 	x, y = a[2].data[i,:], a[1].data[i,:]
		# 	idx = np.isfinite(x) & np.isfinite(y)
		# 	c = np.polyfit(x[idx], y[idx],1)
		# 	p = np.poly1d(c)
		# 	a[1].data[i,:] = a[1].data[i,:]/p(x)
		# 	#plt.plot(x,y/p(x))


		# Right side
		right_flux = flux.copy()
		right_flux[:, 0:1200] = np.nan
		right_flux[:, 1800:] = np.nan
		ff_right  	= np.array([[np.zeros((84,2048))],[right_flux], [frame[3].data], [frame[2].data]]).reshape(4,84,2048)
		RV, eRV, popt, perr,RVdict = RB08.get_RV(ff_right,'CAFE', plot_name='tmp_right.pdf')
		RV += berv
		RV_R[ii] = RV
		eRV_R[ii] = eRV


		# Left side
		left_flux = flux.copy()
		left_flux[:, 0:400] = np.nan
		left_flux[:, 1000:] = np.nan
		ff_left  	= np.array([[np.zeros((84,2048))],[left_flux], [frame[3].data], [frame[2].data]]).reshape(4,84,2048)
		RV, eRV, popt, perr,RVdict = RB08.get_RV(ff_left,'CAFE', plot_name='tmp_left.pdf')
		RV += berv
		RV_L[ii] = RV
		eRV_L[ii] = eRV

		# Center side
		cent_flux = flux.copy()
		cent_flux[:, 0:800] = np.nan
		cent_flux[:, 1400:] = np.nan
		ff_cent  = np.array([[np.zeros((84,2048))],[cent_flux], [frame[3].data], [frame[2].data]]).reshape(4,84,2048)
		RV, eRV, popt, perr,RVdict = RB08.get_RV(ff_cent,'CAFE', plot_name='tmp_cent.pdf')
		RV += berv
		RV_C[ii] = RV
		eRV_C[ii] = eRV

		
		print RV_L[ii], RV_C[ii], RV_R[ii]

	np.savez('cte_test_horizontal',JD=JD, SNR=SNR, RV_L=RV_L, RV_R=RV_R, RV_C=RV_C, eRV_L=eRV_L, eRV_R=eRV_R, eRV_C=eRV_C)

else:
	a = np.load('cte_test_horizontal.npz')
	JD = a["JD"]
	RV_R, eRV_R = a["RV_R"], a["eRV_R"]
	RV_L, eRV_L = a["RV_L"], a["eRV_L"]
	RV_C, eRV_C = a["RV_C"], a["eRV_C"]
	
	
plt.figure(figsize=(10,5))
plt.errorbar(SNR,(RV_R-RV_L)*1.e3,yerr=np.sqrt(eRV_L**2+eRV_R**2)*1.e3, c='k',fmt='o')
plt.xlabel('S/N')
plt.ylabel(r'$RV_{\rm right}-RV_{\rm left}$ (m/s)')
plt.grid(ls=':',c='gray',alpha=0.7)
plt.savefig('CTE_CCD_horizontal_relative.pdf',bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,5))
plt.errorbar(SNR,RV_L,yerr=eRV_L, c='red',fmt='o',label='Left CCD')
plt.errorbar(SNR,RV_C,yerr=eRV_C, c='limegreen',fmt='o',label='Central CCD')
plt.errorbar(SNR,RV_R,yerr=eRV_R, c='blue',fmt='o',label='Right CCD')
plt.legend()
plt.xlabel('S/N')
plt.ylabel('RV (km/s)')
plt.grid(ls=':',c='gray',alpha=0.7)
plt.savefig('CTE_CCD_horizontal_absolute.pdf',bbox_inches='tight')
plt.close()


