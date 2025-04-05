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
from astropy import constants as C
from astropy.table import Table, Column
from astropy.io import ascii
import sys
import matplotlib.gridspec as gridspec # GRIDSPEC !

cc 		= C.c.value*1.e-3	# [km/s]

# Gaussian function
def gauss_function(x, a, x0, sigma, zero):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + zero
# Two Gaussian function
def twoGauss_function(x, a, x0, sigma, zero, a2, x02, sigma2):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + zero + a2*np.exp(-(x-x02)**2/(2*sigma2**2))



# ===== Master Arc ffor Reference
tpath = '/Users/lillo_box/00_Instrumentation/CAFE/CAFExtractor/test_data/12_REDUCED/180725_digest/reduced'
t = fits.open(tpath+'/MasterArc_0_red.fits') #

# ===== ThAr lines from Lovis+2007
lovis_file = '/Users/lillo_box/00_Instrumentation/CAFE/CAFExtractor/cafextractor/ReferenceFrames/ThAr_ReferenceLines/ThAr_Lovis07.txt'
file = np.genfromtxt(lovis_file,dtype=None)
lovis_lines = file['f0']
lovis_int = file['f2']*1.
lovis_Species = file['f3']
s = 1.e4/lovis_lines
n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
lovis_lines = lovis_lines/n

# ===== ThAr lines from CERES
file2 = np.genfromtxt('/Users/lillo_box/00_Instrumentation/CAFE/CAFExtractor/cafextractor/ReferenceFrames/ThAr_ReferenceLines/all_ThAr_lines.lis',dtype=None)
ceres_lines = file2['f2']

Xpix = np.arange(2048)*1.

# ===== LOOP for each order (starting at the first order with Lovis+ information)
for oo in np.arange(80-23)+23:
	oo = 87-60

	wave = t["WAVELENGTH"].data[oo,:]
	flux = t["FLUX"].data[oo,:]
	fig = plt.figure(figsize=(13,6))
	plt.plot(wave, flux,c='k',zorder=0)
	
	yes_cafex 	= np.genfromtxt('auxiliar/yes_cafeX_order_'+str(60+oo).zfill(3)+'.dat',names=True, dtype=(int, float, float, float, str))
	no_cafex  	= np.genfromtxt('auxiliar/no_cafeX_order_'+str(60+oo).zfill(3)+'.dat',names=True, dtype=(int, float, float, float, str))
	nor_cafex  	= np.genfromtxt('auxiliar/noreasons_cafeX_order_'+str(60+oo).zfill(3)+'.dat',names=True, dtype=(int, str, float, float, float, float, float, float, float))
	ceres 		= np.genfromtxt('order_'+str(60+oo).zfill(3)+'.iwdat',dtype=None)
	
	for ii,yes in enumerate(yes_cafex["Wavelength"]): 
		plt.axvline(yes, ls=':',c='blue', zorder=10)
		plt.text(yes,0., yes_cafex["ID"][ii], color='blue')
	for ii,no  in  enumerate(no_cafex["Wavelength"]): 
		plt.axvline(no, ls=':',c='k', alpha=0.6, zorder=7)
		plt.text(no,0., no_cafex["ID"][ii], color='k')

	for ii,cer  in  enumerate(ceres["f2"]): 
		plt.axvline(cer, ls='--',c='red', alpha=0.6, zorder=5)

	print "%5s %5s %10s %10s %10s %10s %10s %10s %10s" % ("ID","ok","Diff(mA)","chi2","SNR","Diff/R","I1/I0","I0","I1")
	for hh in nor_cafex:
		print "%5s %5s %10.1f %10.1f %10.1f %10.1f %10.3f %10.1f %10.1f" % (hh[0],hh[1],hh[2],hh[3],hh[4],hh[5],hh[6],hh[7],hh[8])
	
	plt.ylim(-500,3000)
	plt.xlim(np.min(wave),np.min(wave)+15.)
	plt.show()
	plt.close()
	sys.exit()
	
	







