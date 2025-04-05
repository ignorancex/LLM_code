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
# cafe2_001
# tpath = '/Users/lillo_box/00_Instrumentation/CAFE/CAFExtractor/cafextractor/test_data/11_REDUCED/120705/reduced'
# order0 = 60
# norders = 80

# cafe2_002
# tpath = '/Volumes/willyfog/gang5/jlillo/22_RUNS/2018_07_CAHA_2.2_CAFE_CHRONOS/11_REDUCED/181217/reduced'
# order0 = 60
# norders = 80

# cafe2_003
# tpath = '/Volumes/willyfog/gang5/jlillo/22_RUNS/2019_01_CAHA_2.2_CAFE_Recommisioning_Run2/11_REDUCED/190306/reduced'
# order0 = 63
# norders = 80

# cafe2_004
tpath = '/Volumes/willyfog/gang5/jlillo/22_RUNS/2019_01_CAHA_2.2_CAFE_Recommisioning_Run2/11_REDUCED/190425_digest/reduced'
order0 = 62
norders = 79

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
for oo in np.arange(norders-23)+23:

	wave = t["WAVELENGTH"].data[oo,:]
	flux = t["FLUX"].data[oo,:]
	plt.plot(wave, flux,c='k',zorder=0)

	resolution0 = 2.3 * np.mean(wave[1:-1]-wave[0:-2])
	
	ceres_inorder = np.where((ceres_lines > np.min(wave)) & (ceres_lines < np.max(wave)) )[0]
	for i in range(len(ceres_inorder)): plt.axvline(ceres_lines[ceres_inorder[i]],ls='--',c='red',alpha=0.5,zorder=5)

	lovis_inorder = np.where((lovis_lines > np.min(wave)) & (lovis_lines < np.max(wave)) )[0]
	for i in range(len(lovis_inorder)): 
		plt.axvline(lovis_lines[lovis_inorder[i]],ls=':',c='k',alpha=0.5,zorder=10)
		plt.text(lovis_lines[lovis_inorder[i]], 0.0, i)
		
	ok_line = []
	yes_results = []
	no_results = []
	no_reasons = []
	
	for i,line in enumerate(lovis_lines[lovis_inorder]):
		
		# ===== Subrange
		subspec = np.where((wave > line-3.*resolution0) & (wave < line+3.*resolution0))[0]
		x0, y0, xpix0 = wave[subspec], flux[subspec], Xpix[subspec]
		x, y, xpix    = x0[~np.isnan(y0)], y0[~np.isnan(y0)], xpix0[~np.isnan(y0)]
		
		resolution = 2.3 * np.mean(x[1:-1]-x[0:-2])
		if len(y) == 0: continue
		
		# ===== Look for the closest line and decide if fitting 1 or 2 Gaussians
		twoGauss = None
		diff 		= line-lovis_lines[lovis_inorder]
		sorting 	= np.argsort(np.abs(diff))
		intensity 	= lovis_int[lovis_inorder]
		int_ratio 	= intensity[sorting[1]]/intensity[sorting[0]]
		blend_line 	= lovis_lines[lovis_inorder[sorting[1]]]
		if   (np.abs(diff[sorting[1]]) > 3.0*resolution): 
			twoGauss = False
		elif (np.abs(diff[sorting[1]]) > 1.5*resolution):
			if (int_ratio > 0.1)  : twoGauss = True
			if (int_ratio < 0.1) : twoGauss = False
		elif ( (np.abs(diff[sorting[1]]) < 1.5*resolution) & (int_ratio < 0.1) ):
			twoGauss = False

		# ===== Try fitting 1-Gaussian
		if twoGauss == False:
			try:
				popt, pcov = curve_fit(gauss_function, x, y, 
									   p0 = [np.max(y), line, resolution, 0.0],
									   bounds=([0.0,line-resolution,0.0,-np.inf],
											   [np.inf,line+resolution,3*resolution,np.inf]))
				perr = np.sqrt(np.diag(pcov))
			except:
				popt = np.zeros(4)
			
			ymodel = gauss_function(x,popt[0],popt[1],popt[2],popt[3])

			try:
				xpix0 = np.interp(line,x,xpix)
				poptX, pcovX = curve_fit(gauss_function, xpix, y, 
									   p0 = [np.max(y), xpix0, 2.3, 0.0],
									   bounds=([0.0,xpix0-2.3,0.0,-np.inf],
											   [np.inf,xpix0+2.3,3*2.3,np.inf]))
				perrX = np.sqrt(np.diag(pcovX))
			except:
				poptX = np.zeros(4)
	

		# ===== Try fitting 2-Gaussians
		if twoGauss == True:
			try:
				popt, pcov = curve_fit(twoGauss_function, x, y, 
									   p0 = [np.max(y), line, resolution, 0.0, np.max(y)*int_ratio, blend_line, resolution],
									   bounds=([0.0,line-resolution,0.0,-np.inf,0.0,blend_line-resolution,0.0],
											   [np.inf,line+resolution,3*resolution,np.inf,np.inf,blend_line+resolution,3*resolution]))
				perr = np.sqrt(np.diag(pcov))
			except:
				popt = np.zeros(7)

			ymodel = twoGauss_function(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6])			
	
			try:
				xpix0 = np.interp(line,x,xpix)
				xpix0_blend = np.interp(blend_line,x,xpix)
				poptX, pcovX = curve_fit(twoGauss_function, xpix, y, 
									   p0 = [np.max(y), xpix0, 2.3, 0.0, np.max(y)*int_ratio, xpix0_blend, 2.3],
									   bounds=([0.0,xpix0-2.3,0.0,-np.inf,0.0,xpix0_blend-2.3,0.0],
											   [np.inf,xpix0+2.3,3*2.3,np.inf,np.inf,xpix0_blend+2.3,3*2.3]))
				perrX = np.sqrt(np.diag(pcovX))
			except:
				poptX = np.zeros(7)
			
		if twoGauss == None:
			popt = np.zeros(4)
			ymodel = x*0.0
			poptX = np.zeros(4)


		#| Region around the maximum to check fitting: +/- 2 pix ~ 1xFWHM
		elem_check = np.where( (x > line-1.*resolution) & (x < line+1.*resolution) )[0]	

		ground_noise = np.nanmedian(flux[400:-400])/10.
		snr = popt[0]/ground_noise
		resid = np.nanstd(y[elem_check]-ymodel[elem_check])/ground_noise
		chi2 = np.nansum((y[elem_check]-ymodel[elem_check])**2/(9.*np.abs(y[elem_check]))) / (1.*len(x[elem_check]))
	
		if i == 20000:
			plt.close()
			plt.errorbar(x,y,yerr=3.*np.sqrt(y))
			#plt.plot(x,y)
			plt.plot(x[elem_check],y[elem_check])
			plt.plot(x,ymodel,c='red')
			print twoGauss, int_ratio, np.abs(diff[sorting[1]])/resolution, resolution
			plt.show()
			plt.close()
	
		#diff = np.sort(np.abs(line-lovis_lines[lovis_inorder]))
		
		if ((np.abs(popt[1]-line)*1.e3 < 10.) & (chi2 < 10.) & (snr > 5) ): 
			
			if (np.abs(diff[1]) > 2.0*resolution): 
				ok = 'yes'
			elif (np.abs(diff[1]) > 1.5*resolution):
				ok = 'yes' if ((int_ratio < 0.05) | (int_ratio > 0.3)) else ' no'
			elif (np.abs(diff[1]) < 1.5*resolution):
				ok = 'yes' if (int_ratio < 0.05) else ' no'
			else:
				ok = ' no'					
		else:
			ok = ' no'
					
		if ok == 'yes': ok_line.append(line)
		#print i, ok, np.abs(popt[1]-line)/resolution, chi2, snr, np.abs(diff[sorting[1]])/resolution,int_ratio, intensity[sorting[0]], intensity[sorting[1]], twoGauss
		if ok == 'yes':
			#print i, ok, round(poptX[1],2), line, intensity[sorting[0]], twoGauss
			yes_results.append([i, ok, round(poptX[1],2), line, intensity[sorting[0]], lovis_Species[lovis_inorder[sorting[0]]], twoGauss])
		else:
			no_results.append([i, ok, round(poptX[1],2), line, intensity[sorting[0]], lovis_Species[lovis_inorder[sorting[0]]], twoGauss])
			no_reasons.append([i, ok, np.abs(popt[1]-line)*1.e3, chi2, snr, np.abs(diff[1])/resolution, int_ratio, intensity[sorting[0]], intensity[sorting[1]], twoGauss])

		
	ok_line = np.array(ok_line)
	print oo,len(ok_line), len(ceres_inorder)
	for i in ok_line: plt.axvline(i,ls=':',c='blue',zorder=15)
	
	if 1:
		fyes = open('auxiliar/yes_cafeX_order_'+str(order0+oo).zfill(3)+'.dat','w')
		fno  = open('auxiliar/no_cafeX_order_'+str(order0+oo).zfill(3)+'.dat','w')
		f    = open('auxiliar/cafeX_order_'+str(order0+oo).zfill(3)+'.dat','w')
		fyes.write("%5s %10s %20s %15s %10s \n" % ("# ID","Xpix","Wavelength","Intens","Line"))
		f.write(   "%5s %10s %20s %15s %10s \n" % ("# ID","Xpix","Wavelength","Intens","Line"))
		fno.write( "%5s %10s %20s %15s %10s \n" % ("# ID","Xpix","Wavelength","Intens","Line"))

		for hh in yes_results:
			fyes.write("%5i %10.1f %20.8f %15.1f %10s \n" % (hh[0],hh[2],hh[3],hh[4],hh[5]))
			f.write(   "%5i %10.1f %20.8f %15.1f %10s \n" % (hh[0],hh[2],hh[3],hh[4],hh[5]))
		for hh in no_results:
			fno.write("%5i %10.1f %20.8f %15.1f %10s \n" % (hh[0],hh[2],hh[3],hh[4],hh[5]))
		fyes.close()
		fno.close()
		f.close()
	
	fnoreasons = open('auxiliar/noreasons_cafeX_order_'+str(order0+oo).zfill(3)+'.dat','w')
	fnoreasons.write("%5s %5s %10s %10s %10s %10s %10s %10s %10s \n" % ("ID","ok","Diff(mA)","chi2","SNR","Diff/R","I1/I0","I0","I1"))
	for hh in no_reasons: 
		fnoreasons.write("%5s %5s %10.1f %10.1f %10.1f %10.1f %10.3f %10.1f %10.1f \n" % (hh[0],hh[1],hh[2],hh[3],hh[4],hh[5],hh[6],hh[7],hh[8]))
	fnoreasons.close()
	
	plt.ylim(-500,3000)
	#plt.show()
	#sys.exit()
	plt.close()
