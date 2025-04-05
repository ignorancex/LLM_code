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
from termcolor import colored

import GLOBALutils
import CAFEutilities
import CAFEx_SetupFile as CS


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


# Gaussian function
def gauss_function(x, a, x0, sigma, zero):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + zero

# Two Gaussian function
def twoGauss_function(x, a, x0, sigma, zero, a2, x02):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + zero + a2*np.exp(-(x-x02)**2/(2*sigma**2))



def FindWaveSolution(x_arc,xshift,yshift,cv, plot_name='tmp.pdf'): #x_arc,arcs,arc_names

	CS.var.set_OrderProp(CAFEutilities.jdnight(cv.night))

	# Data (this is to be passed)
	#x_arc = pyfits.getdata(x_arc)#'/Users/lillo_box/00_Instrumentation/CAFE/CAFExtractor/cafextractor/test_data/11_REDUCED/120705/auxiliar/X_arc__120705_0023.fits')
	x_arc = np.swapaxes(x_arc,0,1)

	nframes, nord, npix = np.shape(x_arc)

	# X values
	xpix = np.linspace(0.,npix-1,npix) # "+0.5 pix to set the center of the pixel"

	ncoef_x            = CS.nx
	ncoef_m            = CS.nm
	npar_wsol = (min(ncoef_x,ncoef_m) + 1) * (2*max(ncoef_x,ncoef_m) - min(ncoef_x,ncoef_m) + 2) / 2
	n_useful = 80
	ro0      = CS.var.order0

	#t = np.genfromtxt('/Users/lillo_box/Desktop/thar_px_wave.dat',skip_header=1)
	#lam = t[:,3]

	lam_residuals 		= []
	rv_residuals 		= []
	rv_residuals_order 	= []
	lam_residuals_order = []
	Nlines_used_order 	= []
	Nlines_total_order 	= []
	Nlines_used_all		= 0
	
	All_Pixel_Centers0	= []
	All_Wavelengths0	= []
	All_Orders0			= []
	All_residuals0		= []
	All_residuals_ms0	= []
	All_Pixel_Centers	= []
	All_Wavelengths		= []
	All_Orders			= []
	All_Residuals		= []


	All_now_peak		= []
	All_true_peakL		= []
	All_now_orders		= []
	
	WC_coeffs			= []
	WC_covs				= []
	WC_validity			= []
	Wavelength			= []
	Order 				= []
	
	Nord 				= CS.var.Nominal_Nord	#82
	delta_refThAr		= np.zeros(Nord)
	
	
	xc_global			= np.zeros((Nord,400))

	# ==================================
	#     LOOP in ORDERS
	# ==================================
	'''
	Loop in order to find a rough solution based on previous solutions from the 
	reference arc.
	'''

	print "    --> Individual order analysis..."

	print "        --> Cross-correlation of OBSERVED ThAr lines against ThAr LINELIST"
	print "        --> X-shifts are calculated for every order and should be very similar."
	print "        --> If X-shifts are very different among the orders ==> Bad order identification"
	print " "
	print "            Order ID, X-shift against linelist"

	for i in np.arange(Nord):

		#| Debugging lines...
		#x_arc[1,i,:] = np.flip(x_arc[1,i,:],0)

		#| Select the optimal zero-order solution according to observing date
		fileref = '../ReferenceFrames/ThAr_ReferenceLines/00_CAFE2_ThAr_dates.lis'
		fref = np.genfromtxt(fileref,dtype=None,names=True,encoding='ascii')
		jdnight = CAFEutilities.jdnight(cv.night)
		id = fref['ID'][np.max(np.where((fref['Datestart'] < jdnight) & (fref['Dateend'] > jdnight))[0])]
		
		#| Read the ThAr catalogue lines for this order from the CERES line catalogue
		filename_ceres = '../ReferenceFrames/ThAr_ReferenceLines/cafe2_'+str(id).zfill(3)+'/ceres_order_'+np.str(i+CS.var.order0).zfill(3)+'.dat'#+'.iwdat'
		filename_cafex = '../ReferenceFrames/ThAr_ReferenceLines/cafe2_'+str(id).zfill(3)+'/cafeX_order_'+np.str(i+CS.var.order0).zfill(3)+'.dat'
		
		#| Choose HARPS or CERES files for ThAr mask depending on order
		if os.path.isfile(filename_cafex):
			f = open(filename_cafex).readlines()
			file_id = 'cafex'
		else:
			try:
				f = open(filename_ceres).readlines()
				file_id = 'ceres'
			except:
				print "        --> WARNING: file"+filename_ceres+" not found. Skipping order..."
				Nlines_total_order.append(0)
				continue
	
		#| Divide the file into columns and extract pix,wave
		pix2 = np.array([])
		lam2 = np.array([])
		nblended = []
		for tt,line in enumerate(f):
			if file_id == 'ceres':
				w = line.split()
				nblended = w[0]
				if nblended == '1':
					lam2 = np.append(lam2,np.float(w[2]))
					pix2 = np.append(pix2,np.float(w[1])+0.)   # "+36" is the difference 
																# between CAFE_2012 and CERES pixel 
																# position of the ThAr lines in the detector.
																#If using cafeX* then should be 0
			if file_id == 'cafex':
				if tt == 0: continue
				w = line.split()
				pixnonzero = w[1]
				if np.float(pixnonzero) > 0.:
					lam2 = np.append(lam2,np.float(w[2]))
					pix2 = np.append(pix2,np.float(w[1])+0.)   # "+0" is the difference 
																# between CAFE_2012 and CERES pixel 
																# position of the ThAr lines in the detector.
																#If using cafeX* then should be 0

		Nlines_total_order.append(len(pix2))
		
		if len(np.nonzero(x_arc[1,i,:].reshape(-1))[0]) == 0: 
			delta = 0
			delta_refThAr[i] = np.nan
			print "        --> WARNING: skipping fake order #"+str(i)
			continue
	
		#| Create binary mask based on present ThAr lines in the arc
		binmask = np.zeros(2048)
		binmask[pix2.astype(np.int64)] = 1.0
	
		#| Find peaks in the ThAr extracted spectrum		
		peakind = find_peaks_cwt(x_arc[1,i,:],np.arange(0.5,10),min_snr=5.)
		if len(peakind) == 0:
			print "        --> WARNING: no ThAr line found in order #"+str(i)
			continue
		obsmask = np.zeros(2048)
		obsmask[peakind] = 1.0

		#| Cross-correlation to find X-axis shift against reference values
		ml = pix2 - 2
		mh = pix2 + 2
		xc,offs = GLOBALutils.XCorPix(obsmask, ml, mh, del0=0, del_width=150, del_step=1.)
		#xc_global[i,:] = xc
		ind_max = np.argmax( xc )
		delta   = offs[ind_max] 

		#| Fit the maximum peak of the cross-correlation
		tofit = np.where((offs > delta-10) & (offs < delta+10))[0]
		try:
			popt1, pcov1 = curve_fit(gauss_function, offs[tofit], xc[tofit], p0 = [np.max(xc), delta, 2., 0.0])
			delta = popt1[1]
		except:
			delta = 0.0
			
		delta_refThAr[i] = delta
		
		print "            Order = "+str(i),",   Xshift = ",delta


		"""
		Some debugging stuff...
		"""
		#| Debugging lines...
# 		xc_stack = np.sum(xc_global,axis=0)
# 		print np.shape(xc_stack)
# 		tofit = np.where((offs > delta-10) & (offs < delta+10))[0]
# 		popt1, pcov1 = curve_fit(gauss_function, offs, xc_stack, p0 = [np.max(xc_stack), offs[np.argmax(xc_stack)], 2., 0.0])
# 		delta = popt1[1]
# 		delta_refThAr[i] = delta
		#plt.close()
# 		plt.plot(np.arange(len(x_arc[1,i,:])),x_arc[1,i,:],c='k')
# 		for ppp in pix2: plt.axvline(ppp,ls=':',c='red')
# 		plt.show()
# 		sys.exit()
		
# 		plt.plot(offs,xc)
# 		plt.axvline(delta)
# 		plt.show()
# 		sys.exit()

		#| Debugging plots...
# 		plt.close()
# 		plt.figure(1)
# 		plt.plot(xpix,x_arc[1,i,:])
# 		for l in pix2: plt.axvline(l,ls=':')
# 		print i, "Delta = ",delta,' pix'
# 		plt.show()
# 		sys.exit()
# 	
# 		plt.figure(2)
# 		plt.plot(offs,xc)
# 		plt.show()
# 		plt.close(2)
# 		plt.close()
# 		sys.exit()
	
		# ==================================
		#     Gaussian fit to ThAr lines
		# ==================================

		#| Initialize values  
		now_peak 	= []		# Fitted center of the ThAr line [pix]
		now_peak_err= []		# Fitted center of the ThAr line [pix]
		true_peakL 	= []
		true_peakX 	= []
		now_order	= []
		guess0 		= []
		Nvalid 		= 0
		Nunfitted	= 0
	
		#| Loop for each defined ThAr line
		for j,l in enumerate(pix2):
		
			#| Refine maximum location of the line @ +/- 5 pixels around expected location 
			#| NOTE: If another line is closer than +/- 5 pix, we might get in trouble!
			elem = np.where( (xpix > l+delta_refThAr[i]-5) & (xpix < l+delta_refThAr[i]+5) )
			x0 = xpix[elem]
			y0 = x_arc[1,i,elem].reshape(-1)		
			try:
				newX = x0[np.argmax( y0 )]
			except:
				Nunfitted +=1
				continue
		
		
			#| Define x-range to use in the fit: +/- 10 pixels around previously refined location 
			elem = np.where( (xpix > newX-15) & (xpix < newX+15) )		
			x = xpix[elem]
			y = x_arc[1,i,elem].reshape(-1)
		
			#| Region around the maximum to check fitting: +/- 2 pix ~ 1xFWHM
			elem_check = np.where( (x > newX-2) & (x < newX+2) )	
		
			# ============================================================================
			
			RON = 3.3
			
			#| Try 1-Gaussian fit:
			try:
				popt1G, pcov1G = curve_fit(gauss_function, x, y, 
									       p0 = [np.max(y), newX, 2., 0.0],
									       bounds=([0.0,newX-5,0.0,-0.1],[np.inf,newX+5,20.,0.1]))
				perr1G = np.sqrt(np.diag(pcov1G))
				ymodel1G = gauss_function(x,popt1G[0],popt1G[1],popt1G[2],popt1G[3])
				yerr = 3.*np.sqrt(y)
				chi2_1G = np.nansum(((y[~np.isnan(yerr)]-ymodel1G[~np.isnan(yerr)])/(RON+yerr[~np.isnan(yerr)]))**2)   # 
 				bic_1G = chi2_1G + 1.*len(popt1G)*np.log(1.*len(x))
			except:
				bic_1G = np.inf

			#| Try 2-Gaussian fit:
			try:
				popt2G, pcov2G = curve_fit(twoGauss_function, x, y, 
									   p0 = [np.max(y), newX, 2., 0.0, 1., newX],
									   bounds=([0.0   ,newX-2.3, 0.0, -0.1,    0.0, np.min(x)-5.],
									   		   [np.inf,newX+2.3, 20.,  0.1, np.inf, np.max(x)+5.]))
				perr2G = np.sqrt(np.diag(pcov2G))
				ymodel2G = twoGauss_function(x,popt2G[0],popt2G[1],popt2G[2],popt2G[3],popt2G[4],popt2G[5])
				yerr = 3.*np.sqrt(y)
				chi2_2G = np.nansum(((y[~np.isnan(yerr)]-ymodel2G[~np.isnan(yerr)])/(RON+yerr[~np.isnan(yerr)]))**2 ) # /(3.*np.sqrt(y))**2 
				bic_2G = chi2_2G + 1.*len(popt2G)*np.log(1.*len(x))
			except:
				bic_2G = np.inf
						
			if 	bic_2G < bic_1G:
				popt, pcov, perr 	= popt2G, pcov2G, perr2G
				ymodel 		= ymodel2G		
				flag 		= 'fitted_2G'
				chi2 		= np.nansum((y[elem_check]-ymodel[elem_check])**2/(9.*np.abs(y[elem_check]))) / (1.*len(x[elem_check]))
			elif bic_1G < bic_2G:
				popt, pcov, perr 	= popt1G, pcov1G, perr1G
				ymodel 		= ymodel1G		
				chi2 		= np.nansum((y[elem_check]-ymodel[elem_check])**2/(9.*np.abs(y[elem_check]))) / (1.*len(x[elem_check]))
				flag 		= 'fitted_1G'
			elif bic_2G == np.inf:
				flag 		= 'no_fitted'
				chi2 		= np.inf 


			if ((flag != 'no_fitted') & (chi2 < 5.)):
				now_peak.append(popt[1])  
				now_peak_err.append(perr[1])
				true_peakX.append(l+delta_refThAr[i])
				true_peakL.append(lam2[j])
				now_order.append(i)
				Nvalid += 1	
			else:
				Nunfitted +=1

			
		#| Convert to numpy array
		All_now_peak.append(np.array(now_peak))	
		All_true_peakL.append(np.array(true_peakL))
		All_now_orders.append(np.array(now_order))	
		now_peak 	= np.array(now_peak)	
		true_peakL 	= np.array(true_peakL)
		now_order	= np.array(now_order)

		"""
		Some debugging stuff...
		"""
		#plt.errorbar(true_peakL, now_peak-true_peakX,yerr=now_peak_err,fmt ='o')
		#plt.show()
		#plt.close()
		#sys.exit()
	
		
		# ==============================================
		#    Local Wavelength Solution for each order
		# ==============================================
		'''
		First-order fit to the wavelength solution per order. This is used to
		perform a 3-sigma clipping to remove the ThAr lines that are not well fitted
		'''
		
		if 1:
			#| In principle, use all valid lines
			use = np.ones(len(now_peak))
	
			#| Iterate 5 times rejecting the worse lines (5-sigma clipping)
		
			for iter in range(5):
				skip_flag = iter
				xuse,yuse = now_peak[use == 1], true_peakL[use == 1]			

				if len(xuse) >= 7:
					coeff,cov = np.polyfit(xuse,yuse,3,cov=True)
				elif iter > 0:
					break
				else:
					#Wavelength.append(np.zeros(2048)*np.nan)
					#Nlines_used_order.append(np.nan)
					#lam_residuals_order.append(np.nan)
					#rv_residuals_order.append(np.nan)
					print "        --> WARNING: too few ThAr lines ("+str(len(xuse))+") found in order #"+str(i)+" ...skipping order"
					break
						
			
				#if ((iter == 0) & len(xuse) < 8):
				#	#coeff = GLOBALutils.Cheby_Fit(xuse,yuse,5,2048.)
				#	coeff,cov = np.polyfit(xuse,yuse,3,cov=True)	
				#else:
				#	coeff,cov = np.polyfit(xuse,yuse,3,cov=True)
				#	#coeff = GLOBALutils.Cheby_Fit(xuse,yuse,5,2048.)
				p = np.poly1d(coeff)
				#res = true_peakL-GLOBALutils.Cheby_eval(coeff,now_peak,2048.)
				res = true_peakL-p(now_peak)	
				good_lines = np.where(np.abs(res) < 3.*sigmaG(res))[0]
				bad_lines = np.where(np.abs(res) > 3.*sigmaG(res))[0]
				use[bad_lines] = 0
	

	
			if ((len(xuse) < 7) & (skip_flag == 0)): continue
		
			#Nlines_used = len(np.where(use == 1)[0])
			#Nlines_used_all += Nlines_used
		#	residuals = true_peakL[use == 1]-GLOBALutils.Cheby_eval(coeff,now_peak[use == 1],2048.) #p(now_peak[use == 1])
			residuals = true_peakL[use == 1]-p(now_peak[use == 1]) #p(now_peak[use == 1])
			#rv_residuals = np.append(rv_residuals, residuals/ true_peakL[use == 1] * c.c)
			#lam_residuals = np.append(lam_residuals, residuals*1.e3)  # in mA
	
			#rv_residuals_order.append(sigmaG(res[good_lines] / true_peakL[good_lines]) * c.c.value)
			#lam_residuals_order.append(sigmaG(res[good_lines] ))
			#Nlines_used_order.append(Nlines_used)
	
			All_Pixel_Centers0.append(now_peak[use == 1])
			All_Wavelengths0.append(true_peakL[use == 1])
			All_Orders0.append(np.zeros(len(now_peak[use == 1])) + i+CS.var.order0   )
			All_residuals_ms0.append(res/true_peakL * c.c.value)
			All_residuals0.append(res*1.e3)
		
			#WC_coeffs.append(coeff)
			#WC_covs.append(cov)
			#WC_validity.append(str(round(np.min(true_peakL[use == 1]),1))+"-"+str(round(np.max(true_peakL[use == 1]),1)))
		
			#Wavelength.append(p(np.arange(2048)))
			
			#print i+60, delta,Nlines_used,len(now_peak)-Nlines_used,sigmaG(res[good_lines] / true_peakL[good_lines]) * c.c,np.max(now_peak[use == 1])-np.min(now_peak[use == 1]),37000./c.c.value *np.max(true_peakL[use == 1]) 
	
	
	
			"""
			Some debugging stuff...
			"""

# 			if i < 10:
# 				plt.plot(now_peak[use == 1],residuals,'o',c='black')
#  				plt.plot(p(xpix), x_arc[1,i,:].reshape(-1))
#  				for lll in true_peakL: plt.axvline(lll,ls=':',c='red')
#  				for ttt in true_peakL[use == 1]: plt.axvline(ttt,ls=':',c='green')
# 				plt.show()
# 				plt.close()
# 			else:
#				break
# 			#| Plot residuals
# 			plt.plot(now_peak[use == 1],residuals/ true_peakL[use == 1] * c.c/np.sqrt(2500.),'o',c='black')
# 
# 			#| Plot pixel in X and wavelength in Y
# 			xall = np.arange(2048)
# 			plt.plot(now_peak[use == 1], true_peakL[use == 1] ,'o',c='red')
# 			plt.plot(xall, GLOBALutils.Cheby_eval(coeff,xall,2048.) ,c='black')
# 	
# 			#| Plot pixel in X and wavelength in Y
# 			plt.text(now_peak[use == 1],true_peakL[use == 1]-p(now_peak[use == 1]), np.repeat(i,len(np.where(use == 1)[0])))
# 			plt.plot(now_peak[good_lines],true_peakL[good_lines]-p(now_peak[good_lines]),'o',c='r')
	


	#| Warning for bad order identification:
	if np.nanstd(delta_refThAr) > 5.:
		print colored("        --> IMPORTANT WARNING: Xshifts in ThAr are very sparse ==> ", "red")
		print colored("            ==> Probably due to a bad identification of the orders", "red")
		print colored("                Please check if orders moved and contact J. Lillo-Box", "red")
	else:
		print colored("        --> OK! X-shifts from all orders agree!  ==> ", "green")
	

	#| Debugging plots...
# 	plt.figure(1)
# 	plt.hist(np.concatenate(np.array(All_residuals0),axis=0),bins=30)
# 	plt.figure(2)
# 	plt.hist(np.concatenate(np.array(All_residuals_ms0),axis=0),bins=30)
# 	plt.show()
# 	sys.exit()


	# ==================================
	#     GLOBAL WAVELENGTH SOLUTION
	# ==================================
	'''
	Global solution in 2D taking into account both the pixel and order numbers. 
	The 
	'''
	print "    --> Starting the GLOBAL (2D) wavelength fit..."

	# ===== Compute Global Wavelentgh Solution
	p0    = np.zeros( npar_wsol )
	
	_All_Pixel_Centers 	= np.concatenate(np.array(All_Pixel_Centers0),axis=0)
	_All_Wavelengths 	= np.concatenate(np.array(All_Wavelengths0),axis=0)
	_All_Orders 		= np.concatenate(np.array(All_Orders0),axis=0)
	
	p1, G_pix, G_ord, G_wav, II, rms_ms, G_res, cov1 = GLOBALutils.Fit_Global_Wav_Solution(_All_Pixel_Centers, _All_Wavelengths,\
							     _All_Orders, np.ones(len(_All_Orders)), p0, Cheby=True,       \
							     order0=CS.var.order0, ntotal=CS.var.Nominal_Nord, maxrms=200, Inv=True, minlines=200,  \
							     npix=2048.,nx=CS.nx,nm=CS.nm)

	# ===== Plot results of Wavelentgh Solution
	plt.close()
	fig = plt.figure(figsize=(6,4))
	plt.plot(G_wav,G_res*1.e3,'o',c='tomato',alpha=0.6)
	plt.xlabel('Wavelength (A)')
	plt.ylabel('Residuals (mA)')
	plt.axhline(np.nanmedian(G_res)*1.e3,ls='--',c='k')
	plt.axhline(np.nanmedian(G_res)*1.e3-np.sqrt(np.var(G_res*1.e3)),ls=':',c='k')
	plt.axhline(np.nanmedian(G_res*1.e3)+np.sqrt(np.var(G_res*1.e3)),ls=':',c='k')
	plt.savefig(cv.aux_dir+'/WC_'+plot_name+'.pdf',bbox_inches="tight")
	plt.close()


	# ===== Calculate Wavelength matrix:
	WavelengthGlobal = []
	for oo in range(CS.var.Nominal_Nord):
		m = oo + CS.var.order0
		x = 1.*np.arange(npix)
		chebs = GLOBALutils.Calculate_chebs(x, m, Inverse=True,order0=CS.var.order0,ntotal=CS.var.Nominal_Nord,npix=npix,nx=CS.nx,nm=CS.nm)
		ww = (1.0/m) * GLOBALutils.Joint_Polynomial_Cheby(p1,chebs,CS.nx,CS.nm) 
		WavelengthGlobal.append(ww)

	# ===== Compute statistics from Global Wavelength Solution:
	Nused = 1.*len(II)
	for oo in np.arange(Nord):
		this = np.where(G_ord == oo+CS.var.order0)[0]

		Order.append(oo+CS.var.order0)
		rv_residuals_order.append(np.sqrt(np.var( G_res[this] / G_wav[this] * c.c.value)))
		lam_residuals_order.append(np.sqrt(np.var(G_res[this])))
		Nlines_used_order.append(len(this))

		All_Pixel_Centers.append(G_pix[this])
		All_Wavelengths.append(G_wav[this])
		All_Residuals.append(G_res[this]*1.e3)
		All_Orders.append(np.zeros(len(this)) + oo + CS.var.order0   )
	
		WC_coeffs.append(p1)
		WC_covs.append(cov1)
		try:
			WC_validity.append(str(round(np.min(G_wav[this]),1))+"-"+str(round(np.max(G_wav[this]),1)))
		except:
			WC_validity.append(" ")
		
	print "    --> ...done with rms = ",np.sqrt(np.var(G_res*1.e3)), " mA precision"

	# ==================================
	#     Log results
	# ==================================
	f = open(cv.aux_dir+'/log_'+cv.night,'a')
	f.write("===================\n")
	f.write("Wavelength Solution\n")
	f.write("===================\n")

	f.write("+ General results\n")
	f.write("  --> File = "+plot_name+".fits\n")
	f.write("  --> Number of lines used = "+np.str(Nused)+"\n")
	f.write("  --> Wavelength residuals per line = "+np.str(round(np.sqrt(np.var(G_res)*1.e3),2))+" mA\n")
	f.write("  --> RV residuals per line = "+np.str(round(rms_ms,2))+" m/s\n")
	f.write("  --> Estimated RV precision = "+np.str(round(rms_ms/np.sqrt(Nused),2))+" m/s\n")
	f.write("  --> Pix_shift from reference ThAr mask = "+np.str(round(np.nanmin(delta_refThAr), 2))+" - " +np.str(round(np.nanmax(delta_refThAr), 2))+' pix.\n')
#	f.write("  --> Pix_shift from reference ThAr mask = "+np.str(round(np.nanmin(delta), 2))+' pix.\n')

	WCdict =  { 'rv_residuals_order':rv_residuals_order,			# RV residuals dispersion per line for each order
				'lam_residuals_order':lam_residuals_order,			# Wavelength residuals dispersion per line for each order
				'Nlines_used_order':Nlines_used_order,				# Number of lines used to compute Wavelength Calibration for each order
				'Nlines_total_order':Nlines_total_order,			# Number of detected peaks in order
				'All_Pixel_Centers':All_Pixel_Centers,				# Pixel of used lines for each order
				'All_Wavelengths':All_Wavelengths,					# Wavelength of used lines for each order
				'All_Residuals':All_Residuals,						# Wavelength residuals for each line
				'All_Orders':All_Orders,							# Corresponding order of the used lines for each order
				'Order':Order,										# True order number for each order
				'Mean_lam_residual':round(np.median(G_res),2),		# Full median of all residuals (should be 0) 
				'lam_residuals':round(np.sqrt(np.var(G_res)),2),	# Dispersion of all residuals (should be around 3 mA)
				'RV_error_budget':round(rms_ms/np.sqrt(Nused),2),	# Estimated RV precision achievable by wavelength solution
				'Nlines_total':Nused,								# Final number of lines used
				'WC_validity':WC_validity							# Validity range per order (min-max wavelength of lines finally used)
				}

	
	x_arc = np.append(x_arc,np.atleast_3d(np.array(WavelengthGlobal).T).T,axis=0)	
	
	return np.array(WC_coeffs),np.array(WC_covs),WCdict,x_arc



def WavelengthCalibration(ArcList,arcs,arc_names,cv,xshift,yshift, type):
	
	S_frame = []
	for ii,Xarc in enumerate(ArcList):
		already_done = os.path.isfile(cv.aux_dir+'WC_'+arc_names[ii])
		filename, file_extension = os.path.splitext(arc_names[ii])
		if already_done == False:
			
			#| Compute wavelength solution
			
			coeff,cov,WCdict,SS_frame = FindWaveSolution(Xarc,xshift,yshift,cv, plot_name=filename)
			
			np.savez(cv.aux_dir+'/WC_'+filename, coeff=coeff,cov=cov,WCdict=WCdict)

			#| Save results	
			data = Table([WCdict['Order'],WCdict['Nlines_used_order'],WCdict['Nlines_total_order'],\
						  WCdict['lam_residuals_order'],WCdict['rv_residuals_order']], \
						  names=['# Order', 'Nl_used','Nl_all','wave_res (mA)', 'rv_res (m/s)'])
			
			ascii.write(data, cv.aux_dir+'/WC_'+filename+'.dat',format ='tab')
			
			print '       -> '+arc_names[ii]+'...extracted!'

			hdu = pyfits.PrimaryHDU( SS_frame )
			hdu.writeto( cv.aux_dir+'WC_'+arc_names[ii] )
			
			#CAFEutilities.save_final_file(arc_names[ii], Xarc, 'WC_'+arc_names[ii], cv, type)
			S_frame.append(SS_frame)
			
		else:
			print '       -> '+arc_names[ii]+'...already extracted!'
			#CAFEutilities.save_final_file(arc_names[ii], Xarc, 'WC_'+arc_names[ii], cv, type)
			SS_frame = fits.open(cv.aux_dir+'WC_'+arc_names[ii])
			S_frame.append(SS_frame[0].data)
			w_results = np.load(cv.aux_dir+'WC_'+filename+'.npz',allow_pickle=True)
			WCdict = w_results["WCdict"].tolist()
		
		lam_residuals_order = np.array(WCdict['lam_residuals_order'])	
# 		plt.plot(np.arange(len(lam_residuals_order)), lam_residuals_order*1.e3, 'o',c='red', markeredgecolor='k')
# 		plt.xlabel('Order')
# 		plt.ylabel('Residuals (mA)')
# 		for oo in np.arange(len(lam_residuals_order)): 
# 			mycolor = 'gray' if lam_residuals_order[oo] < 10. else 'red'
# 			plt.axvline(oo,ls='--',c=mycolor,alpha=0.5)
# 		plt.axhline(np.nanmedian(lam_residuals_order)*1.e3,c='k',ls='--')
# 		plt.ylim(0,10)
# 		plt.savefig(cv.aux_dir+'WC_Resid_'+filename+'.pdf')
# 		plt.close()
		
		#sys.exit()
	return S_frame
	
	
	
	
	
	
	
	
	









