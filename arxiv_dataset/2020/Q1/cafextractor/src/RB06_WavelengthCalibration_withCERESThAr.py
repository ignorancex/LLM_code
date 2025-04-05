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
def twoGauss_function(x, a, x0, sigma, zero, a2, x02, sigma2):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + zero + a2*np.exp(-(x-x02)**2/(2*sigma2**2))



def FindWaveSolution(x_arc,xshift,yshift,cv, plot_name='tmp.pdf'): #x_arc,arcs,arc_names

	# Data (this is to be passed)
	#x_arc = pyfits.getdata(x_arc)#'/Users/lillo_box/00_Instrumentation/CAFE/CAFExtractor/cafextractor/test_data/11_REDUCED/120705/auxiliar/X_arc__120705_0023.fits')
	x_arc = np.swapaxes(x_arc,0,1)

	nframes, nord, npix = np.shape(x_arc)

	# X values
	xpix = np.linspace(0.,npix-1,npix) # "+0.5 pix to set the center of the pixel"

	ncoef_x            = 3
	ncoef_m            = 8
	npar_wsol = (min(ncoef_x,ncoef_m) + 1) * (2*max(ncoef_x,ncoef_m) - min(ncoef_x,ncoef_m) + 2) / 2
	n_useful = 80
	ro0      = 60

	#t = np.genfromtxt('/Users/lillo_box/Desktop/thar_px_wave.dat',skip_header=1)
	#lam = t[:,3]

	lam_residuals 		= []
	rv_residuals 		= []
	rv_residuals_order 	= []
	lam_residuals_order = []
	Nlines_used_order 	= []
	Nlines_total_order 	= []
	Nlines_used_all		= 0
	
	All_Pixel_Centers	= []
	All_Wavelengths		= []
	All_Orders			= []
	
	WC_coeffs			= []
	WC_covs				= []
	WC_validity			= []
	Wavelength			= []
	Order 				= []
	
	Nord = 82
	delta_refThAr		= np.zeros(Nord)



	# ==================================
	#     LOOP in ORDERS
	# ==================================
	for i in np.arange(Nord):

		Order.append(i+60)
		#print 'Order',i

		#| Read the ThAr catalogue lines for this order from the CERES line catalogue
		filename = '../ReferenceFrames/ThAr_ReferenceLines/order_'+np.str(i+60).zfill(3)+'.iwdat'
		##	filename = cv.ref_frames+'/order_'+np.str(i+60).zfill(3)+'.iwdat'
		f = open(filename).readlines()
	
		#| Divide the file into columns and extract pix,wave
		pix2 = np.array([])
		lam2 = np.array([])
		nblended = []
		for tt,line in enumerate(f):
			w = line.split()
			nblended = w[0]
			if nblended == '1':
				pix2 = np.append(pix2,np.float(w[1])+0.)   # "+36" is the difference 
															# between CAFE_2012 and CERES pixel 
															# position of the ThAr lines in the detector.
															#If using cafeX* then should be 0
				lam2 = np.append(lam2,np.float(w[2]))

		Nlines_total_order.append(len(pix2))
		
		if len(np.nonzero(x_arc[1,i,:].reshape(-1))[0]) == 0: 
			delta = 0
			delta_refThAr[i] = np.nan
			Wavelength.append(np.zeros(2048)*np.nan)
			Nlines_used_order.append(np.nan)
			lam_residuals_order.append(np.nan)
			rv_residuals_order.append(np.nan)
			print "    --> WARNING: skipping fake order #"+str(i)
			continue
	
		#| Create binary mask based on present ThAr lines in the arc
		binmask = np.zeros(2048)
		binmask[pix2.astype(np.int64)] = 1.0
	
		#| Find peaks in the ThAr extracted spectrum
		peakind = find_peaks_cwt(x_arc[1,i,:],np.arange(0.5,10),min_snr=5.)
		if len(peakind) == 0:
			print "    --> WARNING: no ThAr line found in order #"+str(i)
			Wavelength.append(np.zeros(2048)*np.nan)
			Nlines_used_order.append(np.nan)
			lam_residuals_order.append(np.nan)
			rv_residuals_order.append(np.nan)
			continue
		obsmask = np.zeros(2048)
		obsmask[peakind] = 1.0

		#| Cross-correlation to find X-axis shift against reference values
		ml = pix2 - 2
		mh = pix2 + 2
		xc,offs = GLOBALutils.XCorPix(obsmask, ml, mh, del0=0, del_width=100, del_step=1.)
		ind_max = np.argmax( xc )
		delta   = offs[ind_max] 

		#| Fit the maximum peak of the cross-correlation
		tofit = np.where((offs > delta-10) & (offs < delta+10))[0]
		popt1, pcov1 = curve_fit(gauss_function, offs[tofit], xc[tofit], p0 = [np.max(xc), delta, 2., 0.0])
		delta = popt1[1]
		delta_refThAr[i] = delta
		

		#| Debugging plots
		#plt.plot(xpix,x_arc[1,i,:])
		#for l in pix2: plt.axvline(l+delta,ls=':')
		#print i, "Delta = ",delta,' pix'
		#plt.show()
		#sys.exit()
	
		#plt.figure(2)
		#plt.plot(offs,xc)
		#plt.close(2)
		#plt.close()
		#sys.exit()
	
		# ==================================
		#     Gaussian fit to ThAr lines
		# ==================================

		#| Initialize values  
		now_peak 	= []		# Fitted center of the ThAr line [pix]
		now_peak_err= []		# Fitted center of the ThAr line [pix]
		true_peakL 	= []
		true_peakX 	= []
		guess0 		= []
		Nvalid 		= 0
		Nunfitted	= 0
	
		#| Loop for each defined ThAr line
		for j,l in enumerate(pix2):
		
			#| Refine maximum location of the line @ +/- 5 pixels around expected location 
			#| NOTE: If another line is closer than +/- 5 pix, we might get in trouble!
			elem = np.where( (xpix > l+delta-5) & (xpix < l+delta+5) )
			x0 = xpix[elem]
			y0 = x_arc[1,i,elem].reshape(-1)		
			newX = x0[np.argmax( y0 )]
		
			#| Define x-range to use in the fit: +/- 10 pixels around previously refined location 
			elem = np.where( (xpix > newX-15) & (xpix < newX+15) )		
			x = xpix[elem]
			y = x_arc[1,i,elem].reshape(-1)
		
			#| Region around the maximum to check fitting: +/- 2 pix ~ 1xFWHM
			elem_check = np.where( (x > newX-2) & (x < newX+2) )	
		
			#| Look for the closest line
			twoGauss = False
			diff = newX-peakind
			sorting = np.argsort(np.abs(diff))
			if ( (np.abs(diff[sorting[1]]) < 10) & (np.abs(diff[sorting[1]]) > 2) ): twoGauss = True

			#| Closest line to this ThAr line
			elemBlend = np.where( (xpix > newX-diff[sorting[1]]-10) & (xpix < newX-diff[sorting[1]]+10) )		
			xBlend = xpix[elemBlend]
			yBlend = x_arc[1,i,elemBlend].reshape(-1)
		
			#| LINE FITTING (1 or 2 Gaussians)
			try:
				if twoGauss == False:
					popt, pcov = curve_fit(gauss_function, x, y, 
										   p0 = [np.max(y), newX, 2., 0.0],
										   bounds=([0.0,newX-5,0.0,-np.inf],[np.inf,newX+5,20.,np.inf]))
					perr = np.sqrt(np.diag(pcov))
					ground_noise = np.nanmedian(x_arc[1,i,:])
					snr = popt[0]/ground_noise
					ymodel = gauss_function(x,popt[0],popt[1],popt[2],popt[3])
					resid = np.nanstd(y[elem_check]-ymodel[elem_check])/ground_noise
					chi2 = np.nansum((y[elem_check]-ymodel[elem_check])**2/y[elem_check]) / (1.*len(x))
				if twoGauss == True:
					#print "TwoGauss!"
					popt, pcov = curve_fit(twoGauss_function, x, y, p0 = [np.max(y), newX, 2., 0.0, np.max(yBlend), newX-diff[sorting[1]], 2.],bounds=([0.0,newX-5,0.0,-np.inf,0.0,newX-diff[sorting[1]]-5.,0.0],[np.inf,newX+5,20.,np.inf,np.inf,newX-diff[sorting[1]]+5.,20.]))
					perr = np.sqrt(np.diag(pcov))
					popt2, pcov2 = curve_fit(gauss_function, x, y, p0 = [np.max(y), newX, 2., 0.0],bounds=([0.0,newX-5,0.0,-np.inf],[np.inf,newX+5,20.,np.inf]))				
					ground_noise = np.median(x_arc[1,i,:])
					snr = popt[0]/ground_noise
					ymodel = twoGauss_function(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6])
					resid = np.std(y[elem_check]-ymodel[elem_check])/ground_noise
					chi2 = np.sum((y[elem_check]-ymodel[elem_check])**2/y[elem_check]) / (1.*len(x))

				#print j,lam2[j],snr,resid,chi2
				if (snr > 10) & (chi2 < 1.):
					now_peak.append(popt[1])  
					now_peak_err.append(perr[1])
					true_peakX.append(l+delta)
					true_peakL.append(lam2[j])
					#plt.plot(x,y,c='green')
					#yfit = popt[0]*np.exp(-0.5*(x-popt[1])*(x-popt[1])/(popt[2]**2)) + popt[3]
					#plt.plot(x,yfit,c='red')
					#plt.axvline(popt[1],ls='--',c='red')
					#plt.text(popt[1],popt[0],j)
					Nvalid += 1
				#else:
					#plt.plot(x,y,c='k')
					#yfit = popt[0]*np.exp(-0.5*(x-popt[1])*(x-popt[1])/(popt[2]**2)) + popt[3]
					#plt.plot(x,yfit,c='red',ls=':')
			
			except:
				#print 'Line '+np.str(j)+' not fitted'
				Nunfitted +=1

		#| Convert to numpy array
		now_peak 	= np.array(now_peak)
		true_peakL 	= np.array(true_peakL)
		true_peakX 	= np.array(true_peakX)
		
		
		#plt.errorbar(true_peakL, now_peak-true_peakX,yerr=now_peak_err,fmt ='o')
		#plt.show()
		#plt.close()
		#sys.exit()
	
		# ==================================
		#     Wavelength Solution
		# ==================================
	
		#| In principle, use all valid lines
		use = np.ones(len(now_peak))
	
		#| Iterate 5 times rejecting the worse lines (3-sigma clipping)
		
		for iter in range(5):
			skip_flag = iter
			xuse,yuse = now_peak[use == 1], true_peakL[use == 1]			

			if len(xuse) >= 7:
				coeff,cov = np.polyfit(xuse,yuse,3,cov=True)
			elif iter > 0:
				break
			else:
				Wavelength.append(np.zeros(2048)*np.nan)
				Nlines_used_order.append(np.nan)
				lam_residuals_order.append(np.nan)
				rv_residuals_order.append(np.nan)
				print "    --> WARNING: too few ThAr lines ("+str(len(xuse))+") found in order #"+str(i)+" ...skipping order"
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
		
		Nlines_used = len(np.where(use == 1)[0])
		Nlines_used_all += Nlines_used
	#	residuals = true_peakL[use == 1]-GLOBALutils.Cheby_eval(coeff,now_peak[use == 1],2048.) #p(now_peak[use == 1])
		residuals = true_peakL[use == 1]-p(now_peak[use == 1]) #p(now_peak[use == 1])
		rv_residuals = np.append(rv_residuals, residuals/ true_peakL[use == 1] * c.c)
		lam_residuals = np.append(lam_residuals, residuals*1.e3)  # in mA
	
		rv_residuals_order.append(sigmaG(res[good_lines] / true_peakL[good_lines]) * c.c.value)
		lam_residuals_order.append(sigmaG(res[good_lines] ))
		Nlines_used_order.append(Nlines_used)
	
		All_Pixel_Centers.append(now_peak[use == 1])
		All_Wavelengths.append(true_peakL[use == 1])
		All_Orders.append(np.zeros(len(now_peak[use == 1])) + i+60   )
		
		WC_coeffs.append(coeff)
		WC_covs.append(cov)
		WC_validity.append(str(round(np.min(true_peakL[use == 1]),1))+"-"+str(round(np.max(true_peakL[use == 1]),1)))
		
		Wavelength.append(p(np.arange(2048)))
			
		#print i+60, delta,Nlines_used,len(now_peak)-Nlines_used,sigmaG(res[good_lines] / true_peakL[good_lines]) * c.c,np.max(now_peak[use == 1])-np.min(now_peak[use == 1]),37000./c.c.value *np.max(true_peakL[use == 1]) 
	
	
	
		# ==================================
		#     Checking and debugging plots
		# ==================================

		#| Plot residuals
		plt.plot(now_peak[use == 1],residuals/ true_peakL[use == 1] * c.c/np.sqrt(2500.),'o',c='black')

		#| Plot pixel in X and wavelength in Y
		#xall = np.arange(2048)
		#plt.plot(now_peak[use == 1], true_peakL[use == 1] ,'o',c='red')
		#plt.plot(xall, GLOBALutils.Cheby_eval(coeff,xall,2048.) ,c='black')
	
		#| Plot pixel in X and wavelength in Y
		#plt.text(now_peak[use == 1],true_peakL[use == 1]-p(now_peak[use == 1]), np.repeat(i,len(np.where(use == 1)[0])))
		#plt.plot(now_peak[good_lines],true_peakL[good_lines]-p(now_peak[good_lines]),'o',c='r')
	plt.ylabel("RV (m/s)")
	plt.xlabel("X pixel")
	plt.grid(ls=':',c='gray',alpha=0.5)
	plt.savefig(cv.aux_dir+'/WC_'+plot_name+'.pdf')
	plt.close()
	
	

	# ==================================
	#     GLOBAL WAVELENGTH SOLUTION
	# ==================================

# 	p0    = np.zeros( npar_wsol )
# 	#print np.array(All_Pixel_Centers)
# 	#print np.shape(np.array(All_Pixel_Centers))
# 	p1, G_pix, G_ord, G_wav, II, rms_ms, G_res = GLOBALutils.Fit_Global_Wav_Solution(np.array(All_Pixel_Centers), np.array(All_Wavelengths),\
# 							     np.array(All_Orders), np.ones(len(All_Orders)), p0, Cheby=True,       \
# 							     order0=ro0, ntotal=n_useful, maxrms=200, Inv=True, minlines=200,  \
# 							     npix=2048.,nx=ncoef_x,nm=ncoef_m)


	#print p1
	#print sigmaG(rv_residuals)
	#print np.median(rv_residuals),np.mean(rv_residuals)
	#print sigmaG(lam_residuals),np.std(lam_residuals)

	# ==================================
	#     Log results
	# ==================================
	f = open(cv.aux_dir+'/log_'+cv.night,'a')
	f.write("===================\n")
	f.write("Wavelength Solution\n")
	f.write("===================\n")

	f.write("+ General results\n")
	f.write("  --> File = "+plot_name+".fits\n")
	f.write("  --> Number of lines used = "+np.str(Nlines_used_all)+"\n")
	f.write("  --> Wavelength residuals per line = "+np.str(round(sigmaG(lam_residuals),2))+" mA\n")
	f.write("  --> RV residuals per line = "+np.str(round(sigmaG(rv_residuals),2))+" m/s\n")
	f.write("  --> RV accuracy per line = "+np.str(round(np.mean(rv_residuals),2))+" m/s\n")
	f.write("  --> Estimated RV precision = "+np.str(round(sigmaG(rv_residuals/np.sqrt(Nlines_used_all)),2))+" m/s\n")
	f.write("  --> Estimated RV accuracy = "+np.str(round(np.mean(rv_residuals/np.sqrt(Nlines_used_all)),2))+" m/s\n")
	f.write("  --> Pix_shift from reference ThAr mask = "+np.str(round(np.nanmin(delta_refThAr), 2))+" - " +np.str(round(np.nanmax(delta_refThAr), 2))+' pix.\n')

	WCdict =  { 'rv_residuals_order':rv_residuals_order,
				'lam_residuals_order':lam_residuals_order,
				'Nlines_used_order':Nlines_used_order,
				'Nlines_total_order':Nlines_total_order,
				'All_Pixel_Centers':All_Pixel_Centers,
				'All_Wavelengths':All_Wavelengths,
				'All_Orders':All_Orders,
				'Order':Order,
				'Average_lam_residual':round(np.median(lam_residuals),2),
				'RV_accuracy_budget':round(np.mean(rv_residuals/np.sqrt(Nlines_used_all)),2),
				'RV_error_budget':round(sigmaG(rv_residuals/np.sqrt(Nlines_used_all)),2),
				'Nlines_total':Nlines_used_all,
				'WC_validity':WC_validity
				}

	
	x_arc = np.append(x_arc,np.atleast_3d(np.array(Wavelength).T).T,axis=0)	
	
	return np.array(WC_coeffs),np.array(WC_covs),WCdict,x_arc



def WavelengthCalibration(ArcList,arcs,arc_names,cv,xshift,yshift, type):
	
	S_frame = []
	for ii,Xarc in enumerate(ArcList):
		already_done = os.path.isfile(cv.aux_dir+'WC_'+arc_names[ii])
		filename, file_extension = os.path.splitext(arc_names[ii])
		if already_done == False:
			
			#| Compute wavelength solution
			
			coeff,cov,WCdict,SS_frame = FindWaveSolution(Xarc,xshift,yshift,cv, plot_name=filename)
			
			#| Save results			
			data = Table([WCdict['Order'],WCdict['Nlines_used_order'],WCdict['Nlines_total_order'],\
						  WCdict['lam_residuals_order'],WCdict['rv_residuals_order']],\
						  names=['# Order', 'Nl_used','Nl_all','wave_res (mA)', 'rv_res (m/s)'])
			
			ascii.write(data, cv.aux_dir+'/WC_'+filename+'.dat',format ='tab')
			np.savez(cv.aux_dir+'/WC_'+filename, coeff=coeff,cov=cov,WCdict=WCdict)
			
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
		
		#sys.exit()
	return S_frame
	
	
	
	
	
	
	
	
	









