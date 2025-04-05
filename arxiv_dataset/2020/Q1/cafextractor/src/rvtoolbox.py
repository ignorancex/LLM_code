import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as C
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import quad
from astropy.io import fits
from astroML.stats import sigmaG


cc 		= C.c.value*1.e-3	# [km/s]

# ======================================
#		Useful functions
# ======================================
def gaussfit(x, a0, a1, a2, a3, a4, a5):
	z = (x-a1) / a2
	y = a0 * np.exp(-z**2 / 2) + a3 + a4 * x + a5 * x**2
	return y

def rotprofile(x, a0, a1, a2, a3, a4, a5, mu):
	a 	= a0
	x0 	= a1
	xl	= a2	
	y = 1. -2.*a*(1.-mu) * np.sqrt(1.-((x-x0)/xl)**2) + \
		   (0.5*np.pi*mu*(1.-((x-x0)/xl)**2))/(np.pi*xl*(1.-mu/3.))
	y[~np.isfinite(y)] = 1.0
	y += a3 + a4 * x + a5 * x**2
	return y


def gauss(x, a):
	z = (x-a[1]) / a[2]
	y = a[0] * np.exp(-z**2 / 2) + a[3] #+ a4 * x + a5 * x**2
	return y

def gaussfit_Moon(x, a0, a1, a2, a3, a4, a5, a6):
	ztarg = (x-a1) / a2
	ytarg = a0 * np.exp(-ztarg**2 / 2) + a3 #+ a4 * x + a5 * x**2
	
	zMoon = (x-a5) / a6
	yMoon = a4 * np.exp(-zMoon**2 / 2) 
	
	return ytarg + yMoon

def closest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

# ======================================
#		Velocity array
# ======================================
def create_dvel(inst,w,RVguess=0.0,RVampl=100.,verbose=False):
	
	# ===== CARMENES visible arm
	if inst == 'CAFE':
		RR = []
		for oo in range(w.shape[0]):
			RR.append(w[oo,1:-1]/(w[oo,1:-1]-w[oo,0:-2]))
		RR = np.array(RR)
		R = np.nanmean(RR)/2.2
		Vscale = cc/np.nanmean(R)
		if verbose:
			print "Measured spectral resolution, R = ", np.str(np.round(R))
			print "Measured spectral resolution, Vscale = ", np.str(cc/R),' km/s'

	else:
		RR = w[1:-1]/(w[1:-1]-w[0:-2])
		R = np.mean(RR)/3.
		Vscale = cc/R
		if verbose:
			print "Measured spectral resolution, R = ", np.str(np.round(R))
			print "Measured spectral resolution, Vscale = ", np.str(cc/R),' km/s'
		
	# ===== Velocity array of the CCF
	vmin 	= RVguess-RVampl
	vmax 	= RVguess+RVampl
	if ~np.isfinite(Vscale): Vscale = 4.54 # km/s
	vwidth  = Vscale/2.2/2. 			# km/s  
	vstep 	= vwidth/2. 
	dlogLambda 	= vwidth/cc 
	
	if ((~np.isfinite(RVguess)) | (~np.isfinite(RVampl))):
		vmin, vmax = -100, 100

	dvel = np.linspace(vmin,vmax,(vmax-vmin)/vstep)

	return dvel,vwidth


# ======================================
#		CCF fitting
# ======================================

def fit_CCF(dvel,CCF,eCCF, RVguess=-99.9, AMPLguess=10.0, with_Moon=False, rot_profile=False):

	if RVguess == -99.9:
		guessRV = True 
	else:
		guessRV = False 
		
	# ===== No Moon CCF included in the fit
	if with_Moon == False:
		try:
			if guessRV == False:
				#RVguess = (np.max(dvel)+np.min(dvel))/2.
				#mybounds = ([-np.inf, RVguess-5., 0.0, -np.inf], [0, RVguess+5., 10., np.inf])
				if rot_profile == True:
					mybounds = ([0.0, RVguess-10., 0.0, -np.inf, -np.inf, -np.inf, 0.0], [np.inf, RVguess+10., 100., np.inf, np.inf, np.inf, 1.0])
					myp0 = [np.median(CCF)-np.min(CCF), RVguess, AMPLguess, np.median(CCF), 0.0, 0.0, 0.5]					
				else:
					mybounds = ([-np.inf, RVguess-5., 0.0, -np.inf, -np.inf, -np.inf], [0, RVguess+5., 100., np.inf, np.inf, np.inf])
					myp0 = [np.min(CCF)-np.median(CCF), RVguess, AMPLguess, np.median(CCF), 0.0, 0.0]
			else:
				if rot_profile == True:
					mybounds = ([0.0, np.min(dvel), 0.0, -np.inf, -np.inf, -np.inf, 0.0], [1., np.max(dvel), 100., np.inf, np.inf, np.inf, 1.0])
					myp0 = [np.median(CCF)-np.min(CCF), dvel[np.argmin(CCF)], AMPLguess, np.median(CCF), 0.0, 0.0, 0.5]
				else:
					mybounds = ([-np.inf, np.min(dvel), 0.0, -np.inf, -np.inf, -np.inf], [0, np.max(dvel), 100., np.inf, np.inf, np.inf])
					myp0 = [np.min(CCF)-np.median(CCF), dvel[np.argmin(CCF)], AMPLguess, np.median(CCF), 0.0, 0.0]
		
			if rot_profile == True:
				popt, pcov = curve_fit(rotprofile, dvel, CCF, p0 = myp0, bounds = mybounds) #, sigma=eCCF
			else:
				popt, pcov = curve_fit(gaussfit, dvel, CCF, p0 = myp0, bounds = mybounds) #, sigma=eCCF					
			
			perr = np.sqrt(np.diag(pcov))

		except:
			popt = np.zeros(6)*np.nan
			perr = np.zeros(6)*np.nan

	# ===== Including Moon CCF included in the fit
	if with_Moon == True:
		try:
			if guessRV == False:
				RVguess = (np.max(dvel)+np.min(dvel))/2.
				mybounds = ([-np.inf, RVguess-5., 0.0 , -np.inf, -np.inf, -5., 0.0 ],
							[0      , RVguess+5., 10.,  np.inf, 0.0    , 5.0, 5.])
			else:
				mybounds = ([-np.inf, -2., 0.0 , -np.inf, -np.inf, -5., 0.0 ], 
							[0      ,  2., 10.,  np.inf, 0.0    , 5.0, 5.])
		
			ampl_Moon = np.interp(0.0, dvel, CCF)-np.median(CCF)
			
			myp0 = [np.min(CCF)-np.median(CCF), RVguess, 5., np.median(CCF), # target
					-100., 0.0, 2.] # Moon params
			
			#for tt,p in enumerate(myp0): print p,mybounds[0][tt],mybounds[1][tt]
			
			#sys.exit()
			
			#popt, pcov = curve_fit(gaussfit_Moon, dvel, CCF, sigma=eCCF, absolute_sigma=True,
			#					   p0 = myp0, bounds = mybounds)
			popt, pcov = curve_fit(gaussfit_Moon, dvel, CCF, p0 = myp0, bounds = mybounds)
			perr = np.sqrt(np.diag(pcov))

		except:
			popt = np.zeros(7)*np.nan
			perr = np.zeros(7)*np.nan
	
	return popt, perr
	

# ======================================
#		CCF determination
# ======================================

def CCF(w,f,ef,dvel,vwidth, wmask, fmask, CRAYS=True):

	# Mask cosmic rays
# 	if CRAYS == True:
# 		CRs = np.where(f > np.nanmedian(f)+20.*sigmaG(f[~np.isnan(f)]))[0]
# 		f[CRs] = np.nan	
	
	#print np.nanmedian(f),sigmaG(f[~np.isnan(f)]),np.nanmedian(f)+20.*sigmaG(f[~np.isnan(f)])
	
	# Mask Tellurics
	#tellmask = interp(lam2wave(mask[:,0]), mask[:,1])

	# Remove nans
	noNans = (~np.isnan(f)) & (~np.isnan(ef))
	w   = w[noNans]	
	ef  = ef[noNans]
	f   = f[noNans]
	#ef 	/= np.nanmedian(f)
	#f 	/= np.nanmedian(f)
		
	vmin 	= np.min(dvel)
	vmax 	= np.max(dvel)
	dlogLambda 	= vwidth/cc
	RVguess = (vmax+vmin)/2.

	# Mask lines in range
	inrange = np.where((wmask*(1.+vmin/cc) > np.min(w)+2. ) & (wmask*(1.+vmax/cc) < np.max(w)-2. ))[0]
	if len(inrange) == 0:
		CCF  = np.zeros(len(dvel))
		eCCF  = np.zeros(len(dvel))
		return CCF,eCCF
		
	wmaskR = wmask[inrange]
	fmaskR = fmask[inrange]
	dlogLambda = dlogLambda[inrange]

	CCF  = np.zeros(len(dvel))
	eCCF  = np.zeros(len(dvel))

	# Cumulative spectrum and spline interpolation
# 	plt.close()
# 	plt.plot(np.arange(len(w)-1), w[1:]-w[0:-1])
# 	plt.show()
# 	sys.exit()
	w = np.log(w)	
	fcum = np.cumsum(f)
	#winterp = interp1d()
	#pixs = np.mean(w[1:]-w[0:-1])
	#fspl = interp1d(w+pixs/2., fcum, kind='linear')
	pixs = w[1:]-w[0:-1]
	#print np.shape(pixs), np.shape(w), np.shape(fcum)
	fspl = interp1d(w[:-1]+pixs/2., fcum[:-1], kind='linear')
	
	# Loop in velocity array
	for vv,v in enumerate(dvel):
		wnew = wmaskR*(1.0+v/cc)
		wnew = np.log(wnew)
		F1 		= fspl(wnew+dlogLambda/2.)
		F0 		= fspl(wnew-dlogLambda/2.)
		
		# Error of CCF
		eF = []
		for hh,line in enumerate(wnew):
			sorted = np.argsort(np.abs(w-(line+dlogLambda[hh]/2.)))
			w1 = w[np.nanmin([sorted[0],sorted[1]])]
			w2 = w[np.nanmax([sorted[0],sorted[1]])]
			pixsize = w2-w1
			f1 = (w2-(line+dlogLambda[hh]/2.))/pixsize
			eflux1 = ef[np.nanmin([sorted[0],sorted[1]])] 

			sorted = np.argsort(np.abs(w-(line-dlogLambda[hh]/2.)))
			w1 = w[np.nanmin([sorted[0],sorted[1]])]
			w2 = w[np.nanmax([sorted[0],sorted[1]])]
			pixsize = w2-w1
			f0 = ((line-dlogLambda[hh]/2.)-w1)/pixsize
			eflux0 = ef[np.nanmin([sorted[0],sorted[1]])] 
			
			eF.append( np.sqrt((f0*eflux0)**2+(f1*eflux1)**2) ) 
			
		eF = np.array(eF) #* fmaskR #* np.abs(fmaskR)
		
		# CCF stack of all lines
		Ftotal  = np.nansum((F1-F0))# * fmaskR)  # *np.abs(fmaskR))
		CCF[vv]  = Ftotal
		eCCF[vv] = np.sqrt(np.nansum(eF**2))
		


	# CCF fitting
# 	try:
# 		mybounds = ([-np.inf, RVguess-5., 0.0, 0.0], [0, RVguess+5., 100., np.inf])
# 		popt, pcov = curve_fit(gaussfit, dvel, CCF, sigma=eCCF, absolute_sigma=True,
# 							   p0=[np.min(CCF)-np.median(CCF), dvel[np.argmin(CCF)], 10., np.median(CCF)],bounds=mybounds)
# 		perr = np.sqrt(np.diag(pcov))
# 		RV  = popt[1]#+vwidth/2.
# 		eRV = perr[1]
# 	except:
# 		RV  = np.nan
# 		eRV = np.nan
# 		popt = np.zeros(4)*np.nan
# 		perr = np.zeros(4)*np.nan


	if 0:
		plt.errorbar(dvel,CCF,yerr=eCCF)
		#plt.plot(dvel,gaussfit(dvel,*popt))
		plt.show()
		plt.close()



	return CCF,eCCF#,RV,eRV, dvel, popt, perr


 # ===================================================
 	#CCFl  = np.zeros((len(wmask),len(dvel)))
	#eCCFl  = np.zeros((len(wmask),len(dvel)))

# 	for ll in range(len(wmask)):
# 		for vv,v in enumerate(dvel):
# 			wnew = wmask[ll]*(1.0+v/cc)
# 			wnew = np.log(wnew)
# 			F1 		= fspl(wnew+dlogLambda/2.)
# 			F0 		= fspl(wnew-dlogLambda/2.)
# 			Ftotal  = F1-F0
# 			CCFl[ll,vv]  = Ftotal * np.abs(fmask[ll])
# 			eCCFl[ll,vv] = np.sqrt(np.sum((np.random.randn(2)*white_noise)**2))* np.abs(fmask[ll])

