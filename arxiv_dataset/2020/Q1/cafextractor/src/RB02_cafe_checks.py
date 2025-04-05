import sys
import os

import scipy.signal
import scipy.optimize as opt
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
if os.path.isdir("/pcdisk/kool5/jlillo"): plt.switch_backend('agg')
import progressbar
from scipy import signal
import matplotlib.gridspec as gridspec # GRIDSPEC !

import CAFEutilities
import CAFEx_SetupFile as CS

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


def get_shift(new,ref):
	aa = ref
	bb = new #b[0].data
	corr = scipy.signal.fftconvolve(aa[::-1,::-1],bb, mode='same')
	ymax,xmax = np.unravel_index(corr.argmax(), corr.shape) #np.argmax(corr,axis=0)[0], np.argmax(corr,axis=1)[0]
	ampl = 5
	subcorr= np.array(corr[ymax-ampl:ymax+ampl,xmax-ampl:xmax+ampl])
	x = np.linspace(0, 10, 10)
	y = np.linspace(0, 10, 10)
	x, y = np.meshgrid(x, y)
	initial_guess = (np.max(subcorr),ampl,ampl,2.,2.,0.,0.)
	popt, pcov = opt.curve_fit(twoD_Gaussian, (x,y), subcorr.ravel(), p0 = initial_guess)
	err = np.sqrt(np.diag(pcov))
	exshift,eyshift = err[2],err[1]
	data_fitted = twoD_Gaussian((x, y), *popt)
	yshift = (ymax - 1024.) + (popt[1]-ampl)
	xshift = (xmax - 1024.) + (popt[2]-ampl)
	intensity = popt[0]
	#print xmax,ymax,xshift,yshift,popt[2],popt[1]
	return xshift,yshift,exshift,eyshift, intensity


def cafe_shift(cv,arcs):

	#| Select the optimal zero-order solution according to observing date
	fileref = '../ReferenceFrames/ReferenceCalibs.lis'
	fref = np.genfromtxt(fileref,dtype=None,names=True,encoding='ascii')
	jdnight = CAFEutilities.jdnight(cv.night)
	id = fref['ID'][np.max(np.where((fref['Datestart'] < jdnight) & (fref['Dateend'] > jdnight))[0])]

	CS.var.set_Orientation(CAFEutilities.jdnight(cv.night))

	# Orientation of the CCD (according to grating orientation up/down). Right ==> >2018 
	if CS.var.orientation == 'right':
		ref_data = np.flip(fits.getdata(cv.arc_ref),1)
	elif CS.var.orientation == 'left':
		ref_data = fits.getdata(cv.arc_ref)
	else:
		print "   ---> ERROR: I found no orientation value in the ReferenceCalib.lis "
		print "				  file for thi date. Please check that your observations fall "
		print " 			  into one of the cafe CCD windows defined in that file. "
		sys.exit()	

	#new = fits.open(arcs['files'][0])
	#new_data = new[0].data
	dc = arcs['dc']
	new_data = dc[0,:,:]
	
	# Autocorrelation of the reference arc:
	x0,y0, e_x0,e_y0, int0 = get_shift(ref_data,ref_data)
	
	# Shift of first arc of the night:
	xref,yref, e_xref,e_yref, intref = get_shift(new_data,ref_data)

	# Initialize progress bar
	bar = progressbar.ProgressBar(maxval=len(arcs['files']), \
	      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()

	shifts = np.zeros((len(arcs['files']),2))
	eshifts = np.zeros((len(arcs['files']),2))
	intensity = np.zeros(len(arcs['files']))

	# Shift of each arc against ReferenceARC
	for i,arc in enumerate(dc):
		frame_data = dc[i,:,:] 
		xshift,yshift, e_xshift,e_yshift, int = get_shift(frame_data,new_data)
		shifts[i,:] = [xshift+xref-x0,yshift+yref-y0]
		eshifts[i,:] = [np.sqrt(e_xshift**2+e_xref**2+e_x0**2),np.sqrt(e_yshift**2+e_yref*2-e_y0**2)]
		intensity[i] = int
		bar.update(i+1)
	bar.finish()
	intensity = intensity/np.mean(intensity)
	return shifts,intensity


def cafe_temperature(cv,bias,flats,arcs,sci):
	Tcoll 	= []
	Tbenc	= []
	Tgrat	= []
	Troom	= []
	jd		= []
	for i,fr in enumerate(arcs['files']):
		frame = fits.open(fr)
		try:
			Tcoll.append(frame[0].header['TEMP1'])
			Tbenc.append(frame[0].header['TEMP2'])
			Tgrat.append(frame[0].header['TEMP3'])
			Troom.append(frame[0].header['TEMP4'])
			jd.append(CAFEutilities.get_jd(frame[0].header))
		except:
			0
	for i,fr in enumerate(bias['files']):
		frame = fits.open(fr)
		try:
			Tcoll.append(frame[0].header['TEMP1'])
			Tbenc.append(frame[0].header['TEMP2'])
			Tgrat.append(frame[0].header['TEMP3'])
			Troom.append(frame[0].header['TEMP4'])
			jd.append(CAFEutilities.get_jd(frame[0].header))
		except:
			0
	for i,fr in enumerate(flats['files']):
		frame = fits.open(fr)
		try:
			Tcoll.append(frame[0].header['TEMP1'])
			Tbenc.append(frame[0].header['TEMP2'])
			Tgrat.append(frame[0].header['TEMP3'])
			Troom.append(frame[0].header['TEMP4'])
			jd.append(CAFEutilities.get_jd(frame[0].header))
		except:
			0
	for i,fr in enumerate(sci['files']):
		frame = fits.open(fr)
		try:
			Tcoll.append(frame[0].header['TEMP1'])
			Tbenc.append(frame[0].header['TEMP2'])
			Tgrat.append(frame[0].header['TEMP3'])
			Troom.append(frame[0].header['TEMP4'])
			jd.append(CAFEutilities.get_jd(frame[0].header))
		except:
			0

	jd = np.array(jd)
	Tcoll 	= np.array(Tcoll)
	Tbenc	= np.array(Tbenc)
	Tgrat	= np.array(Tgrat)
	Troom	= np.array(Troom)
	srt = np.argsort(jd)
	
	jd 		= jd[srt].astype(np.float)
	Tcoll 	= Tcoll[srt].astype(np.float)
	Tbenc	= Tbenc[srt].astype(np.float)
	Tgrat	= Tgrat[srt].astype(np.float)
	Troom	= Troom[srt].astype(np.float)
	

	if 1:
		fig = plt.figure(figsize=(12,10))
		gs = gridspec.GridSpec(4,1, height_ratios=[1.,1.,1.,1], width_ratios=[1])
		gs.update(left=0.1, right=0.95, bottom=0.08, top=0.93, wspace=0.08, hspace=0.12)
		
		ax0 = plt.subplot(gs[0,:]) 
		plt.plot(jd,Tcoll-np.nanmean(Tcoll),'o',label='Collimator',c='indigo')
		plt.legend()
		plt.grid(ls=':',c='gray',alpha=0.5)
		
		ax1 = plt.subplot(gs[1,:]) 
		plt.plot(jd,Tbenc-np.nanmean(Tbenc),'o',label='Telescope Dome',c='darkmagenta')
		plt.legend()
		plt.grid(ls=':',c='gray',alpha=0.5)
		
		ax2 = plt.subplot(gs[2,:]) 
		plt.plot(jd,Tgrat-np.nanmean(Tgrat),'o',label='Grating',c='orchid')
		plt.legend()
		plt.grid(ls=':',c='gray',alpha=0.5)

		ax3 = plt.subplot(gs[3,:]) 
		plt.plot(jd,Troom-np.nanmean(Troom),'o',label='CAFE room',c='hotpink')
		plt.legend()
		plt.grid(ls=':',c='gray',alpha=0.5)

		plt.savefig(cv.aux_dir+'cafe_temperatures_'+cv.night+'.pdf',bbox_inches='tight')
		plt.close()
	
	return Tcoll, Tbenc, Tgrat, Troom

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def cafe_preasures(cv,bias,flats,arcs,sci):
	Pres1	= []
	Pres2	= []
	jd		= []
	for i,fr in enumerate(arcs['files']):
		frame = fits.open(fr)
		try:
			Pres1.append(np.float(frame[0].header['PRESS1']))
			Pres2.append(np.float(frame[0].header['PRESS2']))
			jd.append(CAFEutilities.get_jd(frame[0].header))
		except:
			0
	for i,fr in enumerate(bias['files']):
		frame = fits.open(fr)
		try:
			Pres1.append(np.float(frame[0].header['PRESS1']))
			Pres2.append(np.float(frame[0].header['PRESS2']))
			jd.append(CAFEutilities.get_jd(frame[0].header))
		except:
			0
	for i,fr in enumerate(flats['files']):
		frame = fits.open(fr)
		try:
			Pres1.append(np.float(frame[0].header['PRESS1']))
			Pres2.append(np.float(frame[0].header['PRESS2']))
			jd.append(CAFEutilities.get_jd(frame[0].header))
		except:
			0
	for i,fr in enumerate(sci['files']):
		frame = fits.open(fr)
		try:
			Pres1.append(np.float(frame[0].header['PRESS1']))
			Pres2.append(np.float(frame[0].header['PRESS2']))
			jd.append(CAFEutilities.get_jd(frame[0].header))
		except:
			0

	jd = np.array(jd)
	Pres1	= np.array(Pres1)
	Pres2	= np.array(Pres2)
	srt = np.argsort(jd)
		
	jd 		= jd[srt].astype(np.float)
	Pres1	= Pres1[srt].astype(np.float)
	Pres2	= Pres2[srt].astype(np.float)

	if 1:
		fig = plt.figure(figsize=(12,10))
		gs = gridspec.GridSpec(2,1, height_ratios=[1.,1.], width_ratios=[1])
		gs.update(left=0.1, right=0.95, bottom=0.08, top=0.93, wspace=0.08, hspace=0.12)
		
		ax4 = plt.subplot(gs[0,:]) 
		plt.plot(jd,Pres1,'o',label='Absolute pressure [mbar]',c='dodgerblue')
		plt.legend()
		plt.grid(ls=':',c='gray',alpha=0.5)

		ax5 = plt.subplot(gs[1,:]) 
		plt.plot(jd,Pres2,'o',label='Diff. press. Cabinet-Room [mbar]',c='turquoise')
		plt.legend()
		plt.grid(ls=':',c='gray',alpha=0.5)

		plt.savefig(cv.aux_dir+'cafe_preasures_'+cv.night+'.pdf',bbox_inches='tight')
		plt.close()
		np.savez(cv.aux_dir+'cafe_preasures_'+cv.night,jd=jd,pres1=Pres1)
	return Pres1, Pres2



	

