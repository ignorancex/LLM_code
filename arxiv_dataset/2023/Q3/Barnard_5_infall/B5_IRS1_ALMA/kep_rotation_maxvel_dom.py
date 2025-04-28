#!/usr/bin/python
# Filename: posvel.py

import numpy as np
import math
#import matplotlib; matplotlib.use("Agg")

import matplotlib.pylab as plt

import astropy.io.fits
#import pylab
import aplpy
#from numpy import ma
#from matplotlib.pyplot import step, legend, xlim, ylim, show
#import os
#import sys
from lmfit import minimize, Parameters
from lmfit.printfuncs import *

#from astropy.io import fits
from astropy import wcs
#from matplotlib.backends.backend_pdf import PdfPages
# import radiomodule as rmod does not exist

def coord_array(name_file,map_maker):
	pv_map_hdu=astropy.io.fits.open(name_file)
	w=wcs.WCS(pv_map_hdu[0].header)
	dim1=pv_map_hdu[0].header['naxis1'] #offset
	dim2=pv_map_hdu[0].header['naxis2'] #vel
	dim2_arr=np.zeros(dim2)
	dim1_arr=np.zeros(dim1)

	if map_maker=='casa':
		for el in range(len(dim2_arr)):
			dim2_arr[el]=w.wcs_pix2world(0,el,0,0)[1]

		for el in range(len(dim1_arr)):
			dim1_arr[el]=w.wcs_pix2world(el,0,0,0)[0]
	if map_maker=='me':
		for el in range(len(dim2_arr)):
			dim2_arr[el]=w.wcs_pix2world(0,el,0)[1]
			
		for el in range(len(dim1_arr)):
			dim1_arr[el]=w.wcs_pix2world(el,0,0)[0]

	print ('dim1', dim1, len(dim1_arr))
	print ('dim2', dim2, len(dim2_arr))

	return [dim1_arr,dim2_arr]

def residuals_gauss(param,offsets,int_profile,rms):
	line=param['amp'].value*np.exp(-(offsets-param['off_peak'].value)**2/(2*param['sigma_off'].value**2))
	subs1=np.array((line-int_profile)/rms,dtype=float)
	return np.array(subs1,dtype=float)

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def offpeak(name_pv,rms_pv,map_maker,vel_range,fit_line,vsys):
	pv_map_hdu=astropy.io.fits.open(name_pv)
	pv_map_data=pv_map_hdu[0].data[0,:,:] #(stokes, vel m/s,off)
	[off_array,vel_array]=coord_array(name_pv,map_maker)
	if np.isnan(vel_range[0]): vel_i=0
	else: vel_i=find_nearest(vel_array,vel_range[0])
	if np.isnan(vel_range[1]): vel_f=len(off_array)	
	else: vel_f=find_nearest(vel_array,vel_range[1])+1

	peaks=list()
	vel=list()
	for ivel in xrange(vel_i,vel_f,1):
		int_profile=pv_map_data[ivel,:]
		off_peak0=find_nearest(int_profile,np.max(int_profile))
		params=Parameters()
		params.add('amp',value=0.01,vary=True)
		params.add('off_peak',value=off_array[off_peak0],vary=True)#,min=-0.09,max=0.15)
		if fit_line=='grad':
			params.add('sigma_off',value=0.1,vary=True)
		if fit_line=='rotslope':
			params.add('sigma_off',value=1,vary=True)


		#print 'fitting gauss'
		# plt.step(off_array,int_profile)
		# plt.show()
		results=minimize(residuals_gauss,params,args=(off_array,int_profile,rms_pv),method='leastsq')
		if params['amp'].value>=2.0*rms_pv:
			#if params['off_peak'].value>=0:
			peaks.append(params['off_peak'].value)
			vel.append(vel_array[ivel])

	peaks=np.array(peaks)
	vel=np.array(vel)
	if fit_line=='none': return [peaks,vel]
	if fit_line=='grad':# and grad==True:
		params2=Parameters()
		params2.add('s',value=1e4,vary=True)
		vsys0=(vel_range[0]+vel_range[1])/2.
		params2.add('vsys',value=vsys0,vary=True)
		results=minimize(residuals_line,params2,args=(peaks,vel),method='leastsq')
		return [peaks,vel,params2['s'].value,params2['vsys'].value,params2['vsys'].stderr,params2['s'].stderr]
	if fit_line=='rotslope':
		params2=Parameters()
		params2.add('s',value=-0.5,vary=True)
		vsys0=1000.#(vel_range[0]+vel_range[1])/2.
		params2.add('vsys',value=vsys0,vary=True)
		results=minimize(residuals_powerlaw,params2,args=(np.abs(peaks),np.abs(vel-vsys)*1e-3),method='leastsq')
		return [peaks,vel,params2['s'].value,params2['vsys'].value,params2['vsys'].stderr,params2['s'].stderr]

def residuals_line(param,peaks,vel):
	subs1=np.array((param['s'].value*peaks+param['vsys'].value-vel),dtype=float)
	return np.array(subs1,dtype=float)

def residuals_powerlaw(param,peaks,vel):
	subs1=np.array((param['vsys'].value*peaks**param['s'].value-vel),dtype=float)
	return np.array(subs1,dtype=float)


def maxvel(name_pv,rms_pv,map_maker,offset_range):
	pv_map_hdu=astropy.io.fits.open(name_pv)
	if map_maker=='casa':
		pv_map_data=pv_map_hdu[0].data[0,:,:] #(stokes, offset arcsecond,vel m/s) #!!!!!!!!!!!!!!!!!!
	if map_maker=='me':
		pv_map_data=pv_map_hdu[0].data #(stokes, offset arcsecond,vel m/s) #!!!!!!!!!!!!!!!!!!


	[off_array,vel_array]=coord_array(name_pv,map_maker)
	if np.isnan(offset_range[0]):
		off_i=0
	else:
		off_i=find_nearest(off_array,offset_range[0])
	if np.isnan(offset_range[1]): 
		off_f=len(off_array)
	else: 
		off_f=find_nearest(off_array,offset_range[1])+1

	maxvel_array=list()
	maxoff_array=list()

	off=off_i

	while off <off_f:#xrange(len(off_array)):
		if off_array[off]<=0:
			vel_i=0
			c=1
		else:
			vel_i=-1
			c=-1

		while np.abs(vel_i)<len(vel_array):
	# 		print vel_i, 'vel_i'#
	#		print off, 'off'#
	#		print pv_map_data[vel_i,off], rms_pv
			if pv_map_data[vel_i,off]>=rms_pv:
				maxvel_array.append(vel_array[vel_i])
				maxoff_array.append(off_array[off])
				break

			vel_i=vel_i+c

		off=off+1

	maxoff_array=np.array(maxoff_array)
	maxvel_array=np.array(maxvel_array)
	return [maxvel_array,maxoff_array]

def residuals_brokenpowerlaw(param,maxvel_array,maxoff_array):
	maxoff_array=np.abs(maxoff_array)
	maxvel_array=np.abs(maxvel_array)
	minim_res=list()
	for i in range(len(maxoff_array)):
		if maxoff_array[i]<param['r0'].value:
			minim_res.append((maxvel_array[i]-param['constant'].value*maxoff_array[i]**-0.5)/200.)
		else:
			minim_res.append((maxvel_array[i]-param['constant'].value*(param['r0'].value**(-0.5-param['slope2'].value))*maxoff_array[i]**param['slope2'].value)/200.)

	return np.array(minim_res)

def brokenpowerlaw(off_array,r0,slope2,constant):
	broken_function=list()
	for i in range(len(off_array)):
		if off_array[i]<r0:
			broken_function.append(constant*off_array[i]**-0.5)
		else:
			broken_function.append(constant*r0**(-0.5-slope2)*off_array[i]**slope2)

	return np.array(broken_function)


def fit_brokenpowerlaw(maxvel_array,maxoff_array):
	params=Parameters()
	params.add('constant',value=1.,vary=True)
	params.add('r0',value=0.35,vary=True,min=0.2,max=0.75)
	params.add('slope2',value=-1,vary=True,max=-0.1)
	results=minimize(residuals_brokenpowerlaw,params,args=(maxvel_array,maxoff_array),method='leastsq')
	print (results.success)
	print ('chi: '+str(results.chisqr)+ '  red chi: '+str(results.redchi))
	print(results.lmdif_message)
	return [params['r0'].value,params['r0'].stderr,params['slope2'].value,params['slope2'].stderr, params['constant'].value,params['constant'].stderr]


def residuals(param,maxvel_array,maxoff_array,dist):
	maxoff_array=np.abs(maxoff_array)
	sign=np.sign(maxvel_array-param['vsys'].value)
	sign=sign/np.abs(sign)
	mini=maxvel_array-(param['vsys'].value+sign*np.sqrt(6.67e-11*param['mass'].value*2e30/(maxoff_array*dist*1.5e11)))
	return np.array(mini,dtype=float)

def keprot(offsets,mass,dist):
	offsets=np.abs(offsets)
	return np.sqrt(6.67e-11*mass*2e30/(offsets*dist*1.5e11))

def keprot_ring(offsets, mass,radius,dist):

	offsets=np.abs(offsets)
	offsets[offsets==0.]=np.nan
	v=np.sqrt(6.67e-11*mass*2e30/(radius*dist*1.5e11))*np.sin(offsets/radius)
	v[v==np.nan]=0.
	return v



def kepfit_maxvel(maxvel_array,maxoff_array,vsys,mass,dist,vsysmin,vsysmax):

	params=Parameters()
	params.add('vsys',value=vsys[0],vary=vsys[1],min=vsysmin,max=vsysmax)
	params.add('mass',value=mass[0],vary=mass[1])
	results=minimize(residuals,params,args=(maxvel_array,maxoff_array,dist),method='leastsq')
	print (results.success)
	print ('chi: '+str(results.chisqr)+ '  red chi: '+str(results.redchi))

	return [params['vsys'].value,params['vsys'].stderr,params['mass'].value,params['mass'].stderr]

def kepfitbin_maxvel(maxvel_array,maxoff_array,mu,mass,dist,mu_min,mu_max,v1,v2,sep):
	params=Parameters()
	params.add('mu',value=mu[0],vary=mu[1],min=mu_min,max=mu_max)
	params.add('mass',value=mass[0],vary=mass[1])
	results=minimize(residuals_binary,params,args=(maxvel_array,maxoff_array,dist,sep,v1,v2),method='leastsq')
	print (results.success)
	print ('chi: '+str(results.chisqr)+ '  red chi: '+str(results.redchi))
	mu=results.params['mu'].value
	vsys=(v1+v2*mu)/(1+mu)
	vsyserr=0.0
	return [vsys,vsyserr,results.params['mass'].value,results.params['mass'].stderr,results.params['mu'].value,results.params['mu'].stderr]

def residuals_binary(param,maxvel_array,maxoff_array,dist,sep,v1,v2):
	maxoff_array=np.abs(maxoff_array)
	mu=param['mu'].value
	vsys=(v1+v2*mu)/(1+mu)
	sign=maxvel_array-vsys
	sign=sign/np.abs(sign)
	sign2=(sign+1)/2.
	r1=(1./(1+mu))*sep
	r2=(mu/(1+mu))*sep
	maxoff_array=maxoff_array+(r2-r1)*sign2+r1
	mini=maxvel_array-(vsys+sign*np.sqrt(6.67e-11*param['mass'].value*2e30/(maxoff_array*dist*1.5e11)))
	return np.array(mini,dtype=float)



#def gradient_fit



