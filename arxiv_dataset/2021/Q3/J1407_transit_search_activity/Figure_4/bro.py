# -*- coding: utf-8 -*-
#Christiaan Dik and Stan Barmentloo 2021
#Butcher Ringaling Offspring (BRO)

import numpy as np
from scipy.optimize import curve_fit
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
    
def short_correct(time, flux, eflux, frequencies = 1/np.linspace(2, 10, 3000), gap_size=20, min_chunk_size = 20, max_chunk_duration=500):
    #-----------------------------------------------------------------------------------
    short_corr_flux = np.copy(flux)
    short_corr_flux-= np.mean(short_corr_flux)
    
    max_periods = []
    all_powers = []
    
    chunk_beginnings = [0]
    chunk_ends = []
    cutoffs = []
    #-----------------------------------------------------------------------------------
    for j in range(len(time)-1):
        #Look for gaps larger than gap_size days to identify 'chunks' of continuous measurements
        if (time[j]-time[j-1] > gap_size) or ((j-1) in cutoffs):
            chunk_beginnings.append(j)
        if time[j+1]-time[j] > gap_size:
            chunk_ends.append(j)
        #If the length of the chunk is greater than max_chunk_duration, start a new chunk
        if len(cutoffs) == 0 and len(chunk_ends) == 0:
            if time[j]-time[0] >= max_chunk_duration:
                chunk_ends.append(j)
                cutoffs.append(j)
        elif len(cutoffs) == 0 and len(chunk_ends) != 0:
            if time[j]-time[chunk_beginnings[-1]] >= max_chunk_duration:
                chunk_ends.append(j)
                cutoffs.append(j)
        else:
            if time[j]-time[cutoffs[-1]] >= max_chunk_duration:
                chunk_ends.append(j)
                cutoffs.append(j)
    chunk_ends.append(len(time)-1)
    #-----------------------------------------------------------------------------------
    #Get the highest power period for each chunk, fold the chunk over this time and correct for the best fit periodic curve
    for i in range(len(chunk_beginnings)):
        if chunk_ends[i]-chunk_beginnings[i] > min_chunk_size:
            chunk_time = time[chunk_beginnings[i]:chunk_ends[i]]
            chunk_flux = short_corr_flux[chunk_beginnings[i]:chunk_ends[i]]
            chunk_eflux = eflux[chunk_beginnings[i]:chunk_ends[i]]

            power = LombScargle(chunk_time, chunk_flux-np.mean(chunk_flux), dy = chunk_eflux).power(frequencies)
            max_period = 1/frequencies[list(power).index(np.max(power))]
            all_powers.append(power)

            max_periods.append(max_period)
            
            time_folded = chunk_time%max_period
         
            try:
                means, errors, midpoints = binner(np.linspace(0, max_period, int(2*len(chunk_flux)**(1/3))), time_folded, chunk_flux, chunk_eflux)
                popt, pcov = curve_fit(ringaling, np.array(midpoints)*2*np.pi/(max_period), means, sigma = errors, absolute_sigma = True, p0 = [1,0,0, 1, 1], maxfev = 5000)
                phi = np.arange(200)*2*np.pi / 200
                short_corr_flux[chunk_beginnings[i]:chunk_ends[i]] -= ringaling(time_folded*2*np.pi/(max_period), *popt) #remove the trend
            except (RuntimeError, TypeError, ZeroDivisionError) as e:
                try: 
                    popt, pcov = curve_fit(ringaling, time_folded*2*np.pi/(max_period), chunk_flux, sigma = chunk_eflux, absolute_sigma = True, p0 = [1,0,0, 1, 1], maxfev = 5000)
                    phi = np.arange(200)*2*np.pi / 200
                    short_corr_flux[chunk_beginnings[i]:chunk_ends[i]] -= ringaling(time_folded*2*np.pi/(max_period), *popt) #remove the
                except:
                    max_periods[-1] = 42
    
    final_flux = short_corr_flux
    
    return final_flux, np.array(max_periods), frequencies, all_powers

#-----------------------------------------------------------------------------------
def ringaling(phi,*c):
    # c is a numpy array containing an odd number of coeffieicnts
    # so that c[0] + c[1]*np.sin(phi) + c[2]*np.cos(phi) + c[3]*np.sin(2*phi) + c[4]*np.cos(2*phi) + .....
    c = np.array(c)
    npairs = (c.size-1)/2
    result = 0
    for i in np.arange(npairs):
        a_sin = c[((i*2)+1).astype(int)]
        a_cos = c[((i*2)+2).astype(int)]
        result = result + a_sin*np.sin(phi*(i+1)) + a_cos*np.cos(phi*(i+1))
    return result+c[0]
#-----------------------------------------------------------------------------------
def binner(bins, time, flux, eflux):
    means, errors = [], []
    midpoints = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    for i in range(len(bins)-1):
        binned_mask = (time < bins[i+1])*(time > bins[i])
        means.append(np.average(flux[binned_mask], weights = 1/eflux[binned_mask]**2))
        errors.append(np.std(flux[binned_mask])/np.sqrt(len(flux[binned_mask])))
    return means, errors, midpoints