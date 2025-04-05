#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import bootstrap
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle


# In[ ]:


def alt_period_finder(time, flux, eflux, gap_size = 20, sample_size = 30, absolute_minimal_chunk_length =20, periods = np.linspace(3.1,3.3,500), points_per_cycle = 2):
    
    chunk_beginnings = [time[0]]
    chunk_ends = []
    failed_points = []

    for j in range(len(time)-1):
        #Look for gaps larger than gap_size days to identify 'chunks' of continuous measurements
        if (time[j]-time[j-1] > gap_size) and (len(chunk_ends) != 0):
            chunk_beginnings.append(time[j])
        if time[j+1]-time[j] > gap_size:
            chunk_ends.append(time[j])
        
    chunk_ends.append(time[-1])
    
    midpoints, short_periods, found_period_err= [], [], []
    
    for k in range(len(chunk_beginnings)):
        chunk_length = chunk_ends[k]-chunk_beginnings[k]
        if chunk_length > absolute_minimal_chunk_length:
            for l in range(np.max((int(chunk_length/sample_size), 1))):
                segment_mask = (chunk_beginnings[k]+l*sample_size < time) * (time < chunk_beginnings[k]+(l+1)*sample_size)
                
                segment_time = time[segment_mask]
                segment_flux = flux[segment_mask]
                segment_eflux = eflux[segment_mask]
                    
                if len(segment_time) > (float(sample_size)/3.2)*points_per_cycle: #Ensure sufficient sampling of this segment
                    frequencies = 1/np.linspace(2, 5, 3000)
                    power = LombScargle(segment_time, segment_flux-np.mean(segment_flux), dy = segment_eflux).power(frequencies)
                
                     
                    if 3.1 < 1/frequencies[np.argmax(power)] < 3.3: #The dominant signal is the 3.2 day period signal 
                        #Repeat the LS, now for a more zoomed in sample
                        frequencies, power = LombScargle(segment_time, segment_flux- np.mean(segment_flux)).autopower(minimum_frequency=1/(np.max(periods)), maximum_frequency=1/(np.min(periods)), samples_per_peak=10)
                       
                        found_period = float((1/(frequencies[np.argmax(power)])))
                        midpoint = np.median(segment_time)
                        
                        found_err = bootstrap.main(segment_time, segment_flux, segment_eflux, found_period)
                        if (found_err != 'Fail'):
                            found_period_err.append(found_err)                
                            midpoints.append(midpoint)
                            short_periods.append(found_period)

                        print("The table entry for this segment is: ", str(int(np.min(segment_time)))+ ' & ' +str(int(np.max(segment_time))) + ' & ' + str('{0:.3f}'.format(found_period)) +' & '+ str('{0:.3f}'.format(found_err)) +' \\\\')

    midpoints, short_periods, found_period_err = np.array(midpoints), np.array(short_periods), np.array(found_period_err)
    return midpoints, short_periods, found_period_err

