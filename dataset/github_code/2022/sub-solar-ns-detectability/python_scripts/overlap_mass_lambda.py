#Import relevant modules
import socket
import os
import sys
import htchirp
import csv
import time
import numpy as np
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.filter import match, sigmasq
from scipy.interpolate import griddata
from pycbc.waveform import get_td_waveform_from_fd
from pycbc.conversions import f_schwarzchild_isco, mass1_from_mchirp_eta, mass2_from_mchirp_eta, eta_from_q, lambda_tilde
from scipy.stats import loguniform
import matplotlib.pyplot as plt

#Renaming the execute scratch directory to point to .chirp_config file
os.environ['_CONDOR_SCRATCH_DIR']='/srv'
home_dir = os.environ['_CONDOR_SCRATCH_DIR']
chirp = htchirp.HTChirp()
cluster_process = os.environ['cluster_process_var']

f_low=15.0
delta_f=1/pow(2,5)



mchirp_lower=0.15
mchirp_upper=1.0

td1= loguniform.rvs(10, 8e6, size=10000)
td2= loguniform.rvs(10, 8e6, size=10000)
chirpmass=np.random.uniform(mchirp_lower,mchirp_upper,size=10000)
match_array=[]
eta=eta_from_q(1.0)
for i in range (0,10000):
    mass1=mass1_from_mchirp_eta(chirpmass[i],eta)
    mass2=mass2_from_mchirp_eta(chirpmass[i],eta)
    lmd_tilde=lambda_tilde(mass1,mass2,td1[i],td2[i])
    f_isco=f_schwarzchild_isco(mass1+mass2)    
    hp, hc = get_td_waveform_from_fd(approximant = "TaylorF2", 
                             mass1 = mass1, 
                             mass2 = mass2,  
                             f_lower=f_low, 
                             f_final=f_isco, 
                             delta_f=delta_f,
                             delta_t=1.0/(8192),
                             lambda1 = td1[i], 
                             lambda2 = td2[i])
    delta_f=hp.delta_f
    psd = aLIGOZeroDetHighPower(len(hp), delta_f, f_low)
    #Normalization
    hp_norm = sigmasq(hp, psd = psd, low_frequency_cutoff = f_low,high_frequency_cutoff=f_isco)
    #BBH waveform
    sp, sc = get_td_waveform_from_fd(approximant = "TaylorF2", 
                             mass1 = mass1, 
                             mass2 = mass2,  
                             f_lower=f_low, 
                             f_final=f_isco, 
                             delta_t=1.0/(8192),
                             delta_f=delta_f)
    #Normalization 
    sp_norm = sigmasq(sp, psd = psd, low_frequency_cutoff = f_low,high_frequency_cutoff=f_isco)
    #Resizing the waveforms such that they are the same size
    wavelength = max(len(hp),len(sp))
    hp.resize(wavelength)
    sp.resize(wavelength)
    #Calculating the match between injection (hp) and template (sp)
    m, k = match(hp, sp, psd = psd,v1_norm = hp_norm, v2_norm =sp_norm,low_frequency_cutoff = 
                 f_low)
    match_array.append(m)  
    output_data = str(chirpmass[i])+' '+str(lmd_tilde)+' '+str(m)
    for a in range(5):

            try:

                chirp.connect()
                chirp.write(("{}\n".format(output_data)).encode(),'./overlap_data/overlap_mass_lmd_{}.txt'.format(cluster_process),'cwa')
                chirp.disconnect

                break
        
            except socket.timeout:

                time.sleep(5.)