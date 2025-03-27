#!/usr/bin/env python

#Import relevant modules
import socket
import argparse
import os
import sys
import htchirp
import csv
import time
import numpy as np
import pycbc
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.filter import match, sigmasq
from pycbc.waveform import get_td_waveform_from_fd
from pycbc.conversions import mass1_from_mchirp_eta,mass2_from_mchirp_eta, f_schwarzchild_isco
from scipy.interpolate import interp1d

import numpy.random
numpy.random.seed(None)

parser=argparse.ArgumentParser()
parser.add_argument("--choice", type=str, choices=["APR", "SLY4", "BSK21"], required=True)
#The input file should contain 2 columns, with mass,lambda
parser.add_argument("--input-file", type=str, required=True)
opts = parser.parse_args()

#Renaming the execute scratch directory to point to .chirp_config file
os.environ['_CONDOR_SCRATCH_DIR']='/srv'
home_dir = os.environ['_CONDOR_SCRATCH_DIR']
chirp = htchirp.HTChirp()
cluster_process = os.environ['cluster_process_var']

lambda_data = np.loadtxt(opts.input_file) 
mass = lambda_data[:,0]
td = lambda_data[:,1]

eos_name=opts.choice

#Calculate Detector PSD using A-LIGO Noise Curve
f_low=15.0
delta_f=1/pow(2,5)


chirpmass=np.linspace(0.18,1.2,120)
eos_function=interp1d(mass,td,kind='linear')

for i in range(0,len(chirpmass)):
    m1=mass1_from_mchirp_eta(chirpmass[i],0.25)
    m2=mass2_from_mchirp_eta(chirpmass[i],0.25)
    l1=float(eos_function(m1))
    l2=float(eos_function(m2))
    f_isco=f_schwarzchild_isco(m1+m2)
    hp, hc = get_td_waveform_from_fd(approximant = "TaylorF2", 
                             mass1 = m1, 
                             mass2 = m2,  
                             f_lower=f_low, 
                             f_final=f_isco, 
                             delta_f=delta_f,
                             delta_t=1.0/(8192)) 
                             
    delta_f=hp.delta_f
    psd = aLIGOZeroDetHighPower(len(hp), delta_f, f_low)
    #The normalization constant for the bbh waveform 
    hp_norm = sigmasq(hp, psd = psd, low_frequency_cutoff = f_low, high_frequency_cutoff=f_isco)
    
    #Generating the BNS waveform
    sp, sc = get_td_waveform_from_fd(approximant = "TaylorF2", 
                             mass1 = m1, 
                             mass2 = m2,  
                             f_lower=f_low, 
                             f_final=f_isco, 
                             delta_f=delta_f,
                             delta_t=1.0/(8192),
                             lambda1 = l1, 
                             lambda2 = l2)
    #Normalization 
    sp_norm = sigmasq(sp, psd = psd, low_frequency_cutoff = f_low,high_frequency_cutoff=f_isco)
    #Resizing the waveforms such that they are the same size
    wavelength = max(len(hp),len(sp))
    hp.resize(wavelength)
    sp.resize(wavelength)
    #Calculating the match between injection (hp) and template (sp)
    m, k = match(hp, sp, psd = psd,v1_norm = hp_norm, v2_norm =sp_norm,low_frequency_cutoff = f_low)
    output_data=str(chirpmass[i])+" "+str(m)+" "+str(m1)+" "+str(m2)+" "+str(l1)+" "+str(l2)+" "+str(f_isco)
    for a in range(5):

            try:

                chirp.connect()
                chirp.write(("{}\n".format(output_data)).encode(),'./overlap_data/mchirp_vs_match_{}_{}.txt'.format(opts.choice,cluster_process),'cwa')
                chirp.disconnect

                break
        
            except socket.timeout:

                time.sleep(5.)






