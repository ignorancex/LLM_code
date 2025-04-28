#!/usr/bin/env python

import os
import sys
import h5py
import glob
import numpy as np
import logging
import pycbc
from pycbc import detector
from pycbc.psd import from_txt, from_string
from pycbc.types import TimeSeries, zeros, float64
from pycbc.inject import InjectionSet
from pycbc.filter import sigmasq, matched_filter
from pycbc.noise import noise_from_psd
from pycbc.waveform import get_fd_waveform
from pycbc.pool import BroadcastPool
from pycbc.cosmology import redshift
from scipy.interpolate import interp1d
from subprocess import call


injdir=sys.argv[1]


fp=open(injdir+"/snr_output/injection_snrs.txt','r')
lines=fp.readlines()

file=open(injdir+"/snr_output/injection_snrs_filtered.txt','w')


i=0
for x in lines:
    y=x.split(' ')[1]
    y=y[0:len(y)-1]
    snr=float(y)
    inj=x.split(' ')[0]
    injfile=injdir+inj
    if(snr<30.0):
        if(os.path.exists(injfile)):
            os.unlink(injfile)
    else: 
        if(os.path.exists(injfile)):
            file.write(x)
    i=i+1
        
    
fp.close()   
file.close()