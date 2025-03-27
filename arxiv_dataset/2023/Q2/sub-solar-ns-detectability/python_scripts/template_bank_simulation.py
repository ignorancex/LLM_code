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
import scipy.interpolate as interp
import pycbc
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.filter import match, sigmasq
from pycbc.waveform import get_fd_waveform
from pycbc.conversions import tau0_from_mass1_mass2, tau3_from_mass1_mass2, f_schwarzchild_isco
from scipy.interpolate import griddata

import numpy.random
numpy.random.seed(None)

parser=argparse.ArgumentParser()
parser.add_argument("--choice", type=str, choices=["bns_rlo", "bns_isco", "bbh_rlo","bbh_isco"], required=True)
#The input file should contain at least 3 columns, with mass,radius,lambda
parser.add_argument("--input-file1", type=str, required=True)
parser.add_argument("--input-file2", type=str, required=True)

opts = parser.parse_args()

# read Template Bank data
data = np.loadtxt('temp_bank_NonSpin-15Hz-m1m2.txt',delimiter =',')
temp_m1 = data[:,0]
temp_m2 = data[:,1]

#Importing Tidal Deformability Data
lambda_data = np.loadtxt(opts.input_file1, skiprows=1) 
sim_m = lambda_data[:,0]
sim_l = lambda_data[:,1]


if opts.choice == "bns_rlo" or opts.choice == "bns_isco":
    #1D interpolation of lambda as a function of mass using the BSK21 EOS
    tidal_deformability = interp.interp1d(sim_m, sim_l, kind = 'linear')
else:
    sim_l=np.zeros(len(sim_m))
    tidal_deformability = interp.interp1d(sim_m, sim_l, kind = 'linear')

#Rohce Lobe Overflow Frequency using q!=1 and SLY EOS
Lorene_data=np.loadtxt(opts.input_file2,skiprows=1)
m1 = Lorene_data[:,1]
m2 = Lorene_data[:,0]
f_Lorene  = Lorene_data[:,2]
f_rlo=2*f_Lorene

#points and values arrays are arrays of (m1,m2) and (coalescence frequency), respectively.
#These arrays will be used for interpolation purposes
points = []

for i in range(len(m1)):
    points.append([m1[i],m2[i]])

values = f_rlo

#Calculate Detector PSD using A-LIGO Noise Curve
f_low = 15. 
srate = 16384
N = 1048576
f_len = int( N/2+1)
delta_f = float(srate)/float(N)
psd = aLIGOZeroDetHighPower(f_len, delta_f, f_low)


os.environ['_CONDOR_SCRATCH_DIR']='/srv'
home_dir = os.environ['_CONDOR_SCRATCH_DIR']
chirp = htchirp.HTChirp()
cluster_process = os.environ['cluster_process_var']


#A difference of 3 in the tau_0  parameter is used to filter out templates that are too "far away" from the injection
delta_tau_0 = 3.0

#min and max m1, m2 of the template bank
m1_min = min(temp_m1)
m1_max = max(temp_m1)
m2_min = min(temp_m2)
m2_max = max(temp_m2)

#Find tau_0 for each template                                                            
temp_tau_0_list = []
for j in range(len(temp_m1)):
    temp_tau_0_list.append(tau0_from_mass1_mass2(temp_m1[j],temp_m2[j],f_low))

while True:

    #Choose a random mass from simulation range and corresponding lambda
    rand_m1 = numpy.random.uniform(m1_min,m1_max)
    rand_m2 = numpy.random.uniform(m2_min,m2_max)
    #Ensure that the mass ratio is >1, since the assumed mass ratio in the RLO-frequency simulations is >1
    if rand_m1<rand_m2:
        rand_m1, rand_m2= rand_m2,rand_m1

    f_merger=0
    #The merger frequency is taken to be Roche Lobe overflow frequency or ISCO frequency, depending on argument choice.
    if opts.choice == "bns_rlo" or opts.choice == "bbh_rlo":
        f_merger = griddata(points, values, (rand_m1, rand_m2), method='cubic')
    else:
        f_merger=f_schwarzchild_isco(rand_m1+rand_m2)
    
    #The interpolated value of tidal deformability from the BSk21_F EOS 
    rand_l1 = float(tidal_deformability(rand_m1))                                                                                            
    rand_l2 = float(tidal_deformability(rand_m2))
    
    
    #Find tau_0 for injection and do +/- 3 seconds difference between tau_0 of injection and templates to minimize template bank match
    
    inj_tau_0 = tau0_from_mass1_mass2(rand_m1,rand_m2,f_low)
    
    #Lists that will be appended with the template parameters that were "close" to the injection
    temp_m1_final = []
    temp_m2_final = []
    temp_tau_0_final_list = []
    
    #Appending to these lists
    for k in range(len(temp_m1)):
        if np.abs(temp_tau_0_list[k]-inj_tau_0) <= delta_tau_0:
            temp_tau_0_final_list.append(temp_tau_0_list[k])
            temp_m1_final.append(temp_m1[k])
            temp_m2_final.append(temp_m2[k])
    
            
    #Generate BSk21_F injection waveform
    hp, hc = get_fd_waveform(approximant = "TaylorF2", 
                             mass1 = rand_m1, 
                             mass2 = rand_m2,  
                             f_lower=f_low, 
                             f_final=f_merger, 
                             delta_f=delta_f,
                             lambda1 = rand_l1, 
                             lambda2 = rand_l2)
    
    #We will match the injected waveform with a template waveform which gives a match >0. 
    # We keep matching with all waveforms and update this max_overlap variable if it is greater 
    # than the match of all previous match calculations.
    max_overlap = 0
    delta_f=hp.delta_f
    psd = aLIGOZeroDetHighPower(len(hp), delta_f, f_low)
    #The normalization constant for the injection 
    hp_norm = sigmasq(hp, psd = psd, low_frequency_cutoff = f_low, high_frequency_cutoff=f_merger)
    
    #Match with BBH waveform having same component masses
    f_isco=f_schwarzchild_isco(rand_m1+rand_m2)
    xp,xc= get_fd_waveform(approximant = "TaylorF2", 
                             mass1 = rand_m1, 
                             mass2 = rand_m2,  
                             f_lower=f_low, 
                             f_final=f_isco, 
                             delta_f=delta_f)
    delta_f=hp.delta_f
    psd = aLIGOZeroDetHighPower(len(xp), delta_f, f_low)
    #The normalization constant for the injection 
    xp_norm = sigmasq(xp, psd = psd, low_frequency_cutoff = f_low, high_frequency_cutoff=f_isco)
    wavelength = max(len(hp),len(xp))
    hp.resize(wavelength)
    xp.resize(wavelength)
    delta_f=xp.delta_f
    psd = aLIGOZeroDetHighPower(len(xp), delta_f, f_low)
    #Calculating the match between injection (hp) and template (sp)
    match_same_mass, _ = match(hp, xp, psd = psd, v1_norm = hp_norm, v2_norm =xp_norm, low_frequency_cutoff = f_low)     

    #Here we will create a template waveform, find its normalization constant, then match it with the injection. We then
    # comapare this match with max_overlap and continue until we match with all the templates in temp_tau_0_final_list.
    #The parameters and match value of the template that gave the best match are fed into an output file.
    for t in range(len(temp_tau_0_final_list)):
   
        #Template (BBH waveform) 
        f_isco=f_schwarzchild_isco(temp_m1_final[t]+temp_m2_final[t])
        sp, sc = get_fd_waveform(approximant = "TaylorF2", 
                                 mass1 = temp_m1_final[t], 
                                 mass2 = temp_m2_final[t],  
                                 f_lower=f_low, 
                                 f_final=f_isco, 
                                 delta_f=delta_f)
        delta_f=sp.delta_f
        psd = aLIGOZeroDetHighPower(len(sp), delta_f, f_low)
        #Normalization constant of template
        sp_norm = sigmasq(sp, psd = psd, low_frequency_cutoff = f_low, high_frequency_cutoff=f_isco)                                    
       
        #Resizing the waveforms such that they are the same size
        wavelength = max(len(hp),len(sp))
        hp.resize(wavelength)
        sp.resize(wavelength)
        delta_f=sp.delta_f
        psd = aLIGOZeroDetHighPower(len(sp), delta_f, f_low)
        #Calculating the match between injection (hp) and template (sp)
        m, _ = match(hp, sp, psd = psd, v1_norm = hp_norm, v2_norm =sp_norm, low_frequency_cutoff = f_low)     
        
        
        #Comparing this match with the max_overlap. If this match is bigger then we update max_overlap. If it's not
        # we move on to the next template.
        if(m > max_overlap): 
            max_overlap = m
            m1_best_guess = temp_m1_final[t]
            m2_best_guess = temp_m2_final[t]
                   
    #We are now done with looping over all the templates that were "close" to the injection and found the 
    # template that gave the best match.
    output_data = str(rand_m1)+' '+str(rand_m2)+' '+str(rand_l1)+' '+str(rand_l2)+' '+str(m1_best_guess)+' '+str(m2_best_guess)+' '+str(max_overlap)+' '+str(match_same_mass)+' '+str(f_merger)
    print(output_data)
    
    for i in range(0,5):

        try:

            chirp.connect()
            chirp.write(("{}\n".format(output_data)).encode(),'./simulation_data_{}/banksims_{}_{}.txt'.format(opts.choice,opts.choice,cluster_process),'cwa')
            chirp.disconnect

            break
        
        except socket.timeout:

            time.sleep(5.)
           
            pass


    sys.stdout.flush()



