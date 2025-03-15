#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 12:18:44 2021

@author: sukrit

To back out CO3, HCO3 limits
"""
import numpy as np #This furnishes array operations.
import scipy as sp
from scipy import interpolate as interp
import matplotlib.pyplot as plt
import pickle
import scipy
import scipy.integrate
import pdb
import pandas as pd

#####
#First, import data.
#####
ctrl_pH_time, ctrl_pH_low, ctrl_pH_middle, ctrl_pH_high=np.genfromtxt('./kadoya/pH/control_pH_o.dat', skip_header=1, skip_footer=0, unpack=True) 
std_pH_time, std_pH_low, std_pH_middle, std_pH_high=np.genfromtxt('./kadoya/pH/Standard_pH_o.dat', skip_header=1, skip_footer=0, unpack=True) 

#DIC is in units of log10(mol/Kg)
ctrl_DIC_time, ctrl_DIC_low, ctrl_DIC_middle, ctrl_DIC_high=np.genfromtxt('./kadoya/DIC/control_DIC_o.dat', skip_header=1, skip_footer=0, unpack=True) 
std_DIC_time, std_DIC_low, std_DIC_middle, std_DIC_high=np.genfromtxt('./kadoya/DIC/Standard_DIC_o.dat', skip_header=1, skip_footer=0, unpack=True) 

#Convert DIC to physical units mol/Kg ~ M
ctrl_DIC_low=10.0**ctrl_DIC_low
ctrl_DIC_middle=10.0**ctrl_DIC_middle
ctrl_DIC_high=10.0**ctrl_DIC_high

std_DIC_low=10.0**std_DIC_low
std_DIC_middle=10.0**std_DIC_middle
std_DIC_high=10.0**std_DIC_high


#####
#Next, specify speciation constants
#####

pKa_co2_1=6.35 #H2CO3 --> H + HCO3-, Rumble+2017 va Ranjan+2018
pKa_co2_2=10.33 #HCO3- --> H+ + CO3(2-), Rumble+2017 va Ranjan+2018

#####
#Get carbonate values from calculation. 
#####
def get_carbonate_speciation(DIC, pH):
    """
    Calculate [H2CO3], [HCO3-], [CO3(2-)] given DIC (just those 3 species) and pH.
    Returns: H2CO3, HCO3-, CO3(2-).
    """
    
    conc_h2co3=DIC/(1.0+10.0**(pH-pKa_co2_1) + 10.0**(2.0*pH-pKa_co2_1-pKa_co2_2))
    conc_hco3=conc_h2co3*10.0**(pH-pKa_co2_1)
    conc_co3=conc_h2co3*10.0**(2.0*pH-pKa_co2_1-pKa_co2_2)
    
    return conc_h2co3, conc_hco3, conc_co3
    
ctrl_h2co3_low, ctrl_hco3_low, ctrl_co3_low=get_carbonate_speciation(ctrl_DIC_low, ctrl_pH_low)   
ctrl_h2co3_middle, ctrl_hco3_middle, ctrl_co3_middle=get_carbonate_speciation(ctrl_DIC_middle, ctrl_pH_middle)   
ctrl_h2co3_high, ctrl_hco3_high, ctrl_co3_high=get_carbonate_speciation(ctrl_DIC_high, ctrl_pH_high)   

std_h2co3_low, std_hco3_low, std_co3_low=get_carbonate_speciation(std_DIC_low, std_pH_low)   
std_h2co3_middle, std_hco3_middle, std_co3_middle=get_carbonate_speciation(std_DIC_middle, std_pH_middle)   
std_h2co3_high, std_hco3_high, std_co3_high=get_carbonate_speciation(std_DIC_high, std_pH_high)

#####
#Plot
#####

def plot_kadoya_carbonate_speciation(title, time, DIC_low, DIC_middle, DIC_high, pH_low, pH_middle, pH_high, h2co3_low, h2co3_middle, h2co3_high, hco3_low, hco3_middle, hco3_high, co3_low, co3_middle, co3_high):
    fig, ax=plt.subplots(3, figsize=(8., 10.), sharex=True)
    
    ax[0].set_title(title)
    ax[0].plot(time, pH_low, color='black', linestyle='--')
    ax[0].plot(time, pH_middle, color='black', linestyle='-')
    ax[0].plot(time, pH_high, color='black', linestyle=':')
    ax[0].set_xlabel('Ga')
    ax[0].set_ylabel('pH')
    ax[0].set_yscale('linear')
    
    ax[1].plot(time, DIC_low, color='black', linestyle='--')
    ax[1].plot(time, DIC_middle, color='black', linestyle='-')
    ax[1].plot(time, DIC_high, color='black', linestyle=':')
    ax[1].set_xlabel('Ga')
    ax[1].set_ylabel('DIC (mol/Kg)')
    ax[1].set_yscale('log')
    

    ax[2].plot(time, h2co3_low, color='black', linestyle='--')
    ax[2].plot(time, h2co3_middle, color='black', linestyle='-', label='H2CO3')
    ax[2].plot(time, h2co3_high, color='black', linestyle=':')
    ax[2].plot(time, hco3_low, color='red', linestyle='--')
    ax[2].plot(time, hco3_middle, color='red', linestyle='-', label='HCO3-')
    ax[2].plot(time, hco3_high, color='red', linestyle=':')
    ax[2].plot(time, co3_low, color='blue', linestyle='--')
    ax[2].plot(time, co3_middle, color='blue', linestyle='-', label='CO3(2-)')
    ax[2].plot(time, co3_high, color='blue', linestyle=':')
    ax[2].set_xlabel('Ga')
    ax[2].set_ylabel('Concentration (mol/Kg)')
    ax[2].set_yscale('log')    
    ax[2].set_xscale('linear')
    ax[2].set_xlim([3.5, 4.4])
    ax[2].legend()


###Get carbonate, bicarbonate concentrations for prebiotic ocean. Look at below plots and for 3.9 Ga take the spanning values for each.
plot_kadoya_carbonate_speciation('Standard', std_pH_time, std_DIC_low, std_DIC_middle, std_DIC_high, std_pH_low, std_pH_middle, std_pH_high, std_h2co3_low, std_h2co3_middle, std_h2co3_high, std_hco3_low, std_hco3_middle, std_hco3_high, std_co3_low, std_co3_middle, std_co3_high)

plot_kadoya_carbonate_speciation('Control', ctrl_pH_time, ctrl_DIC_low, ctrl_DIC_middle, ctrl_DIC_high, ctrl_pH_low, ctrl_pH_middle, ctrl_pH_high, ctrl_h2co3_low, ctrl_h2co3_middle, ctrl_h2co3_high, ctrl_hco3_low, ctrl_hco3_middle, ctrl_hco3_high, ctrl_co3_low, ctrl_co3_middle, ctrl_co3_high)

#HCO3: minimum is 0.002 from std. maximum is 0.2 from std.
#CO3: minumum is 2E-7=0.2 uM from std. maximum is 0.001 = 1 mM from std.


###Get carbonate, bicarbonate spanning values for freshwater lake. 
#HCO3: 5.70*0.187 mM = 1.0659 mM = 1 mM
print('mM')
print(5.70*0.187)#mM
#CO3: calculate from HCO3 and pH=6.34. Multiply by 10**(pH-pKa)
print((5.70*0.187)*10**(6.34-pKa_co2_2))#mM


###Get carbonate, bicarbonate spanning values for carbonate lake. 
print('M')
#pCO2=0.01, pH=9
print(0.01*3.3E-2*10**(9.0-pKa_co2_1)) #M, HCO3
print(0.01*3.3E-2*10**(2.0*9.0-pKa_co2_2-pKa_co2_1)) #M, CO3

#pCO2=1, pH=9
print(1.0*3.3E-2*10**(6.5-pKa_co2_1)) #M, HCO3
print(1.0*3.3E-2*10**(2.0*6.5-pKa_co2_2-pKa_co2_1)) #M, CO3--