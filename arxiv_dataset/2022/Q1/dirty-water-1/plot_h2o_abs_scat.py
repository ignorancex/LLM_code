 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sukrit

Purpose: plot parameters for pure water.
"""
import numpy as np #This furnishes array operations.
import scipy as sp
from scipy import interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt #this furnishes plotting operations
import pickle
import scipy
import scipy.integrate
import pdb


##############
###Quickenden+1980
##############
# waterfilename='/Users/sukrit/Documents/Research_Code/Python/MITPDF/Dirty_Water/Azra_Project/RSI_UV/Processed-Data/quickenden.dat' #From Azra's work, copied from Table 2 of Quickenden & Irvin 1980. 
# q1980_wav, q1980_dec_ext=np.genfromtxt(waterfilename, skip_header=2, skip_footer=0, unpack=True) #wavelength, nm ; decadic extinction (NOT ABSORPTION?!?!) in cm**-1. 

q1980_wav, q1980_dec_ext, q1980_dec_ext_err= np.genfromtxt('./Azra_Project/quickenden_with_uncerts.dat', skip_header=2, unpack=True, usecols=(0,1,2))#nm, cm**-1. Includes both scattering, absorption. 


##############
###Krockel+2014
##############
##They say they are pretty much equivalent so skipping.

##############
###Rayleigh scattering for liquid water
##############

###Formalism from Krockel & Schmidt 2014
def rayleigh_h2o_krockel2014(wav):
    """
    Decadic extinction due to Rayleigh scattering.
    
    wav: in units of nm. 
    Values of all parameters following Krockel+2014 (near their equation 4)
    
    For now n and (dn/dp)_T are taken as constant. Formally WRONG, let's fix it later. 
    ####n(lambda) and (dn/dp)_T taken from Weiss+2012 following Krockel+2014, but honestly can probably be approximated as constant and would be fine. 
    
    Convert all to SI, work, convert all back to useful units. 
    """
    rho=0.108 #average value, based on VIS.
    Beta_T=4.52472e-10 #units: N**-1 m**2, value at 25C
    k_boltz=1.38054e-23 #J K**-1
    T=298.15 #K
    
    n=1.39 #mean inferred from Krockell+2003 Fig 3
    dndp=0.095 #mean inferred from Krockell+2003 Fig 3. Units: um**2 kN**-1.
    
    wav_si=wav*1.e-9 #convert nm to m
    dndp_si=dndp*1.e-9 #convert from kN**-1 um**2 to Pa^-1
        
    prefactor=32.0*np.pi**3.*k_boltz*T/(3.0*wav_si**4.*Beta_T*np.log(10.0))
    term1=n**2.*(dndp_si)**2.
    term2=((6.0+3.0*rho)/(6.0-3.0*rho))

    decadic_ext_coeff_si=prefactor*term1*term2 #units: m**-1
    
    decadic_ext_coeff=decadic_ext_coeff_si*1.0e-2 #convert m**-1 to cm**-1
    return decadic_ext_coeff

rayleigh_krockel2014=rayleigh_h2o_krockel2014(q1980_wav) #cm**-1

#############
##Plot
#############
fig, ax=plt.subplots(3, figsize=(8., 10.), sharex=True)

# ax[0].plot(q1980_wav, q1980_dec_ext, linewidth=2, linestyle='-', marker='d', color='black', label='Decadic Extinction (Quickenden+1980)')
ax[0].errorbar(q1980_wav, q1980_dec_ext, yerr=q1980_dec_ext_err, linewidth=2, linestyle='-', marker='d', color='black', label='Decadic Extinction (Quickenden+1980)')
ax[0].plot(q1980_wav, rayleigh_krockel2014, linewidth=2, linestyle='-', marker='d', color='blue', label='Decadic Scattering (After Krockel+2014)')
ax[0].set_yscale('log')
ax[0].set_ylabel(r'Decadic Extinction Coefficient (cm$^{-1}$)')
ax[0].legend(ncol=1, loc='best')

# ax[1].plot(q1980_wav, rayleigh_krockel2014/q1980_dec_ext, linewidth=2, linestyle='-', marker='d', color='black')
ax[1].errorbar(q1980_wav, rayleigh_krockel2014/q1980_dec_ext, yerr=rayleigh_krockel2014*q1980_dec_ext_err/(q1980_dec_ext)**2.0, linewidth=2, linestyle='-', marker='d', color='black')
ax[1].set_yscale('linear')
ax[1].set_ylabel('Single-Scattering Albedo')
ax[1].set_ylim([0, 1])

ax[2].plot(q1980_wav, np.log10(np.exp(1.0))/rayleigh_krockel2014, linewidth=2, linestyle='-', marker='d', color='black',)
ax[2].set_yscale('log')
ax[2].set_ylabel(r'Depth for $\tau_s=1$ (cm)')
ax[2].set_ylim([1.0e3, 1.0e5])


ax[2].set_xscale('linear')
ax[2].set_xlabel('Wavelength (nm)')
ax[2].set_xlim([200., 320.])

plt.savefig('./h2o_l_abs_scat.pdf', orientation='portrait',papertype='letter', format='pdf')

#############
##Generate table for future use
#############
data=np.zeros([len(q1980_wav), 4])
data[:,0]=q1980_wav
data[:,1]=q1980_dec_ext
data[:,2]=rayleigh_krockel2014
data[:,3]=q1980_dec_ext-rayleigh_krockel2014

np.savetxt("./pure_h2o.dat", data, delimiter=",", newline="\n", fmt="%3.1f %1.6e %1.6e %1.6e", header="Pure H2O (Via Quickenden+1980, Krockel+2014)\nWavelength (nm), Total Extinction (cm**-1), Rayleigh Scattering (cm**-1), Absorption (cm**-1)")
