#!/usr/bin/env python
# coding: utf-8

# # Functions for Gravity observation analysis
# 
# Pablo Marcos-Arenal
# 
# This file contains the independent functions required in the Jupyter Notebook *gravi_plot.ipynb*. 
# 
# ## Import modules

# First thing is importing all required modules
# 


import platform, sys, os
import numpy as np
from pylab import *
from astropy.convolution import interpolate_replace_nans, Gaussian1DKernel, convolve



# 
# 
# # Functions
# 
# ====================================================================================
# ## Continuum corrector
# This function provides the continuum corrected differential phase and visibility of a spectral line, and their errors.
# See Eqs. (2) and (3) in Kraus et al. 2012, ApJ, 744, 19.
# 
# $$F_l^2 V_l^2 = F^2 V^2 + F_c^2 V_c^2 - 2 · FV · F_c V_c ·\cos\phi$$
# $$\sin\phi_l = \sin\phi\frac{\mid FV\mid}{\mid F_l V_l \mid}$$
# $$F_l = F - F_c$$
# 
# The function is call as:
# 
# *continuum_corrector(wavelength, F, error_F, V2, V2_error, phase, phase_error, cont_F, cont_V2, result_Vl2, result_phasel)*
# 
# INPUT:
# - wavelength: wavelength [arbitrary units] (number or vector)
# - F, error_F: observed flux (and error) at wavelength [arbitrary units] (number or vector)
# - V2, V2_error: observed squared visibility (and error) at wavelength (number or vector)
# - phase, phase_error: observed differential phase (and error) at wavelength [degs] (number or vector)
# - cont_F: observed continuum flux close to wavelength [same arbitrary units as flux] (number)
# - cont_V2: observed continuum squared visibility close to wavelength (number)
# 
# OUTPUT:
# - result_Vl2: 3 columns array, first column wavelength [arbitrary units], second and third columns the corresponding continuum corrected squared visibilities and errors []. Array length is the number of "wavelengths" introduced.
# - result_phasel: 3 columns array, first column wavelength [arbitrary units], second and third columns the corresponding continuum corrected differential phases and errors [degs]. Array length is the number of "wavelengths" introduced.
# 


def continuum_corrector(wavelength, F, error_F, V2, V2_error, phase, phase_error, cont_F, cont_V2):
    V = sqrt(V2)
    V_error = V2_error / (2 * V)
    cont_V = sqrt(cont_V2)
    phase_rad = phase * (pi / 180)
    phase_rad_error = phase_error * (pi / 180)

    FlVl2 = abs(F * V)**2 + abs(cont_F * cont_V)**2 - (2 * cont_F * cont_V * F * V * cos(phase_rad))
    FlVl = sqrt(FlVl2)
    error_FlVl2 = sqrt(((2 * F * (V*V) - 2 * V * cont_V * cont_F * cos(phase_rad))**2) * error_F*error_F + 
                       ((2 * (F*F) * V - 2 * F * cont_F * cont_V * cos(phase_rad))**2) * (V_error*V_error) + 
                       ((2 * F * V * cont_F * cont_V * sin(phase_rad))**2) * (phase_rad_error*phase_rad_error))
    Vl = FlVl / (F - cont_F)
    Vl2 = FlVl2/ ((F - cont_F)**2)
    error_Vl = Vl * sqrt(((error_FlVl2 / (2 * FlVl2))**2) + ((error_F / (F - cont_F))**2))
    phasel_rad = arcsin(sin(phase_rad) * abs(F * V) / abs(FlVl))
    phasel_deg = phasel_rad * (180 / pi)
    sin_phasel_rad = sin(phasel_rad)
    error_sin_phasel_rad = sin_phasel_rad *                            sqrt(((error_F / F)**2) + ((V_error / V)**2) + ((phase_rad_error / tan(phase_rad))**2) + ((error_FlVl2 / (2 * FlVl2))**2))
    error_phasel_rad = error_sin_phasel_rad / sqrt(1 - (sin_phasel_rad**2))
    error_phasel_deg = error_phasel_rad * (180 / pi)
    
    #Set results of Vl2 and phasel in arrays including wavelength, values and errors
    result_Vl2 = np.empty((3, len(wavelength)))
    result_phasel = np.empty((3, len(wavelength)))
    result_Vl2 = [wavelength, Vl2, abs(2 * Vl * error_Vl)]
    result_phasel = [wavelength, phasel_deg, abs(error_phasel_deg)]
    

    return result_Vl2, result_phasel


# ====================================================================================
#
### Continuum corrector
#This function provides the continuum corrected differential phase and visibility of a spectral line, without their errors.
#See Eqs. (2) and (3) in Kraus et al. 2012, ApJ, 744, 19.
#
#$$F_l^2 V_l^2 = F^2 V^2 + F_c^2 V_c^2 - 2 · FV · F_c V_c ·\cos\phi$$
#$$\sin\phi_l = \sin\phi\frac{\mid FV\mid}{\mid F_l V_l \mid}$$
#$$F_l = F - F_c$$
#
#The function is call as:
#
#*continuum_corrector(wavelength, F, V2, phase, cont_F, cont_V2, result_Vl2, result_phasel)*
#
#INPUT:
#- wavelength: wavelength [arbitrary units] (number or vector)
#- F: observed flux at wavelength [arbitrary units] (number or vector)
#- V2: observed squared visibility at wavelength (number or vector)
#- phase: observed differential phase at wavelength [degs] (number or vector)
#- cont_F: observed continuum flux close to wavelength [same arbitrary units as flux] (number)
#- cont_V2: observed continuum squared visibility close to wavelength (number)
#
#OUTPUT:
#- result_Vl2: 2 columns array, first column wavelength [arbitrary units], second column the corresponding continuum corrected squared visibilities []. Array length is the number of "wavelengths" introduced.
#- result_phasel: 2 columns array, first column wavelength [arbitrary units], second column the corresponding continuum corrected differential phases [degs]. Array length is the number of "wavelengths" introduced.

def continuum_corrector2(wavelength, F, V2, phase, cont_F, cont_V2):
    V = sqrt(V2)
    cont_V = sqrt(cont_V2)
    phase_rad = phase * (pi / 180)

    FlVl2 = abs(F * V)**2 + abs(cont_F * cont_V)**2 - (2 * cont_F * cont_V * F * V * cos(phase_rad))
    FlVl = sqrt(FlVl2)

    Vl = FlVl / (F - cont_F)
    Vl2 = FlVl2/ ((F - cont_F)**2)

    phasel_rad = arcsin(sin(phase_rad) * abs(F * V) / abs(FlVl))
    phasel_deg = phasel_rad * (180 / pi)
    sin_phasel_rad = sin(phasel_rad)
    
    #Set results of Vl2 and phasel in arrays including wavelength and values 
    result_Vl2 = np.empty((2, len(wavelength)))
    result_phasel = np.empty((2, len(wavelength)))
    result_Vl2 = [wavelength, Vl2]
    result_phasel = [wavelength, phasel_deg]
    

   
   
    return result_Vl2, result_phasel
# ====================================================================================


# ====================================================================================
# 
# ## Disk size calculator
# This function provides the disk size (d) of a source given its distance (D) and angular size (${\delta}$).
# 
# $$d = 2 · D · tan \frac{\delta}{2}$$
# 
# 
# This function can be aproximated to :
# 
# $$d[AU] \simeq \frac{D[pc]·\delta[mas]}{1000}$$
# 
# (See figure for description)
# ![sources_size_calculation](figures/sources_size_calculation.jpeg)
# 
# The function is called as:
# 
# *disk_size_calculator(distance, angular_size)*
# 
# INPUT:
# - distance(D): distance to source [pc] 
# - angular_size(${\delta}$): angular size of source [mas] 
# 
# 
# OUTPUT:
# - disk_size[AU]

def disk_size_calculator(distance, angular_size):
    
    disk_size =  distance * angular_size / 1000
   
    return disk_size

# ====================================================================================
### Disk size error calculator
#This function provides the disk size (d) of a source given its distance (D) and angular size (${\delta}$), and its errors.
#
#$$d = 2 · D · tan \frac{\delta}{2}$$
#
#
#This function can be aproximated to :
#
#$$d[au] \simeq \frac{D[pc]·\delta[mas]}{1000}$$
#
#Error in disk size is:
#$$\Delta d[au] = \lvert \frac{D[pc]}{1000}\rvert·\Delta \delta[mas]+ \lvert \frac{\delta[mas]}{1000}\rvert·\Delta D[pc]$$
#
#
#The function is called as:
#*disk_size_error_calculator(distance, distance_error, angular_size, angular_size_error)*
#
#INPUT:
#- distance(D): distance to source [pc] 
#- distance_error: error in distance to source [pc] 
#- angular_size(${\delta}$): angular size of source [mas] 
#- angular_size_error: error in angular size of source [mas] 
#
#
#OUTPUT:
#- disk_size[au], disk_size_error[au]


def disk_size_error_calculator(distance, distance_error, angular_size, angular_size_error):
    
    disk_size =  distance * angular_size / 1000
    
    disk_size_error = (1 / 1000) * (distance * angular_size_error + angular_size * distance_error)
   
    return disk_size, disk_size_error

# ====================================================================================
# 
# ## Angular size calculator
# This function provides the angular size (${\delta}$) of a source given its distance (D) and disk size (d).
# 
# $${\delta} = 2 · arctan \frac{d}{2D}$$
# 
# 
# This function can be aproximated to :
# 
# $$\delta[mas] \simeq \frac{1000·d[AU]}{D[pc]}$$
# 
# (See figure for description)
# ![sources_size_calculation](figures/sources_size_calculation.jpeg)
# 
# The function is called as:
# 
# 
# *angular_size_calculator(distance, disk_size)*
# 
# INPUT:
# - distance(D): distance to source [pc] 
# - disk_size(d): disk size of source [AU] 
# 
# 
# OUTPUT:
# - angular_size(${\delta}$): angular size of disk [mas]



def angular_size_calculator(distance, disk_size):
    
    angular_size = 1000*disk_size/distance
   
    return angular_size 

# ====================================================================================
# 
def calculate_max_angular_resolution_mas(baseline,wavelength=2.1667):  
    """
    This function calculates the maximum angular resolution of an observation at a given wavelegth (\lambda) based on its maximum projected baseline (B) as $\lambda$/2B.
        
    When the observed source is unresolved, its size can be given as an upper limit for certain emitting wavelength.
    This upper limit is given by $\lambda$/2B.
    
    The function is called as:
    
    *max_angular_resolution(baseline,wavelength)*
    
    
    INPUT:
    - baseline: largest projected baseline [meters] 
    - wavelength ($\lambda$): Wavelength of the emitting region whose size is calculated [Default value Br$\gamma$ = 2.1667 $\mu$m]
        
    OUTPUT:
    - max_angular_resolution_mas: maximum angular resolution [mas].
    
    """

    #Calculate maximum angular resolution in radians (wavelength is given in microns)
    max_angular_resolution = (wavelength/1000000)/(2*baseline) # in radians
    
    #Transform to mas 
    max_angular_resolution_mas = max_angular_resolution * (180/np.pi) *3600 *1000 
    
    return max_angular_resolution_mas

# ====================================================================================
#     
def replace_nans_by_interpolated_gaussian(data_array, convolution_range = 10):  
    """
    This function retrieves an array with NaN values and replaces these values by their gaussian convolved values in the corresponding positions. 
    
    INPUT:
    - data_array: Array including NaN values. This *data_array* should be flux, visibilitiy of differential phase. 
    - convolution_range: Convolution kernel range. By default is set to 10. 
        
    OUTPUT:
    - result_without_nans: Array without NaN values.
    
    """


    #Copy input data_array 
    input_copy_with_nan = data_array.copy()
    
    #Initialize output result data array
    result_without_nan = input_copy_with_nan
    
        
    #Gaussian kernel must be odd size (That's why x_size=len(input_copy_with_nan)-1,)
    gaussian_kernel = Gaussian1DKernel(stddev=convolution_range, x_size=len(input_copy_with_nan)-1, mode='oversample')
        

    #Sustitute NaNs by gaussian convolved values around that value
    result_without_nan = interpolate_replace_nans(input_copy_with_nan,gaussian_kernel)  
    
    #Some cleaning
    del input_copy_with_nan
    
    return result_without_nan

# ====================================================================================
#     
def auxiliary_telescope_names_to_UT(AT_array):
    """
    Transform Auxiliary Telescopes (AT; 1.8mdiameter) names to  Unit Telescope (UT; 8mdiameter)  
    (for HD141926 to use the same processing file than the rest of the sources on the sample).
    
    """
    
    UT_array = AT_array

    #Iteration in all repeated dict keys to ensure values are replaced in all of them (there are 6 FLUX)
    for i in range(6):
        for key1,value1 in UT_array.items():  
    
            if isinstance(value1, dict):
                for key2, value2 in value1.items():
                                    
                    #For Flux:
                    if key2=='C1':
                        value1['U4']=value1.pop(key2)
                    if key2=='D0':
                        value1['U3']=value1.pop(key2)
                    if key2=='B2':
                        value1['U2']=value1.pop(key2)
                    if key2=='A0':
                        value1['U1']=value1.pop(key2)
            
                    #For visibility:
                    if key2=='C1D0':
                        value1['U4U3']=value1.pop(key2)
                    if key2=='C1B2':
                        value1['U4U2']=value1.pop(key2)
                    if key2=='C1A0':
                        value1['U4U1']=value1.pop(key2)
                    if key2=='D0B2':
                        value1['U3U2']=value1.pop(key2)
                    if key2=='D0A0':
                        value1['U3U1']=value1.pop(key2)
                    if key2=='B2A0':
                        value1['U2U1']=value1.pop(key2)
            
                    #For phase closure:
                    if key2=='C1D0B2':
                        value1['U4U3U2']=value1.pop(key2)
                    if key2=='C1D0A0':
                        value1['U4U3U1']=value1.pop(key2)
                    if key2=='C1B2A0':
                        value1['U4U2U1']=value1.pop(key2)
                    if key2=='D0B2A0':
                        value1['U3U2U1']=value1.pop(key2)
    

    return UT_array
# ====================================================================================
#     


def disk_inclination_calculator(elongation, elongation_error):
    """
    This function provides the disk inclination ($i$) given its elongation ratio (e).

    $$\cos(i) = \frac{b}{a}$$
    
    Since its elongation ratio is 
    $$e = \frac{a}{b}$$ 
    
    with a = major axis and b = minor axis,
    this function can be aproximated to :
    
    $$i = \frac{180}{\pi}·\arccos\frac{1}{e}$$
    
    $$\Delta i = \frac{180}{\pi}·\lvert \frac{1}{e^4 - e^2}\rvert \Delta e$$
 
    INPUT:
    - elongation: disk major axis divided by disk minor axis 
    - elongation_error: error in elongation ratio 
        
    OUTPUT:
    - inclination: angle of inclination of the disk [deg] 
    - inclination_error: error in angle of inclination of the disk [deg] 
    """
    
    
    inclination = (180 / np.pi) * np.arccos(np.divide(1,elongation))

    inclination_error = (180/ np.pi) * (np.divide(1,(np.power(elongation,4) - np.power(elongation,2)))) * elongation_error
    
    return inclination, inclination_error    
    
# ====================================================================================
#     
