# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 21:50:43 2022

@author: madir
"""

import numpy as np
from bisect import bisect_left
from scipy import interpolate
import glob
import pdb


Model_data = {}


filenames=glob.glob('MIST_models/*M.track.eep')
M_star_all=np.zeros(len(filenames))
for i in range(len(filenames)): 
    if('\\' in filenames[i]): 
        M_star_all[i]=int(filenames[i].split('M.')[0].split('\\')[1])*1e-4
    elif('/' in filenames[i]): 
        M_star_all[i]=int(filenames[i].split('M.')[0].split('/')[1])*1e-4
def construct_grid(filename):

    eeps = np.genfromtxt(filename, skip_header=11, names=True)
    Data = {}
    Data['Age'] = eeps['star_age']
    Data['L_bol'] = eeps['log_L']

    return Data


for i in range(len(filenames)):
    Model_data[M_star_all[i]] = construct_grid(filenames[i])


def luminosity(Mstar, age):
    '''Returns the luminosity of a star of a given mass and age 
    
    Inputs: 
        Mstar - star mass in solar masses 
        age- age of the system in yrs'''
    
    
    if (bisect_left(M_star_all, Mstar) > 21): 
        mass_upper_pos=21
        mass_lower_pos = mass_upper_pos-1
        
        Mstarmin = M_star_all[mass_lower_pos]
        Mstarmax = M_star_all[mass_upper_pos]

        L_min, L_max = interpolation(Mstarmin, Mstarmax, Mstar,
                            Model_data[Mstarmin]['Age'], Model_data[Mstarmax]['Age'], age,
                            Model_data[Mstarmin]['L_bol'], Model_data[Mstarmax]['L_bol'])
        luminosity=10**L_max
        
    else: 
        mass_upper_pos = bisect_left(M_star_all, Mstar)
        
        mass_lower_pos = mass_upper_pos-1
        
    
        diff1= abs(Mstar-M_star_all[mass_lower_pos])
        diff2= abs(Mstar-M_star_all[mass_upper_pos])
    
        if diff2==0: 
            coeff1 = 1
            coeff2 = 0
        else: 
            coeff1 = diff1/(M_star_all[mass_upper_pos]-M_star_all[mass_lower_pos])
            coeff2 = diff2/(M_star_all[mass_upper_pos]-M_star_all[mass_lower_pos])
    
        Mstarmin = M_star_all[mass_lower_pos]
        Mstarmax = M_star_all[mass_upper_pos]

        L_min, L_max = interpolation(Mstarmin, Mstarmax, Mstar,
                        Model_data[Mstarmin]['Age'], Model_data[Mstarmax]['Age'], age,
                        Model_data[Mstarmin]['L_bol'], Model_data[Mstarmax]['L_bol'])
    
        interp_l_bol= L_min*coeff2 + L_max*coeff1
        luminosity=10**interp_l_bol
        
    return luminosity


def interpolation(Mstarmin, Mstarmax, Mstar, agearrmin, agearrmax, age, Larrmin, Larrmax):

    #get a luminosity value for a given age within a designated stellar mass bin
    #find the luminosity for the lower stellar mass bound
    L_func_min = interpolate.interp1d(agearrmin, Larrmin)
    L_min = L_func_min(age)
    #find the luminosity for the higher stellar mass bound
    L_func_max = interpolate.interp1d(agearrmax, Larrmax)
    L_max = L_func_max(age)
    

    return L_min, L_max

