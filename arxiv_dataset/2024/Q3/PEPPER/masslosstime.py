# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 10:29:40 2022

@author: madir
"""

import numpy as np 
import planetstructure as ps
import constants as cs
from scipy.optimize import minimize_scalar
import pdb

get_shape = cs.eff_scalings.item()

def calc_efficiency(Rp,Mp, constant=False):
    '''Returns the mass loss efficiency of the planet as a function of its escape velocity 
        
    Parameters: 
        Mp - mass of the planet in grams
        Rp - planet radius in cm
        constant - boolean, if True use constant efficiency rate, if False use scaled'''
        
    if constant:
        return 0.1
    else:
        # returns the scaled efficiency factor of the Owen & Jackson (2012) evaporation rates

        # the shape is scaled via the escape velocity to a mass of 7.603262769401823e+27 g

        Mp_scale = 7.603262769401823e+27

        # Radius scale first 

        scaled_eff = 10.**get_shape(np.log10(Rp*Mp_scale/Mp))

        # this scaled efficiency returns either a scaled value (scaled to max value in table) for the
        # efficiency or extropolates the efficiency at a constant value of large planets
        # that would be undergoing Roche Lobe overflow in the Owen & Jackson (2012) tables
        # E.g. grey region of Figure 5 in Owen & Jackson (2012)

        # scale mass

        mass_scale = (1. + (np.sqrt(Mp/1e29))**10.)**(1./10.)
    
        return scaled_eff * mass_scale
    

def calc_masslosstime_rocky(system, Rcore, Mcore, a, Teq, Xiron, Tkh_Myr):
    '''Returns the maximized envelope width and the maximized scaled mass loss timescale for the rocky planet 
    
    Parameters: 
        system - Planetary system object containing a rocky and enveloped planet and a star
        Rcore - the radius of the rocky planet in Earth radii
        Mcore - the mass of the rocky planet in Earth masses
        a - the semimajor axis of the rocky planet in cm
        Teq - rocky planet equilibrium temp, K
        Xiron - iron mass fraction
        Tkh_Myr - cooling timescale in Myr'''
    
    #set constraints on envelope width for optimization function
    DR_rcb_min = 0.1 * cs.Rearth2cm(Rcore)
    DR_rcb_max = 5 * cs.Rearth2cm(Rcore)
    #print(Rcore)
    #optimization function that finds the value of DR_rcb that maximize mass loss time scale
    result=minimize_scalar(max_tmdot_objective, bounds=(DR_rcb_min, DR_rcb_max), args=(Rcore, Mcore, Teq, Xiron, Tkh_Myr), method='bounded')
    
    if (result.success):
        DR_rcbmaximized = result.x
        #find maximized envelope mass fraction 
        #X_max = ps.calc_X_adiabatic(cs.Rearth2cm(Rcore), DR_rcbmaximized, Tkh_Myr, Teq, Xiron)
        #find eta using Rp
        X_max, _, Rplanet = ps.calc_Rplanet(cs.Rearth2cm(Rcore), DR_rcbmaximized, Tkh_Myr, Teq, Xiron)
        #eta = calc_efficiency(cs.Mearth2g(Mcore), Rplanet)
        eta = calc_efficiency( Rplanet, cs.Mearth2g(Mcore))
        
        masslosstime_max_rocky=masslosstime_eq(Mcore, a, X_max, Rplanet, eta)
        
        system.planetRocky.X_max_PE = X_max 
        system.planetRocky.DR_rcbmax_PE = DR_rcbmaximized
        system.planetRocky.scaled_masslosstime_max_PE = masslosstime_max_rocky
        system.planetRocky.eta= eta
        
        return masslosstime_max_rocky
    
    else: 
        print('Failed to find solution for maximum mass-loss timescale for rocky planet with radius', Rcore)
        return -1
        
def max_tmdot_objective(DR_rcb, Rcore, Mcore, Teq, Xiron, Tkh_Myr):
    '''Returns the scaled mass loss timescale that we wish to maximize
    
    Parameters: 
        DR_rcb - width of the envelope to the rcb in cm
        Rcore - the radius of the rocky planet in Earth radii
        Mcore - the mass of the rocky planet in Earth masses
        Teq - rocky planet equilibrium temp, K
        Xiron - iron mass fraction
        Tkh_Myr - cooling timescale in Myr'''
    
   
    X, _, Rplanet = ps.calc_Rplanet(cs.Rearth2cm(Rcore), DR_rcb, Tkh_Myr, Teq, Xiron)

    eta = calc_efficiency( Rplanet, cs.Mearth2g(Mcore))
    
    func_to_max = masslosstime_eq(1, 1, X, Rplanet, eta)
    
    return 1/func_to_max    
    
    
def masslosstime_eq(Mcore, a, X, Rplanet, eta):
    '''Returns the scaled mass loss timescale of a given planet
    
    Parameters: 
        Mcore - planet mass in Earth masses
        a - semimajor axis of the planet in cm
        X - envelope mass fraction
        Rplanet - planet radius in cm
        eta - atmospheric mass loss efficiency in cgs units'''

    masslosstime= X* (cs.Mearth2g(Mcore)**2) * a**2/ (Rplanet**3 *eta)
    
    return masslosstime



   
    
    
    
    
    
    