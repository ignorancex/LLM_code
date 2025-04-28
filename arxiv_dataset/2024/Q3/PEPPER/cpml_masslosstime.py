# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:19:35 2022

@author: madir
"""

import numpy as np
import constants as cs
import planetstructure as ps
from scipy.optimize import minimize_scalar
import pdb

def calc_eta():
    eta=0.1
    return eta 

def calc_max_masslosstime_rocky(system, Rcore, Mcore, Teq, Xiron, Tkh_Myr):
    '''Returns the maximized mass loss timescale for the rocky planet under CPML
    
    Parameters: 
        system- Planetary System object
        Rcore - rocky planet radius in Earth radii
        Mcore - rocky planet core mass in Earth masses
        Teq - rocky planet equilibrium temperature in K
        Xiron - iron mass fraction
        Tkh_Myr - cooling timescale in Myr
        '''
   
    DR_rcb_min = 0.1 * cs.Rearth2cm(Rcore)
    DR_rcb_max = 5 * cs.Rearth2cm(Rcore)
    #find the value of DR_rcb that maximizes the mass loss timescale
    result=minimize_scalar(escape_lim_masslosstime_objective, bounds=(DR_rcb_min, DR_rcb_max), args=(Rcore, Mcore, Teq, Xiron, Tkh_Myr), method='bounded')
    
    if (result.success):
        
        DR_rcbmaximized = result.x
        Rrcb = DR_rcbmaximized + cs.Rearth2cm(Rcore)
        X_max, _, Rplanet = ps.calc_Rplanet(cs.Rearth2cm(Rcore), DR_rcbmaximized, Tkh_Myr, Teq, Xiron)
        rho_rcb = ps.calc_rho_rcb(cs.Rearth2cm(Rcore), DR_rcbmaximized, X_max, Tkh_Myr, Teq, Xiron)
        eta = calc_eta()
        mltime_escape_lim= escape_lim_masslosstime_eq(Mcore, Teq, rho_rcb, Rrcb, X_max, eta)
        system.planetRocky.scaled_masslosstime_max_CPML = mltime_escape_lim
        system.planetRocky.eta= eta
        return mltime_escape_lim
    
    else: 
        print('error')
        return -1
        
    
def escape_lim_masslosstime_objective(DR_rcb, Rcore, Mcore, Teq, Xiron, Tkh_Myr): 
    '''Returns the mass loss timescale that we wish to maximize, only considering the mass loss rate dictated by
    the escape rate of molecules at the Bondi radius
    
    Parameters: 
        DR_rcb- width of the envelope to the RCB in cm
        Rcore - rocky planet radius in Earth radii
        Mcore - rocky planet core mass in Earth masses
        Teq - rocky planet equilibrium temperature in K
        Xiron - iron mass fraction
        Tkh_Myr - cooling timescale in Myr
        
        '''
    
    
    c_atmos_sq = cs.kb*Teq/ cs.mu
    R_sonic = cs.G * cs.Mearth2g(Mcore) / (2*c_atmos_sq)
    
    Rrcb = DR_rcb + cs.Rearth2cm(Rcore)
    X, _, Rplanet = ps.calc_Rplanet(cs.Rearth2cm(Rcore), DR_rcb, Tkh_Myr, Teq, Xiron)
    M_atm=X*Mcore 
    
    eta = calc_eta()
    
    rho_rcb = ps.calc_rho_rcb(cs.Rearth2cm(Rcore), DR_rcb, X, Tkh_Myr, Teq, Xiron)
    escape_lim_masslossrate = 4*np.pi*R_sonic**2*np.sqrt(c_atmos_sq)*rho_rcb*np.exp((-cs.G*cs.Mearth2g(Mcore))/(c_atmos_sq*Rrcb))
    mltime_escape_lim = M_atm/(escape_lim_masslossrate*eta)
    
    
    return 1/mltime_escape_lim

def escape_lim_masslosstime_eq(Mcore,Teq,rho_rcb, Rrcb, X, eta):
    '''Returns the mass loss timescale under core-powered mass loss, only considering the mass loss rate dictated by
    the escape rate of molecules at the Bondi radius
    
    Parameters: 
        Mcore- Core mass in earth masses
        Teq- planet equilibrium temperature in K
        rho_rcb- density at the RCB in cgs units
        Rrcb- radius to the RCB in cm
        X - envelope mass fraction
        eta - atmospheric mass loss efficiency in cgs units
        '''
    
    c_atmos_sq = cs.kb*Teq/ cs.mu
    R_sonic = cs.G * cs.Mearth2g(Mcore) / (2*c_atmos_sq)
    M_atm=X*Mcore
    escape_lim_masslossrate = 4*np.pi*R_sonic**2*np.sqrt(c_atmos_sq)*rho_rcb*np.exp((-cs.G*cs.Mearth2g(Mcore))/(c_atmos_sq*Rrcb))
    mltime_escape_lim = M_atm/(escape_lim_masslossrate*eta)
    return mltime_escape_lim 

