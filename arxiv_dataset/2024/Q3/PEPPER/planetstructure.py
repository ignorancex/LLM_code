# -*- coding: utf-8 -*-

import time

import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from bisect import bisect_left
import constants as cs
from scipy.optimize import fsolve


def weights(Xiron):
    '''Returns the coefficients needed to find the weighted average of two mass/radius columns
       
           Parameters: 
               Xiron - iron mass fraction'''
    
    Xirons = np.arange(0, 1.025, 0.025)
    #returns the position within the list at which the given Xiron value would be inserted
    #if Xiron is equal to value in list it returns the index of the matching value
    pos = bisect_left(Xirons, Xiron)
    diff1 = abs(Xirons[pos]-Xiron)
    diff2 = abs(Xirons[pos-1]-Xiron)
    
    #if the given Xiron is equal to a value in the list the coefficient is 1
    if diff1 == 0: 
        coeff1 = 0
        coeff2 = 1
    else: 
        coeff1 = diff1/.025
        coeff2 = diff2/.025
  
    return pos, coeff1, coeff2

    
def table_interpolation(Xiron):
    pos, coeff1, coeff2 = weights(Xiron)
    
    f1 = interpolate.interp1d(cs.masstable[:,pos-1], cs.radtable[:,pos-1], bounds_error=False)
    f2 = interpolate.interp1d(cs.masstable[:,pos], cs.radtable[:,pos], bounds_error=False)
    massarr=np.logspace(-2,2,500)
    radtable1=f1(massarr)
    radtable2=f2(massarr)
    
    if coeff1 == 0: 
        radii = coeff2*radtable2
    else: 
        radii = coeff1*radtable1 + coeff2*radtable2
        
    return radii, massarr
    
def Rcore_to_Mcore(Rcore, Xiron):
    '''Returns the mass of a planet in Earth masses. 
    Mass-radius relationships derived from Li Zeng's models of Fe-MgSiO3 planets
    
    Parameters: 
        Rcore - planet core radius in Earth radii
        Xiron - iron mass fraction '''
    
    radii, massarr = table_interpolation(Xiron)
    
        
    #use linear interpolation to find the mass value for a given radius value as a function of Xiron
    massfunc = interpolate.interp1d(radii, massarr, bounds_error=False)
    mass = massfunc(Rcore)
  
    return mass
    

def Mcore_to_Rcore(Mcore, Xiron):
    '''Returns the radius of a planet in Earth radii
    Mass-radius relationships derived from Li Zeng's models of Fe-MgSiO3 planets
    
    Parameters: 
        Mcore - planet mass in Earth masses
        Xiron - iron mass fraction'''
    
    pos, coeff1, coeff2 = weights(Xiron)
    
    f1 = interpolate.interp1d(cs.radtable[:,pos-1], cs.masstable[:,pos-1], bounds_error=False)
    f2 = interpolate.interp1d(cs.radtable[:,pos], cs.masstable[:,pos], bounds_error=False)
   
    radarr=np.linspace(.1,10,500)
    masstable1 = f1(radarr)
    masstable2 = f2(radarr)
    
    if coeff1 == 0: 
        masses = coeff2*masstable2
    else: 
        masses = coeff1*masstable1 + coeff2*masstable2
    
    #use linear interpolation to find the radius value for a given mass value as a function of Xiron 
    radiusfunc = interpolate.interp1d(masses, radarr, bounds_error=False)
    radius = radiusfunc(Mcore)
    
    return radius
    
##Envelope structure 

def integrand1(x):

    return x*(1/x-1)**(1/(cs.gamma-1))

def integrand2(x):

    return x**2*(1/x-1)**(1/(cs.gamma-1))

#integrates eq 5 in Owen & Wu from Rc/Rrcb to 1 

def get_I1(Rcore, Rrcb):
    
 
    I1 = quad(integrand1, Rcore/Rrcb, 1)[0]
    return I1

def get_I2(Rcore, Rrcb):
    #make input args same units
   
    #print (Rcore/Rrcb, Rcore, Rrcb)
    I2 = quad(integrand2, Rcore/Rrcb, 1)[0]
   
    
    return I2 

def get_I2_I1(Rcore, Rrcb):
    
    
    ratio = get_I2( Rcore, Rrcb)/get_I1( Rcore, Rrcb)
    return ratio 

def calc_X_adiabatic(Rcore, DR_rcb, Tkh_Myr, Teq, Xiron): 
    ''''
    Calculates the envelope mass fraction X
    
    Parameters: 
        Rcore - planet core radius in cm
        DR_rcb - the width of the adiabatic portion of the atmosphere in cm  
        Tkh_Myr - cooling timescale in Myr
        Teq- planet equilibrium temp in K
        Xiron - envelope mass fraction'''
    
   
      
    Rrcb = DR_rcb + Rcore
   
    #sound speed of atmosphere
    c_atmos_sq = cs.kb * Teq / cs.mu
    
    adiabatic_grad = (cs.gamma-1)/cs.gamma
    
    #average core density 
    rho_core = cs.Mearth2g(Rcore_to_Mcore(cs.cm2Rearth(Rcore), Xiron)) / ((4/3)* np.pi* Rcore**3)
    
    
    #using eq 13 from Owen & Wu 2017, factor out X
    
    rho_rcb_factoroutx = calc_rho_rcb( Rcore, DR_rcb, 1, Tkh_Myr, Teq, Xiron)   
    
    
    #X=Menv/Mcore
    #by factoring out X from eq 13 we can replace the planet density in eq 4 with rho_rcb times x**(1+alpha)
    #using eq 4 from Owen & Wu, divide out Mcore, resulting in an expression of Menv/Mcore 
    Menv_Mcore = 3 * (Rrcb/Rcore)**3 * (rho_rcb_factoroutx/rho_core) * (adiabatic_grad *\
                    (cs.G * cs.Mearth2g(Rcore_to_Mcore(cs.cm2Rearth(Rcore), Xiron))) / (c_atmos_sq * Rrcb))**(1/(cs.gamma-1)) * get_I2(Rcore, Rrcb)
   
    #because Menv/Mcore=X we can solve for X 
    X = Menv_Mcore**(1 / (1 + (1 / (1+cs.alpha))))
    

    return X
    


def calc_rho_rcb(Rcore, DR_rcb, X, Tkh_Myr, Teq, Xiron): 
    '''Calculate the density at the radiative-convective boundary
    
    Parameters: 
        Rcore - planet radius in cm
        DR_rcb- the width of the adiabatic portion of the atmosphere in cm 
        X - envelope mass fraction
       Tkh_Myr - cooling timescale in Myr
       Teq - equilibrium temperature of the planet in K
       Xiron - iron mass fraction'''
    
    
    #height of the rcb
    Rrcb = DR_rcb + Rcore    
    
    I2_I1 = get_I2_I1(Rcore, Rrcb)
    
    Tkh_sec= cs.Myr2sec(Tkh_Myr)
    
    
    rho_rcb = (cs.mu/cs.kb) * (I2_I1 * 64 * np.pi * cs.sigma * Teq**(3-cs.alpha-cs.beta) * Rrcb * Tkh_sec / 
                            (3 * cs.kappa0 * cs.Mearth2g(Rcore_to_Mcore(cs.cm2Rearth(Rcore), Xiron))* X))**(1/(1+cs.alpha))
    
    return rho_rcb
 
    
def calc_Rplanet(Rcore, DR_rcb, Tkh_Myr, Teq, Xiron):
     '''Returns the radius of the planet over the radius to the rcb and returns the radius of the planet 
     Parameters: 
         Rcore - planet radius in cm
         DR_rcb- the width of the adiabatic portion of the atmosphere in cm
         Tkh_Myr- cooling timescale in Myr
         Teq - equilibrium temperature of the planet in K
         Xiron - iron mass fraction'''
     
     #calculate the denisty at the photosphere
     #pressure at the rcb is used as an approx for photospheric pressure 
     #because the isothermal layer is assumed to be thin
     
     #height of the rcb
     Rrcb = DR_rcb + Rcore
     
     Mcore = Rcore_to_Mcore(cs.cm2Rearth(Rcore), Xiron)
     
     X=calc_X_adiabatic( Rcore, DR_rcb, Tkh_Myr, Teq, Xiron)
     
     #pressure at the photosphere
     
     _, rho_phot, H = calc_photo_pressure( Rrcb, Mcore, Teq)
    
    
     #f=Rp/Rrcb
     rho_rcb = calc_rho_rcb(Rcore, DR_rcb, X, Tkh_Myr, Teq, Xiron)  
     Rp_Rrcb = 1 + (H/Rrcb) * np.log(rho_rcb/rho_phot)
     
     if Rp_Rrcb < 1: 
         print('Error: rho_rcb < rho_phot')
         
     else: 
        Rplanet = Rp_Rrcb * Rrcb
      
        
        return X, Rp_Rrcb, Rplanet
    
 
def calc_X_isothermal(Rp, Mcore, Teq, Xiron):
    '''Returns the envelope mass fraction and planet radius in cgs units. This function is only used if the adiabatic layer is less than 1 scale height
    
    Parameters: 
        Rp - planet radius in cm
        Mcore - planet mass in Earth masses
        Teq - equilibrium temp in K
        Xiron - iron mass fraction '''
  
    Rcore = cs.Rearth2cm(Mcore_to_Rcore(Mcore, Xiron))
    
    _, rho_phot, H = calc_photo_pressure(Rp, Mcore, Teq)
    
    #find mass in radiative layer
    rho_atmos = rho_phot * np.exp(Rp/H * (Rp/Rcore-1))
    
    Menv = rho_atmos * 4 * np.pi * Rcore**2 * H
    
    X = Menv / (cs.Mearth2g(Mcore))
    
    return X, Rp/Rcore, Rp
    
    
def calc_photo_pressure( R, Mcore, Teq): 
    '''Returns the photospheric pressure, density of the photosphere, and scale height in cgs units
    
    Parameters: 
        R - radius in cm, the user will input Rrcb or Rp depending on if you wish to calculate the envelope mass fraction
        of the adiabatic or isothermal layer
        Mcore - mass in Earth masses
        Teq - equilibrium temp in K'''
    
    #here you will either insert Rrcb or Rp depending on if you need the adiabatic or isothermal layer 
    pressure_phot = (2/3 * (cs.G*cs.Mearth2g(Mcore)/(R**2.*cs.kappa0*Teq**cs.beta)))**(1/(1+cs.alpha))
    
    rho_phot_calc = (cs.mu/cs.kb) * pressure_phot / Teq
    
    H = cs.kb * Teq * R** 2 / ( cs.mu * cs.G * cs.Mearth2g(Mcore))
    
    return pressure_phot, rho_phot_calc, H
   
  
def Rplanet_solver(Rp, Mcore, planet_age, Teq, Xiron):
    '''Returns the envelope mass fraction and planet radius at the current age of the system
    
    Parameters: 
        Rp - current planet radius in Earth radii
        Mcore - planet mass in Earth masses
        planet_age - current age of the system in Myr
        Teq - equilibrium temp in K
        Xiron - iron mass fracttion'''
        
    #this will only get called when evaluating the gaseous planet 
    #we will then make sure that the Rplanet returned converges with the current Rp input by the user 
     
    Rcore = Mcore_to_Rcore(Mcore, Xiron)
    
    if (Rp < Rcore):
        print('failed to find planet radius at current age of the system')
        return -1 
    
    Rcore = cs.Rearth2cm(Rcore)
    
    lg_DR_rcb_guess = np.log10(cs.Rearth2cm(Rp) - Rcore)
    
    
    #find the DR_rcb where the calculated Rplanet is equal to the observed Rplanet 
    #use log values to make the solver behave better
   
    lg_DR_rcb_sol = fsolve(Rplanet_solver_eq, lg_DR_rcb_guess, args=(Rp, Mcore, Teq, planet_age, Xiron))
    
    H = cs.kb * Teq * cs.Rearth2cm(Rp)** 2 / ( cs.mu * cs.G * cs.Mearth2g(Mcore))
    
    DR_rcb_sol = 10**lg_DR_rcb_sol
    #print(DR_rcb_sol)
    if DR_rcb_sol > H: 
        
        #X = calc_X_adiabatic(Rcore, DR_rcb_sol, planet_age, Teq, Xiron)
        X, Rp_Rrcb, Rplanet = calc_Rplanet(Rcore, DR_rcb_sol, planet_age, Teq, Xiron)
        
        
        return X, Rp_Rrcb, Rplanet
    
    else: 
        
        X, Rp_Rcore, Rplanet = calc_X_isothermal( cs.Rearth2cm(Rp), Mcore, Teq, Xiron)
        return X, Rp_Rcore, Rplanet
      
def Rplanet_solver_eq(lg_DR_rcb, Rp, Mcore, Teq, planet_age, Xiron): 
    '''Returns the difference between the current radius of the enveloped planet and the calculated radius for a given mass estimate
    
    Parameters: 
        lg_DR_rcb - log of the width of the rcb in cm
        Rp - current planet radius in Earth radii
        Mcore - planet mass in Earth masses
        planet_age - current age of the system in Myr
        Teq - equilibrium temp in K
        Xiron - iron mass fracttion'''
        
  
    #evaluate the envelope mass fraction and atmosphere structure for this mass/radius guess
    Rcore = cs.Rearth2cm(Mcore_to_Rcore(Mcore, Xiron))
   
    DR_rcb = 10**lg_DR_rcb
   
    X, Rp_Rrcb, Rplanet = calc_Rplanet(Rcore, DR_rcb, planet_age, Teq, Xiron)
   
    
    return cs.Rearth2cm(Rp) - Rplanet


def solve_envelope_structure(X, Mcore, Tkh_Myr, Teq, Xiron): 
    '''given an envelope mass fraction this finds the radius of the rcb and returns a value for 
    the radius of the planet at a given time
    
    Parameters: 
        X - envelope mass fraction
        Mcore - mass of the enveloped planet in Earth masses
        Tkh_Myr - cooling timescale in Myr
        Teq - equilibrium temp of the enveloped in K
        Xiron - Iron mass fraction'''
    
    
    #find the core radius using the given mass value 
    Rcore = cs.Rearth2cm(Mcore_to_Rcore(Mcore, Xiron))
    
    #equation from Owen & Wu 2017
    DR_rcb_guess = 2 * Rcore * (X/0.027)**(1/1.31) * (Mcore/5)**(-0.17)

    lg_DR_rcb_guess = np.log10(DR_rcb_guess)
    
    #optimization function behaves better using log inputs
    lg_DR_rcb_sol = fsolve(Rrcb_solver_eq, lg_DR_rcb_guess, args = (Mcore, Xiron, X, Tkh_Myr, Teq))
    
    DR_rcb_sol = 10**lg_DR_rcb_sol
    
    Rrcb_sol = DR_rcb_sol + Rcore
    
    _, rho_phot, H = calc_photo_pressure( Rrcb_sol, Mcore, Teq)
   
    rho_rcb = calc_rho_rcb(Rcore, DR_rcb_sol, X, Tkh_Myr, Teq, Xiron)  
    
    Rp_Rrcb = 1 + (H/Rrcb_sol) * np.log(rho_rcb/rho_phot)
    
    Rplanet = Rp_Rrcb * Rrcb_sol
    
    
    return Rplanet, DR_rcb_sol
    
    
    
def Rrcb_solver_eq( lg_DR_rcb, Mcore, Xiron, X, Tkh_Myr, Teq): 
    '''Analytically computes the envelope mass fraction and returns the difference between this X value 
    and the value returned by Rplanet_solver. The difference should be ~0.
    
    Parameters: 
        lg_DR_rcb - log of the width of the rcb
        Mcore - mass of the enveloped planet in Earth masses
        Xiron - iron mass fraction
        X - envelope mass fraction
        Tkh_Myr- cooling timescale in Myr
        Teq - equilibrium temp of the enveloped planet in K'''
    
    Rcore = cs.Rearth2cm(Mcore_to_Rcore(Mcore, Xiron))
    
    DR_rcb = 10**lg_DR_rcb
    
    Rrcb = DR_rcb + Rcore
    
    rho_core = cs.Mearth2g(Mcore) / (4/3 * np.pi * Rcore**3)
    
    rho_rcb = calc_rho_rcb(Rcore, DR_rcb, X, Tkh_Myr, Teq, Xiron)
    
    c_atmos_sq = cs.kb * Teq / cs.mu
    
    adiabatic_grad = (cs.gamma-1)/cs.gamma
    
    X_analytic = 3 * (Rrcb/Rcore)**3 * (rho_rcb/rho_core) * (adiabatic_grad *\
                    (cs.G * cs.Mearth2g(Rcore_to_Mcore(cs.cm2Rearth(Rcore), Xiron))) / (c_atmos_sq * Rrcb))**(1/(cs.gamma-1)) * get_I2(Rcore, Rrcb)
    
    return X_analytic - X





    
    
    
    
    
    
     
      

    
    
    
    




    
    

    
    

    
    