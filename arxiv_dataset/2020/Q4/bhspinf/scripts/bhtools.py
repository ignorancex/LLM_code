import numpy as np
import constants as c

#====================================================================================================
# Convert a black hole spin (a [-]) to an ISCO location (risco [Rg])
# NOTE: Input "a" must be a single value and -1 <= a <= 1
def spin2risco(a):
    
    # Check that -1 <= a <= 1
    if not np.logical_and(a>=-1.0, a<=1.0):
        print("\nWARNING: Input black hole spin must satisfy -1 <= a <= 1\n")
        if   (a >=  1.0): risco = 1.0; return risco
        elif (a <= -1.0): risco = 9.0; return risco
        else:             quit()
        
    # Calculate risco
    Z1 = 1.0 + (1.0 - a**2.0)**(1.0/3.0) * ((1.0 + a)**(1.0/3.0) + (1.0 - a)**(1.0/3.0))
    Z2 = (3.0 * a**2.0 + Z1**2.0)**(0.5)
    if (a >= 0.0):
        risco = 3.0 + Z2 - ((3.0 - Z1)*(3.0 + Z1 + 2.0*Z2))**(0.5)
    if (a < 0.0):
        risco = 3.0 + Z2 + ((3.0 - Z1)*(3.0 + Z1 + 2.0*Z2))**(0.5)
    return risco

#====================================================================================================
# For an input black hole spin parameter (a [-])...
# xalculate the Eddington accretion efficiency, eta --> Mdot = eta L_Edd / c^2
def radiative_efficiency(a):
    risco = spin2risco(a=a)
    eta   = 1.0 - np.sqrt(1.0 - 2.0 / (3.0 * risco))
    return eta

#====================================================================================================
# For an input black hole mass (M [Msun])...
# calculate the Eddington luminosity,  L_Edd = 4 pi G M m_p c / sigma_T [erg s^-1]
def Eddington_luminosity(M):
    L_Edd = 4.0 * np.pi * c.G * M * c.sun2g * c.mp * c.c / c.sigmaT
    return L_Edd

#====================================================================================================
