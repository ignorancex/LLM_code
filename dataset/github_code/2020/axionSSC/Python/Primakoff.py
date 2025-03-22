# Primakoff Production

import numpy as np

# Various Conversions
keV2erg = 1.60218e-9 # keV -> erg
GeV2keV = 1e6 # GeV -> keV
Rsol2cm = 6.957e10 # Rsol -> cm
K2keV = 0.086/1e6 # Kelvin -> keV
m_u = .931*GeV2keV # Atomic mass unit
g2GeV = 5.61e23 # gram -> GeV
cm2GeVinv = 1/1.98e-14 # cm -> GeV^-1
GeV2sinv = 1.52e24 # GeV -> s^-1
alpha_EM = 1./137
keV2output_units = keV2erg*GeV2sinv/GeV2keV # converts keV to erg/s/keV

def Primakoff(E,g_agg,R,dR,rho,T,X_H,Y_He,Z_C,Z_O,Z_Ne):
    '''
    Computes the luminosity spectrum [erg/s/keV] of the Primakoff axion production mechanism
    in a nonrelativistic nondegenerate plasma.
    
    E, Photon/Axion Energy [keV]
    g_agg, Axion-photon coupling [GeV^-1]
    R, Distance from Center of Star [R_sol]
    dR, Width of Zone [cm]
    rho, Density [g/cm^3]
    T, Temperature [Kelvin]
    X_H, Hydrogen abundance
    Y_He, Helium abundance
    Z_C, Carbon abundance
    Z_O, Oxygen abundance
    Z_Ne, Neon abundance
    Further metals are not abundant in our stars.
    '''
    
    # Convert all inputs to keV^n
    gaggkeVinv = g_agg/GeV2keV
    RkeVinv = R*Rsol2cm*cm2GeVinv/GeV2keV
    dRkeVinv = dR*cm2GeVinv/GeV2keV
    TkeV = T*K2keV
    rho_keV4 = rho*g2GeV/cm2GeVinv**3*GeV2keV**4

    # Relevant parameters
    Volume = 4*np.pi*RkeVinv**2*dRkeVinv # Volume of Slice
    nhat = (2*X_H+(3*Y_He+7*Z_C+9*Z_O+11*Z_Ne)/2.)*rho_keV4/m_u # Weighted number density
    xi = np.sqrt(np.pi*alpha_EM*nhat/TkeV**3)
    
    # Energy_Emission_Per_Unit_Volume is in keV^4 (i.e. erg/cm^3/s/keV)
    # See Eq. S3
    Energy_Emission_Per_Unit_Volume = gaggkeVinv**2/(8*np.pi**3) * xi**2*TkeV**3*E/(np.exp(E/TkeV)-1) *\
    ((E**2+(xi*TkeV)**2)*np.log(1+(E/(xi*TkeV))**2)-E**2)
    
    # Energy_Emission_Per_Unit_Volume*Volume is in keV
    Erg_s_keV = Energy_Emission_Per_Unit_Volume*Volume*keV2output_units
  
    return Erg_s_keV