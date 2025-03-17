import numpy as np
from . import bns_fits

# Units CGS
Lplanck = 3.628504984913064e59 # Planck luminosity c^5/G [erg/s]
clight  = 2.99792458e+10 # [cm/s]
Msun    = 1.98892e+33 # [g]
G       = 6.674e-8  # [cm^3/(g s^2)]
Energy_cgs = Msun*clight**2 
AngMom_cgs = G * Msun**2/clight

def egw(m1,m2,l1,l2):
    """
    Energy radiated in GW (including pre- and post- merger) by merging binary neutron stars.
    
    Input:
    ------
    - m1,m2: masses in Msun (m1>=m2)
    - la,lb: dimensionless tidal deformabilities
    
    Returns:
    -------
    - E_gw: radiated energy *in Msun*
    """
    
    # mass ratio
    q = m1/m2
    
    # Symmetric mass ratio
    nu = bns_fits.q_to_nu(q)
    
    # Compute the kappa's
    kappa_A = bns_fits.lam_q_to_kappa(l1, q)  
    kappa_B = bns_fits.lam_q_to_kappa(l2, 1./q)
        
    # Compute kappaT2 and kappa effective for luminosity fit
    kappaT2 = kappa_A + kappa_B
    kapelum = bns_fits.kappalum(kappa_A, q) + bns_fits.kappalum(kappa_B, 1./q)
    
    # Eval fits
    e_pm = bns_fits.e_pm_fit(kappaT2)
    e_mrg = bns_fits.e_mrg_fit(kappaT2, nu)

    e_gw = e_mrg + e_pm

    # Rescale
    E_gw     = e_gw*nu*(m1+m2)
    
    return E_gw



