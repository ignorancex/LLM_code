import numpy as np
import scipy.integrate as it
import scipy.interpolate as ip
from scipy.special import gamma

# The main purpose of this module is to get a CDM (dark matter only) power spectrum
# that includes the effect of baryons not following the dark matter on small scales
# For this we use (same as Delos & White (2022)) a CLASS power spectrum (http://www.class-code.net/)
# and the (matched) analytic descriptions from Hu & Suigyama (1996)
# Basic usage is like this:
# myclass = power.get_class(z=zref, kmax=2.*kmatch, cosm=cosmology)
# Dk = power.dimless_cdmpower_all_scales(k, myclass, z=..., A_s=..., ...)
        
def dimless_power_from_transfer(k, T, ns=0.96, A_s = 2.100549e-09, k_pivot = 0.05):
    pk = T**2 * A_s * (k / k_pivot)**ns *k**-4  
    return pk *k**3 / (2.*np.pi**2)

from scipy.special import gamma, digamma, hyp2f1
def hu_sugiyama_deltac(k, a, phi0=1., fnu=0.605, omega_cdm=0.26067, omega_m=0.30964, Tcmb=2.7255, hubble=0.678):
    """
    Hu & Suigyama (1996)
    https://arxiv.org/pdf/astro-ph/9510117.pdf
    
    fnu : neutrino fracion
    """
    
    def alpha(i):
        pass
    
    H0 = 100.*hubble
    omega_0 = omega_m
    theta27 = Tcmb / 2.7 # before (1)
    
    aeq = 2.35e-5 * (omega_0*hubble**2)**-1 * (1 - fnu)**-1 * theta27**4 # Close to (1)
    #keq = (2. * omega_0 * H0**2 / aeq)**0.5  # Close to B2
    keq = 9.67e-2 * omega_0 * hubble**2 * (1 - fnu)**0.5 * theta27**-2  # 1/Mpc
    
    aH = aeq * (1. + np.sqrt(1. + 8.*(k/keq)**2)) / (4.*(k/keq)**2) # B13
    
    I1 = 9.11 * (1. + 0.128*fnu + 0.029*fnu**2)   # B14
    I2 = 0.594 * (1. - 0.631*fnu + 0.284*fnu**2)

    y = a/aeq
    
    # Equation D5
    alpha1 = (1. - np.sqrt(1. + 24. * omega_cdm / omega_0)) / 4.
    alpha2 = (1. + np.sqrt(1. + 24. * omega_cdm / omega_0)) / 4.
        
    # Equation D4
    def U(alphai):
        F = hyp2f1
        return (1. + y)**(-alphai) * F(alphai, alphai + 0.5, 2.*alphai + 0.5, 1./(1+y))
    U1 = U(alpha1)
    U2 = U(alpha2)
    
    # Equation D10:
    def A(alpha1, alpha2, I2):
        psi = digamma
        A  = - (gamma(alpha1)*gamma(alpha1 + 0.5))
        A /= gamma(2*alpha1 + 0.5) * (psi(alpha1) + psi(alpha1 + 0.5) - psi(alpha2) - psi(alpha2 + 0.5))
        A *= (np.log(I2 * aeq / aH) + 2.*psi(1.) - psi(alpha2) - psi(alpha2 + 0.5))
        return A
    
    A1 = A(alpha1, alpha2, I2)
    A2 = A(alpha2, alpha1, I1)
    
    #phi0
    return I1*phi0 * (A1*U1 + A2*U2)

def class_power_anyspecies(cosmo_class, z=30.6, comp="cdm"):
    Tk = cosmo_class.get_transfer(z)
    k = Tk["k (h/Mpc)"] * cosmo_class.h()
    
    pkmatter = [cosmo_class.pk(ki, z) for ki in k[:-1]]
    
    if comp=="cdm":
        return k[:-1], pkmatter * (Tk["d_cdm"]/Tk["d_m"])[:-1]**2
    elif comp=="matter":
        return k[:-1], pkmatter
    elif comp=="baryons":
        return k[:-1], pkmatter * (Tk["d_b"]/Tk["d_m"])[:-1]**2
    else:
        assert 0

def dimless_cdmpower_all_scales(k, cosmo_class, z=30.6, kmatch=2e3, ksel=None, A_s=2.100549e-09):
    """Returns the dimensionless power spectrum of the combination of 
    CLASS spectrum on resolved scales
    + Hu & Suigyama (1996) spectrum on unresolved scales"""
    cosmpar = dict(omega_m=cosmo_class.Omega0_m(), omega_cdm=cosmo_class.Omega0_cdm(), hubble=cosmo_class.h(), Tcmb=cosmo_class.T_cmb())
    
    if ksel is None:
        ksel = kmatch
    
    sel_class = k < ksel
    sel_ana = k >= ksel
    
    power = np.zeros_like(k)
    
    k_class, pk_class = class_power_anyspecies(cosmo_class, comp="cdm", z=z)
    
    power[sel_class] = np.interp(k[sel_class], k_class, pk_class * k_class**3 / (2.*np.pi**2))
    
    cosmpar = dict(omega_m=cosmo_class.Omega0_m(), omega_cdm=cosmo_class.Omega0_cdm(), hubble=cosmo_class.h(), Tcmb=cosmo_class.T_cmb())
    def power_ana(k):
        Tk_ana = hu_sugiyama_deltac(k, 1./(1.+z), **cosmpar, fnu=cosmo_class.Neff() / (cosmo_class.Neff() + 2.))
        return dimless_power_from_transfer(k, Tk_ana, A_s=A_s, ns=cosmo_class.n_s())
    
    amplitude_ratio = (pk_class * k_class**3 / (2.*np.pi**2))[-1] / power_ana(k_class[-1])
    
    
    power[sel_ana] = power_ana(k[sel_ana]) * amplitude_ratio
    
    return power



cosm_planck_18 = {'omega_cdm': 0.26067, 'omega_baryon': 0.04897, 'hubble': 0.6766, 'ns': 0.9665, 'tau': 0.0561, 'A_s': 2.105e-09, 'neutrino_mass': 0.0}
def get_class(z=30.6, kmax=1e4, cosm=cosm_planck_18):
    """Creates an instace of classy.Class.
    Requires that classy is installed"""
    
    import classy
    cosmo_class = classy.Class()
    par = {'output': "mPk,mTk,vTk", 
           'z_max_pk': z,
           'P_k_max_h/Mpc': kmax,
           'A_s': cosm['A_s'], 
           'Omega_cdm': cosm['omega_cdm'],
           'h': cosm['hubble'],
           'Omega_b': cosm['omega_baryon'],
           'Omega_cdm': cosm['omega_cdm'],
           'n_s': cosm['ns']
          }
    
    cosmo_class.set(par)
    cosmo_class.compute()
    return cosmo_class