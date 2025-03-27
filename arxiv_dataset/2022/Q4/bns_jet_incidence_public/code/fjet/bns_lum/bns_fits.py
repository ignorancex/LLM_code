#!/usr/bin/env python

""" 
Fit models for BNS peak luminosity, GW energy, and angular momentum based on numerical relativity simulations
"""

__author__      = "F.Zappa (Jena U), S.Bernuzzi (Jena U), D.Radice (Princeton & IAS), A.Perego (INFN & Parma U), T.Dietrich (AEI)"
__copyright__   = "Copyright 2017/18"

import numpy as np
from scipy.special import factorial2

# ---------- fit coefficients -------------------------------------------
coefs_e_pm_fit_pow = [ 2.44358614, -0.01892881]

slope_e_pm_extrap = -5.12643669729e-05
offset_e_pm_extrap = 0.038079362826

#~ coefs_Lpeak_fit_q_nr = [ 0.0162958980711,  7.87825615e-04,  -2.09234890e-07,  2.09277467e-02 ]
coefs_Lpeak_fit_q_nr = [ 0.021782890899735566, 0.00052426885345038126, -9.3658539544098073e-08, 0.027742691256389923 ]

coef0_Lpeak_fit_q_extrap = 0.021782890899735566
high_kappaL_Lpeak_fit_q_extrap = 3581.76666667
slope_Lpeak_fit_q_extrap = -6.08468023236e-06
offset_Lpeak_fit_q_extrap = 0.0167245171163

coefs_e_j_final_fit = [ 0.94665103, -0.43994502, 0.05334342 ]

coefs_j_e_final_fit = [ 4.39506775, -17.211734 , 38.549884 ]

e0_e_mrg_fit = 0.12
a_e_mrg_fit = 1.2e3
coefs_e_mrg_fit = [ 5.09166459e-02, 6.44073902e-05, 9.53252086e-02, 2.64027080e-04 ]

j0_j_mrg_fit = 2.8
a_j_mrg_fit = 1.2e3
coefs_j_mrg_fit = [ 0.07831028, 0.00019308, 0.06631369, 0.00012563 ]
# ------------------------------------------------------------------------

def lam_q_to_kappa(lam, q):
    """ Compute kappa_A parameter from Lambda_A parameter and mass ratio """
    return 3*(q**4)/((1+q)**5)*lam

def barlamdel_to_kappal(q, barlamAl, barlamBl, ell):
    """$\kappa^{A,B}_\ell(\bar{\lambda}_\ell)$
    Assume $q=M_A/M_B>=1$
    """
    XA = q/(1.+q);
    XB = 1. - XA;
    f2l1 = factorial2(2*ell-1);
    p = 2*ell + 1;
    kappaAl = f2l1 * barlamAl * XA**p / q; 
    kappaBl = f2l1 * barlamBl * XB**p * q; 
    #kappaTl = kappaAl + kappaBl;
    return  kappaAl, kappaBl

def q_to_nu(q):
        """ Compute sym mass ratio from mass ratio, assume q>=1 """
        if np.any(q<1.):
                raise ValueError("q must be >=1")
        return q/((1.+q)**2)

def kappalum(kappa, q):
    """ Compute effective kappa_A for lluminosity fit from kappa_A and mass ratio """
    return 2 * (3 + q) * kappa

def e_pm_fit_const(ktid):
    """ Fit energy post merger, prompt collapse 
    Valid for ktid < 63
    """
    cpcol = 0.018
    return np.full_like(ktid, cpcol, dtype=np.double)

def e_pm_fit_avg6373(ktid):
    """ Fit energy post merger, NS remnant 
        Average value in region 63 < ktid < 73 """
    avg = (0.018 + 0.1)/2.
    return np.full_like(ktid, avg, dtype=np.double)

def e_pm_fit_pow(ktid):
    """ Fit energy post merger, NS remnant 
        Valid for ktid > 73 """
    return coefs_e_pm_fit_pow[0] * ktid**(-7./10.)+ coefs_e_pm_fit_pow[1]

def e_pm_extrap(ktid):
    """ Extrapolate the fit e_pm, 
        Valid in 457.9 < ktid < 742.8 """
    return slope_e_pm_extrap*ktid + offset_e_pm_extrap
    
def e_pm_fit(ktid):
    """ Fit energy post merger, note that this is EGW/(M nu) """
    if np.any(ktid < 0.):
        raise ValueError("Invalid values of tidal parameter!")
    conds = [ktid <= 63.,
             (ktid > 63.) & (ktid < 73.),
             (ktid >= 73.) & (ktid <= 457.9),
             (ktid > 457.9) & (ktid <= 742.803726537),
             ktid > 742.803726537]
    funcs = [e_pm_fit_const,
             e_pm_fit_avg6373, #lambda x: np.full_like(x,np.nan,dtype=np.double),
             e_pm_fit_pow,
             e_pm_extrap,
             lambda x: np.full_like(x,0.,dtype=np.double)]
    return np.piecewise(ktid, conds, funcs)

def Lpeak_fit_q_nr(klum):
    """ Fit Lpeak, note this is Lpeak/(nu^2) * q^2 
        Valid for klum < 3581.76666667 """
    return coefs_Lpeak_fit_q_nr[0] * (1 + coefs_Lpeak_fit_q_nr[1] * klum + coefs_Lpeak_fit_q_nr[2] * klum**2)/(1 + coefs_Lpeak_fit_q_nr[3] * klum)

def Lpeak_fit_q_extrap(klum):
    """ Extrapolate the fit Lpeak, note this is Lpeak/(nu^2) * q^2 
    Valid in 3581.76666667 < klum < 6330.39378883 """
    return coef0_Lpeak_fit_q_extrap * (slope_Lpeak_fit_q_extrap*(klum - high_kappaL_Lpeak_fit_q_extrap) + offset_Lpeak_fit_q_extrap); 

def Lpeak_fit_q(klum):
    """ Fit Lpeak, note this is Lpeak/(nu^2) * q^2 """
    if np.any(klum < 0.):
        raise ValueError("Invalid values of tidal parameter!")
    high_kappaL = 3581.76666667
    extreme_kappaL = 6330.39378883;
    conds = [klum <= high_kappaL, 
             (klum > high_kappaL) & (klum < extreme_kappaL),
             klum >= extreme_kappaL]
    funcs = [Lpeak_fit_q_nr, 
             Lpeak_fit_q_extrap, 
    lambda x: np.full_like(x,0.,dtype=np.double)]
    return np.piecewise(klum, conds, funcs)

def e_j_final_fit(j_fin):
    """ Fit final/total radiated energy as a function of final angular momentum e^tot_GW (j_rem) """
    return coefs_e_j_final_fit[0] + coefs_e_j_final_fit[1] * j_fin + coefs_e_j_final_fit[2] * j_fin**2

def j_e_final_fit(e_fin):
    """ Fit final angular momentum as a function of final/total radiated energy j_rem(e^tot_GW) """
    return coefs_j_e_final_fit[0] + coefs_j_e_final_fit[1] * e_fin + coefs_j_e_final_fit[2] * e_fin**2

def e_mrg_fit(ktid, nu):
    """ Fit for the energy emitted up to merger with a correction in nu. Note that this EGW/(M nu)(t=tmrg) """
    k = ktid + a_e_mrg_fit * (1 - 4 * nu)
    if np.any(ktid<0.):
        raise ValueError("Invalid values of tidal parameters!")
    return e0_e_mrg_fit * (1 + coefs_e_mrg_fit[0]  * k + coefs_e_mrg_fit[1] * k**2)/(1 + coefs_e_mrg_fit[2] * k + coefs_e_mrg_fit [3] * k**2)

def j_mrg_fit(ktid, nu):
    """ Fit for the angular momentum of the binary at merger with a correction in nu. Note that this  J/(M**2 nu) """
    k = ktid + a_j_mrg_fit * (1 - 4 * nu)
    if np.any(ktid<0.):
                raise ValueError("Invalid values of tidal parameters!")
    return j0_j_mrg_fit * (1 + coefs_j_mrg_fit[0]  * k + coefs_j_mrg_fit[1] * k**2)/(1 + coefs_j_mrg_fit[2] * k + coefs_j_mrg_fit[3] * k**2)
    
if __name__=='__main__':

    """ Usage example """
        
    # Units CGS
    Lplanck = 3.628504984913064e59 # Planck luminosity c^5/G [erg/s]
    clight  = 2.99792458e+10 # [cm/s]
    Msun    = 1.98892e+33 # [g]
    G       = 6.674e-8  # [cm^3/(g s^2)]
    Energy_cgs = Msun*clight**2 
    AngMom_cgs = G * Msun**2/clight

    # Choose pars for a binary
    M = np.array([2.8, 2.4]) # Msun
    lam_A = np.array([500., 300.])
    lam_B = np.array([500., 200.])
    q = np.array([1., 1.2])
        
    # Symmetric mass ratio
    nu = q_to_nu(q)
    
    # Compute the kappa's
    kappa_A = lam_q_to_kappa(lam_A, q)  
    kappa_B = lam_q_to_kappa(lam_B, 1./q)
        
    # Compute kappaT2 and kappa effective for luminosity fit
    kappaT2 = kappa_A + kappa_B
    kapelum = kappalum(kappa_A, q) + kappalum(kappa_B, 1./q)
    
    # Eval fits
    Lpeak_q2 = Lpeak_fit_q(kapelum)
    e_pm = e_pm_fit(kappaT2)
    e_mrg = e_mrg_fit(kappaT2, nu)
    j_mrg = j_mrg_fit(kappaT2, nu)

    e_fin = e_mrg + e_pm
    j = j_e_final_fit(e_fin) # j final requires e_fin !

    # Rescale and CGS units
    E_cgs     = e_fin*nu*M                   * Energy_cgs
    Lpeak_cgs = Lpeak_q2*(nu**2 * 1./(q**2)) * Lplanck
    J_cgs     = j_mrg*M*nu**2                * AngMom_cgs

    for i in range(len(q)):
        # Print results of total energy, luminosity peak and angular momentum at merger
        print('\n')
        print("M = %e q = %e nu = %e lam_A,B = %e, %e"%(M[i], q[i], nu[i], lam_A[i], lam_B[i]))
        print("kappa_A = %e kappa_B = %e"%(kappa_A[i], kappa_B[i]))
        print("kappaT2 = %e kappa_eff^lum = %e"%(kappaT2[i], kapelum[i]))
        print("Natural units (c=G=Msun=1)")
        print("\tEGW_tot/(nu M) = %e\n\tLpeak/nu^2*q^2 = %e\n\tJ_merger/(nu M^2) = %e"%(e_fin[i], Lpeak_q2[i], j_mrg[i]))
        print("CGS units")
        print("\tEGW_tot = %e [erg]\n\tLpeak = %e [erg/s]\n\tJ_merger = %e [g cm^2/s]"%(E_cgs[i], Lpeak_cgs[i], J_cgs[i]))


