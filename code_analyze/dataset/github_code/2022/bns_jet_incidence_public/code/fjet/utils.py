import numpy as np
from . import NS
import bns_lum
from .fitting_formulae import Mdisk_KF20 as Mdisk_fit
from .fitting_formulae import Mej_KF20 as Mej_fit
from .fitting_formulae import Mb
from scipy.optimize import bisect


def lambda_tilde(m1,L1,m2,L2):
    """
    Effective binary tidal deformability parameter.
    
    Input masses can be in any (consistent) units. L1 and L2 are the individual
    dimensionless tidal deformabilities of the two stars.
    """
    
    return 16.*((m1+12*m2)*m1**4*L1 + (m2+12*m1)*m2**4*L2)/(13.*(m1+m2)**5)
    

def get_fshortlived_fGRB(ma,mb,EoS,Mdisk_min=1e-3):
    """
    Fraction of binaries that yield a short-lived (HMNS or BH prompt collapse) 
    remnant, and fraction of binaries that can produce a Blandford-Znajek-powered
    jet (i.e. short-lived & Mdisk>Mdisk_min).
    
    Input:
    - ma,mb: component msses, in Msun
    - EoS: either a string defining the chosen equation of state, or a
           tuple (M_TOV, R1.4)
    - Mdisk_min: the minimum disk mass (in Msun) that yields a GRB. Default: 1e-3
    
    Returns: f_shortlived, f_GRB
    - f_shortlived: the fraction of systems that yields either a HMNS or
                    a prompt BH
    - f_GRB: the fraction of systems that yields a HMNS or a prompt BH and
             produces an accretion disc with a mass >Mdisk_min
    """
    if type(EoS)==str:
        M_max = NS.M_max[EoS]
    else:
        M_max = EoS[0]
    
    BNS = (ma<=M_max)&(mb<=M_max)
    
    m1 = np.maximum(ma[BNS],mb[BNS])
    m2 = np.minimum(ma[BNS],mb[BNS])
    
    if len(m1)<1:
        return 0,0
    
    if type(EoS)==str:
        l1 = NS.Lambda(m1,EoS)
        l2 = NS.Lambda(m2,EoS)
        C1 = NS.C(m1,EoS)
        C2 = NS.C(m2,EoS)
        lt = lambda_tilde(m1,l1,m2,l2)
        mgw = np.zeros_like(m2)
        mgw = bns_lum.egw(m1,m2,l1,l2)
        md = Mdisk_fit(m2,C2)
            # md[i] = Mdisk_fit(lt,m1[i],m2) # if using the Barbieri et al. 2021 fitting formula
        mej = Mej_fit(m1,m2,C1,C2)
        mrem = m1+m2-mgw-md-mej
    else:
        C2 = 1.48*m2/EoS[1]
        md = Mdisk_fit(m2,C2)
        q = m2/m1
        M = m1+m2
        
        mrem = M*(1-0.25*q/(1.+q)**2*1.48*M/EoS[1])-md
    
    f_shortlived = len(mrem[mrem>(1.2*M_max)])/len(mrem)
    f_GRB = len(mrem[(mrem>(1.2*M_max))&(md>Mdisk_min)])/len(mrem)
    return f_shortlived,f_GRB


def M_rem(m1,m2,EoS,method='energy',O_OK=2.):
    """
    Remnant gravitational mass.
    
    Parameters:
    - m1,m2: binary component masses (in Msun)
    - EoS: either a string corresponding to one of the EoS implemented in the NS.py
           module, or a tuple (MTOV,R14), where MTOV is the maximum non-rotating
           NS mass in Msun and R14 is the radius of a 1.4 Msun NS in km.
    
    Keywords:
    - method: if "energy", compute mass threshold based on conservation of energy;
              if "baryon", use baryon number conservation instead.
    - O_OK:   if method='baryon', then O_OK is the ratio of the angular frequency of
              the remnant to the Keplerian (i.e. mass-shedding) angular frequency
              (this is needed to convert the remnant baryon mass into a gravitational
              mass).
    
    Returns: M_HMNS
    - M_HMNS: the binary total mass that separated HMNS remnants from SMNS remnants.
    """

    M = m1+m2
    
    if type(EoS)==str:
        l1 = NS.Lambda(m1,EoS)
        l2 = NS.Lambda(m2,EoS)
        C1 = NS.C(m1,EoS)
        C2 = NS.C(m2,EoS)
        lt = lambda_tilde(m1,l1,m2,l2)
        mgw = np.zeros_like(m2)
        mgw = bns_lum.egw(m1,m2,l1,l2)
        md = Mdisk_fit(m2,C2)
        mej = Mej_fit(m1,m2,C1,C2)
        if method=='energy':
            mrem = m1+m2-mgw-md-mej
        else:
            R14 = NS.R(1.4,EoS)
            m1b = Mb(m1,R14=R14,O_OK=0.)
            m2b = Mb(m2,R14=R14,O_OK=0.)
            mbrem = m1b+m2b-md-mej
            fz = lambda mg:Mb(mg,NS.R(NS.M_max[EoS],EoS),O_OK=O_OK)-mbrem
            mrem = bisect(fz,0.5*mbrem,mbrem)
            
    else:
        C1 = 1.48*m1/EoS[1]
        C2 = 1.48*m2/EoS[1]
        md = Mdisk_fit(m2,C2)
        mej = Mej_fit(m1,m2,C1,C2)
        mrem = M*(1-0.25*q/(1.+q)**2*1.48*M/EoS[1])-md-mej
        
    return mrem

    

def M_HMNS(q,EoS,method='energy',O_OK=2.):
    """
    Threshold binary total gravitational mass separating SMNS and HMNS remnants.
    
    Note: a 1 Msun NS minimum mass for the primary is assumed.
    
    Parameters:
    - q: binary mass ratio (0<q<=1)
    - EoS: either a string corresponding to one of the EoS implemented in the NS.py
           module, or a tuple (MTOV,R14), where MTOV is the maximum non-rotating
           NS mass in Msun and R14 is the radius of a 1.4 Msun NS in km.
    
    Keywords:
    - method: if "energy", compute mass threshold based on conservation of energy;
              if "baryon", use baryon number conservation instead.
    - O_OK:   if method='baryon', then O_OK is the ratio of the angular frequency of
              the remnant to the Keplerian (i.e. mass-shedding) angular frequency
              (this is needed to convert the remnant baryon mass into a gravitational
              mass).
    
    Returns: M_HMNS
    - M_HMNS: the binary total mass that separated HMNS remnants from SMNS remnants.
    """
    
    m1 = np.linspace(1.,3.,300)
    m2 = q*m1
    M = m1+m2
    
    if type(EoS)==str:
        l1 = NS.Lambda(m1,EoS)
        l2 = NS.Lambda(m2,EoS)
        C1 = NS.C(m1,EoS)
        C2 = NS.C(m2,EoS)
        lt = lambda_tilde(m1,l1,m2,l2)
        mgw = np.zeros_like(m2)
        mgw = bns_lum.egw(m1,m2,l1,l2)
        md = Mdisk_fit(m2,C2)
        mej = Mej_fit(m1,m2,C1,C2)
        if method=='energy':
            mrem = m1+m2-mgw-md-mej
        else:
            R14 = NS.R(1.4,EoS)
            m1b = Mb(m1,R14=R14,P_PK=100.)
            m2b = Mb(m2,R14=R14,P_PK=100.)
            mbrem = m1b+m2b-md-mej
            fz = lambda mg:Mb(mg,NS.R(NS.M_max[EoS],EoS),O_OK=O_OK)-mbrem
            mrem = bisect(fz,mbrem,1.5*mbrem)
            
        mhmns = np.max(M[mrem<=(1.2*NS.M_max[EoS])])
    else:
        C1 = 1.48*m1/EoS[1]
        C2 = 1.48*m2/EoS[1]
        md = Mdisk_fit(m2,C2)
        mej = Mej_fit(m1,m2,C1,C2)
        mrem = M*(1-0.25*q/(1.+q)**2*1.48*M/EoS[1])-md-mej
        mhmns = np.min(M[mrem>=(1.2*EoS[0])])
    
    return mhmns
        

def M_GRB(q,EoS,Mdisk_min=1e-3):
    """
    Threshold binary total gravitational mass below which the Blandford-Znajek mechanism
    cannot operate (the assumption here is that you need a HMNS or BH and a disk mass > Mdmin).
    
    Note: a 1 Msun NS minimum mass for the primary is assumed.
    
    Parameters:
    - q: binary mass ratio (0<q<=1)
    - EoS: either a string corresponding to one of the EoS implemented in the NS.py
           module, or a tuple (MTOV,R14), where MTOV is the maximum non-rotating
           NS mass in Msun and R14 is the radius of a 1.4 Msun NS in km.
    - Mdisk_min: the minimum disk mass
    
    Returns: M_GRB
    - M_GRB: the binary total mass that separates GRBs from non-GRBs.
    """
    
    m1 = np.linspace(1.,3.,300)
    m2 = q*m1
    M = m1+m2
    
    if type(EoS)==str:
        l1 = NS.Lambda(m1,EoS)
        l2 = NS.Lambda(m2,EoS)
        C1 = NS.C(m1,EoS)
        C2 = NS.C(m2,EoS)
        lt = lambda_tilde(m1,l1,m2,l2)
        mgw = np.zeros_like(m2)
        mgw = bns_lum.egw(m1,m2,l1,l2)
        md = Mdisk_fit(m2,C2)
        mej = Mej_fit(m1,m2,C1,C2)
        mrem = m1+m2-mgw-md-mej
        mhmns = np.max(M[mrem<=(1.2*NS.M_max[EoS])])
    else:
        C1 = 1.48*m1/EoS[1]
        C2 = 1.48*m2/EoS[1]
        md = Mdisk_fit(m2,C2)
        mej = Mej_fit(m1,m2,C1,C2)
        mrem = M*(1-0.25*q/(1.+q)**2*1.48*M/EoS[1])-md-mej
        try:
            mhmns = np.min(M[(mrem>=(1.2*EoS[0]))&(md>=Mdisk_min)])
        except: # if no array element satisfies the BZ conditions, return 0
            mhmns = 0.
    
    return mhmns

def M_nodisk(q,EoS,Mdisk_min=1e-3):
    """
    Threshold binary total gravitational mass above which disk mass < Mdisk_min).
    
    Parameters:
    - q: binary mass ratio (0<q<=1)
    - EoS: either a string corresponding to one of the EoS implemented in the NS.py
           module, or a tuple (MTOV,R14), where MTOV is the maximum non-rotating
           NS mass in Msun and R14 is the radius of a 1.4 Msun NS in km.
    - Mdisk_min: the minimum disk mass
    
    Returns: M_nodisk
    - M_nodisk: the binary total mass that above which Mdisk<Mdisk_min consistently.
    """
    
    m1 = np.linspace(1.,3.,300)
    m2 = q*m1
    M = m1+m2
    
    if type(EoS)==str:
        
        C2 = NS.C(m2,EoS)
        md = Mdisk_fit(m2,C2)
    else:
        
        C2 = 1.48*m2/EoS[1]
        md = Mdisk_fit(m2,C2)
        
    try:
        mnodisk = np.max(M[(md>=Mdisk_min)])
    except: # if no array element satisfies the BZ conditions, return 0
        mnodisk = np.max(M)
    
    return mnodisk
