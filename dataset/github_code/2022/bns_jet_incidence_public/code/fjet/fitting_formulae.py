import numpy as np
from CLove import Love_C

def lambda_tilde(m1,L1,m2,L2):
    """
    Effective binary tidal deformability parameter.
    
    Input masses can be in any (consistent) units. L1 and L2 are the individual
    dimensionless tidal deformabilities of the two stars.
    """
    
    return 16.*((m1+12*m2)*m1**4*L1 + (m2+12*m1)*m2**4*L2)/(13.*(m1+m2)**5)


def MBH_Coughlin19(m1,m2,lt,a=0.98,b=-0.093):
    nu = m1*m2/(m1+m2)**2
    return a*(4*nu)**2*(m1+m2+b*lt/400.)
    

def Mb(Mg,R14=12.5,O_OK=0.):
    """
    Baryonic mass as a function of gravitational mass Mg, R_1.4 (in km) and O/O_K (ratio of
    frequency over Keplerian - i.e. mass-shedding - frequency, from Gao et al. 2020
    """
    return Mg+np.exp(-0.25*O_OK)*Mg**2/R14

def Mthres_Tootle21(q,Mmax,Rmax,chi=0.,b=1.01,c=1.34,a2=0.11,a3=0.12,a4=0.07,a5=-0.3,chimax=1.3):
    """
    Threshold total mass towards prompt BH collapse, as given in Tootle et al. 2021.
    
    Parameters:
    - q: mass ratio m2/m1 (< 1)
    - Mmax: maximum mass of a non-rotating neutron star, in Msun
    - Rmax: radius of a neutron star with M=Mmax, in km
    
    Keywords:
    - chi = chi1 + chi2 = sum of the spin parameters of the components.
    - b,c,a1,a2,a3,a4,a5: fitting formula parameters.
    - chimax = maximum possible dimensionless spin.
    
    Returns: Mthres
    - Mthres: threshold total mass in Msun
    
    """
    a = 2*b/(2.-c)
    a1 = 1.
    a6 = (a3+a4*(1.-q))/(2*chimax)
    
    Cmax = 1.48*Mmax/Rmax
    k = (a-b/(1.-c*Cmax))*Mmax
    f = a1 + a2*(1.-q) + a3*chi + a4*(1.-q)*chi + a5*(1.-q)**2 + a6*chi**2
    
    return k*f


def Mthres_Bauswein21(q,Mmax,R16,c1=0.578,c2=0.161,c3=-0.218,c4=8.987,c5=-1.767):
    """
    Threshold total mass towards prompt BH collapse, as given in Bauswein et al. 2021.
    
    Parameters:
    - q: mass ratio m2/m1 (< 1)
    - Mmax: maximum mass of a non-rotating neutron star, in Msun
    - R16: radius of a 1.6 Msun neutron star, in km
    
    Keywords:
    - c1,c2,c3,c4,c5: fitting formula parameters. Default values correspond 
                      to the base hadronic sample of Bauswein+21.
    
    Returns: Mthres
    - Mthres: threshold total mass in Msun
    
    """
    return c1*Mmax+c2*R16+c3+c4*(1.-q)**3*Mmax+c5*(1.-q)**3*R16

def Mdisk_KF20(m2,C2,a=-8.1324,cc=1.4820,d=1.7784,Mmin=5e-4): # a=-8.1324,cc=1.4820,d=1.7784 original K&F 2020 params
    """
    BNS disk mass from Kruger & Foucart 2020.
    
    Parameters:
    - m2: gravitational mass of the secondary (Msun)
    - C2: compactness of the secondary
    
    Returns: Mdisk
    - Mdisk: disk baryonic mass in Msun
    """
    return np.maximum(Mmin,m2*np.maximum(0.,a*C2+cc)**d)

def Mej_KF20(m1,m2,C1,C2,a=-9.3335,b=114.17,cc=-337.56,n=1.5465):
    """
    BNS dynamical ejecta mass from Kruger & Foucart 2020.
    
    Parameters:
    - m1: gravitational mass of the primary (Msun)
    - m2: gravitational mass of the secondary (Msun)
    - C1: compactness of the primary
    - C2: compactness of the secondary
    
    Returns: Mej
    - Mej: ejecta baryonic mass in Msun
    """
    me1 = m1*(a/C1+b*(m2/m1)**n+cc*C1)
    me2 = m2*(a/C2+b*(m1/m2)**n+cc*C2)
    return np.maximum(0.,1e-3*(me1+me2))

# ----- my fitting func for Mdisk ----------

def spherical_cap_volume(x):
    mf = 0.25*(2+x)*(x-1)**2
    return mf

x0 = np.linspace(0.,1.,1000)
cap = spherical_cap_volume(x0)

def Mdisk_B21(M1,M2,C1,C2,x=[2.25690996e+02,1.04974433e-01,2.33574894e-01]):
    """
    BNS disk mass from Barbieri et al. 2021.
    
    Parameters:
    - M1: gravitational mass of the primary (Msun)
    - M2: gravitational mass of the secondary (Msun)
    - C1,C2: compactness of primary and secondary
    
    Returns: Mdisk
    - Mdisk: disk baryonic mass in Msun
    """
    
    L1 = Love_C(C1)
    L2 = Love_C(C2)
    lt = lambda_tilde(M1,L1,M2,L2)
    
    q = M2/M1
    l = (lt/x[0])**x[1]
    l1 = l*q**x[2]
    l2 = l*q**(-x[2])
    
    MD2 = np.interp(2./(1.+q**-1)  +2./l2-2.,x0,cap)*M2
    MD1 = np.interp(2./(1.+q**1 )  +2./l1-2.,x0,cap)*M1
    
    return np.maximum(1e-3,MD2+MD1)
