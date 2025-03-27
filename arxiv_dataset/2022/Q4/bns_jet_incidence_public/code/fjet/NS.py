import numpy as np
import CLove
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

EoS_list = ['APR4','DD2','SFHo']

M_NS = {}
R_NS = {}
C_NS = {}
L_NS = {}
M_max = {}

for EoS in EoS_list:
    M_NS[EoS] = np.loadtxt(os.path.join(dname,'M_R/{}.txt'.format(EoS)),usecols=(1),skiprows=1)
    R_NS[EoS] = np.loadtxt(os.path.join(dname,'M_R/{}.txt'.format(EoS)),usecols=(0),skiprows=1)
    C_NS[EoS] = 1.477*M_NS[EoS]/R_NS[EoS]
    L_NS[EoS] = CLove.Love_C(C_NS[EoS])
    M_max[EoS] = np.max(M_NS[EoS])

def m_to_mb(mns,comp):
    # return mns + 0.08*mns**2 # Timmes-like formula, with A=0.08 from arXiv:1905.03784
    return mns * ( 1. + (0.6 * comp / (1. - 0.5 * comp)) )

def Mb(M,EoS):
    R14 = R(1.4,EoS)
    return M+M**2/R14

def Lambda(M,EoS):
    if EoS not in EoS_list:
        print('EoS not implemented.')
        return None
    else:
        return np.interp(M,M_NS[EoS],L_NS[EoS])

def R(M,EoS):
    if EoS not in EoS_list:
        print('EoS not implemented.')
        return None
    else:
        return np.interp(M,M_NS[EoS],R_NS[EoS])

def C(M,EoS):
    if EoS not in EoS_list:
        print('EoS not implemented.')
        return None
    else:
        return np.interp(M,M_NS[EoS],C_NS[EoS])

def Risco(M,f=1e3):
    """
    R_isco (in cm) for a neutron star of mass M (Msun), rotating at frequency f (Hz).
    Computed using universal relations from Luk & Lin 2018 (arXiv:1805.10813)
    """
    a1 = 8.809
    a2 = -9.166e-4
    a3 = 8.787e-8
    a4 = -6.019e-12
    x = M*f
    Riscof = a1*x+a2*x**2+a3*x**3+a4*x**4
    return Riscof/f*1e5


# Functions of Spin to calculate R_isco
def Z1(x):
    return(1.+np.power(1.-x**2,1./3.)*(np.power(1.+x,1./3.)+np.power(1.-x,1./3.)))

def Z2(x):
    return(np.sqrt(3.*(x**2)+Z1(x)**2))

def Risco_BH(aBH):
    """
    Innermost Stable Circular Orbit in units of the gravitational radius
    Input: 
      aBH    : dimensionless spin paramater [-]
    """
    return((3.+Z2(aBH)-np.sign(aBH)*np.sqrt((3.-Z1(aBH))*(3.+Z1(aBH)+2.*Z2(aBH)))))    


bkp_coeff = {'APR4':(-0.21,0.69),'SFHo':(-0.27,0.74),'DD2':(-0.20,0.93)} # coefficiencts to compute breakup frequency from Radice et al. 2018

def f_breakup(M,EoS='APR4'):
    """
    Breakup frequency (Hz) for a rotating neutron star of mass M (Msun) with a given EoS
    """
    a,b = bkp_coeff[EoS]
    
    Mb = m_to_mb(M,C(M,EoS))
    
    P = a*(Mb-2.5)+b # breakup period in ms
    return (P/1000.)**-1
    
def spherical_cap_volume(x):
    mf = 0.25*(2+x)*(x-1)**2
    return mf

x0 = np.linspace(0.,1.,1000)
cap = spherical_cap_volume(x0)

def myMdisk(lt,M1,M2,x=[2.25690996e+02,1.04974433e-01,2.33574894e-01]):
    
    q = M2/M1
    l = (lt/x[0])**x[1]
    l1 = l*q**x[2]
    l2 = l*q**(-x[2])
    
    MD2 = np.interp(2./(1.+q**-1)  +2./l2-2.,x0,cap)*M2
    MD1 = np.interp(2./(1.+q**1 )  +2./l1-2.,x0,cap)*M1
    
    return max(1e-3,MD2+MD1)
