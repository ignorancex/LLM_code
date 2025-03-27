import numpy as np
from .particle_masses import *
from .fundamental_constants import GF

D_TO_E_NUE_X = 16.07/100 # +/- 0.30%
D_TO_MU_NUMU_X = 17.6/100 # +/- 3.2%
D0_TO_E_NUE_X = 6.49/100 # +/- 0.11%
D0_TO_MU_NUMU_X = 6.8/100 # +/- 0.6%
DS_TO_TAU_X = 5.48/100 # +/- 0.23%

def l_tilde(x, y):
    l = lambda x, y: 1 + x**2 + y**2 - 2*x*y - 2*x - 2*y
    return l(x,y)/l(x,0)

def h_tilde(x, y):
    h = lambda x, y: x + y - (x-y)**2
    return h(x,y)/h(x,0)

def D_TO_MU_HNL(mN, mixing_squared):
    mD = D_MASS # GeV
    ml = MUON_MASS # GeV
    x = (ml/mD)**2
    y = (mN/mD)**2
    l_til = l_tilde(x, y)
    h_til = h_tilde(x, y)
    # BR_D_TO_E_NUE = 8.3e-5
    BR_D_TO_MU_NUMU = 3.74e-4 # +/- 0.17e-4
    return BR_D_TO_MU_NUMU*np.sqrt(l_til)*h_til*mixing_squared

def D_TO_E_HNL(mN, mixing_squared):
    x = (ELECTRON_MASS/D_MASS)**2
    y = (mN/D_MASS)**2
    l_til = l_tilde(x, y)
    h_til = h_tilde(x, y)
    BR_D_TO_E_NUE = 1.5e-8
    return BR_D_TO_E_NUE*np.sqrt(l_til)*h_til*mixing_squared

def B_TO_ELL_HNL(mN, mixing_squared, mEll):
    x = (mEll/B_MASS)**2
    y = (mN/B_MASS)**2
    l_til = l_tilde(x,y)
    h_til = h_tilde(x, y)
    BR_B_TO_TAU_NUTAU = 1.09e-4
    return BR_B_TO_TAU_NUTAU*np.sqrt(l_til)*h_til*mixing_squared
    

def DS_TO_TAU_HNL(mN, mixing_squared):
    x = (TAU_MASS/DS_MASS)**2
    y = (mN/DS_MASS)**2
    l_til = l_tilde(x, y)
    h_til = h_tilde(x, y)
    DS_TO_TAU_NUTAU = 5.48/100 # +/- 0.23%
    return DS_TO_TAU_NUTAU*np.sqrt(l_til)*h_til*mixing_squared

def DS_TO_ELECTRON_HNL(mN, mixing_squared):
    x = (ELECTRON_MASS/DS_MASS)**2
    y = (mN/DS_MASS)**2
    l_til = l_tilde(x, y)
    h_til = h_tilde(x, y)
    DS_TO_E_NUE = 5.49e-3*(ELECTRON_MASS/MUON_MASS)**2 
    return DS_TO_E_NUE*np.sqrt(l_til)*h_til*mixing_squared

def TAU_TO_PI_HNL(mN, mixing_squared):
    x = (PION_MASS/TAU_MASS)**2
    y = (mN/TAU_MASS)**2
    l = lambda x,y: ((1-x)**2 + (1-y)**2 - 2*x*y)/((1-x)**2)
    g = lambda x,y: ((1-y)**2 - x*(1+y))/(1-x)
    TAU_TO_PI_NUTAU = 10.82/100 # +/- 0.05%
    return TAU_TO_PI_NUTAU*np.sqrt(l(x,y))*g(x,y)*mixing_squared

def TAU_TO_HNL_L_NU_L(mN, mixing_squared):
    BR_TAU_TO_NUTAU_MU_NUMU = 17.39/100 # +/- 0.04%
    BR_TAU_TO_NUTAU_E_NUE= 17.82/100 # +/- 0.04%
    BR_TAU_TO_NUTAU_L_NU_L = BR_TAU_TO_NUTAU_E_NUE + BR_TAU_TO_NUTAU_MU_NUMU 
    if mN + MUON_MASS > TAU_MASS:
        BR_TAU_TO_NUTAU_L_NU_L = BR_TAU_TO_NUTAU_E_NUE
    yN = mN/TAU_MASS
    factor = (1 - 8*yN**2 + 8*yN**6 - yN**8 -12*yN**4*np.log(yN**2))
    return BR_TAU_TO_NUTAU_L_NU_L*factor*mixing_squared