# %%

# PHOTOMETRIC DISTANCE CALCULATOR
# Using the polynomial fits in Table 5 (Cifuentes et al. 2020)
# Cifuentes et al. (2021)

import numpy as np
import matplotlib

def dphot_G(G, J, eG, eJ):
    a = 16.24
    b = -13.04
    c = 5.64
    d = -0.622
    aerr = 4.57
    berr = 4.80
    cerr = 1.66
    derr = 0.188
    #
    x = G - J
    Mabs = a + b * x + c * x**2 + d * x**3
    xerr = np.sqrt(eG**2 + eJ**2)
    Mabserr = np.sqrt(aerr**2 + x**2*berr**2 + x**4*cerr **
                      2 + x**6+derr**6 + (b + 2*x*c + 3*d*x**2)**2*xerr**2)
    d = 10**((5 + G - Mabs)/5)
    derr = np.sqrt(d*(eG**2 + Mabserr**2)**2/np.log(10))
    return(d)

def dphot_r(r, J, er, eJ):
    a = 8.38
    b = -2.74
    c = 1.47
    d = -0.132
    aerr = 2.68
    berr = 2.36
    cerr = 0.68
    derr = 0.063
    #
    x = r - J
    Mabs = a + b * x + c * x**2 + d * x**3
    xerr = np.sqrt(er**2 + eJ**2)
    Mabserr = np.sqrt(aerr**2 + x**2*berr**2 + x**4*cerr **
                      2 + x**6+derr**6 + (b + 2*x*c + 3*d*x**2)**2*xerr**2)
    d = 10**((5 + r - Mabs)/5)
    return d

def (G, r, J, eG, er, eJ):
    d_G = dphot_G(G, J, eG, eJ)
    d_r = dphot_r(r, J, er, eJ)
    return (d_G, d_r)
