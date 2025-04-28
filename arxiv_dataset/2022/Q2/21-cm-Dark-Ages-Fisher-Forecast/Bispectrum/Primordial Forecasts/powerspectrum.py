import numpy as np
# from numba import njit

As = 2.1056*1e-9;
A = 2*np.pi*np.pi*As
ns = 0.9665
kpivot = 0.05
pfactor = A*kpivot**(1.-ns)

# @njit
def P(k):
    return np.power(k,ns-4.)*pfactor

