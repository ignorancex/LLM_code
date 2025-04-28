import numpy as np
# from numba import njit

h = 0.6711
As = 2.13e-09;
A = 2*np.pi*np.pi*As
ns = 0.9624
kpivot = 0.05
pfactor = A*kpivot**(1.-ns)

# @njit
def P(k):
    return np.power(k,ns-4.)*pfactor

