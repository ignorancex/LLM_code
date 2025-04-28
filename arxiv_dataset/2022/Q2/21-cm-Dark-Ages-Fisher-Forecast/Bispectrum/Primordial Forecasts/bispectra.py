import numpy as np
from powerspectrum import *
# from numba import njit

# @njit
def BLocal(P1prim,P2prim,P3prim):
    return 6/5*(P1prim*P2prim + P1prim*P3prim + P2prim*P3prim)

# @njit
def BOrtho(P1prim,P2prim,P3prim):
    return 18/5*(-3.*(P1prim*P2prim+P1prim*P3prim+P2prim*P3prim)-8.*np.power(P1prim*P2prim*P3prim,2./3.) + 3.*(np.power(P1prim,1./3.)*np.power(P2prim,2./3.)*P3prim + np.power(P1prim,1./3.)*np.power(P3prim,2./3.)*P2prim + np.power(P2prim,1./3.)*np.power(P1prim,2./3.)*P3prim + np.power(P2prim,1./3.)*np.power(P3prim,2./3.)*P1prim + np.power(P3prim,1./3.)*np.power(P2prim,2./3.)*P1prim + np.power(P3prim,1./3.)*np.power(P1prim,2./3.)*P2prim))

# @njit
def BEquil(P1prim,P2prim,P3prim):
    return 18/5*(-(P1prim*P2prim+P1prim*P3prim+P2prim*P3prim)-2.*np.power(P1prim*P2prim*P3prim,2./3.) + np.power(P1prim,1./3.)*np.power(P2prim,2./3.)*P3prim + np.power(P1prim,1./3.)*np.power(P3prim,2./3.)*P2prim + np.power(P2prim,1./3.)*np.power(P1prim,2./3.)*P3prim + np.power(P2prim,1./3.)*np.power(P3prim,2./3.)*P1prim + np.power(P3prim,1./3.)*np.power(P2prim,2./3.)*P1prim + np.power(P3prim,1./3.)*np.power(P1prim,2./3.)*P2prim)


# @njit
def BClockPermVec(k1,k2,k3,mu,alpha0):
    alpha = (k1 + k2)/k3
    bools = alpha < alpha0

    results = alpha**(-1./2.) *  np.sin(mu*np.log(alpha/2))
    results[bools] *= 0
    return A*A*((k1*k2*k3)**(-2.)) * 3.**(9./2.)/10. * results

# @njit
def BClockVec(k1,k2,k3,mu,alpha0):
    a = BClockPermVec(k1,k2,k3,mu,alpha0)
    b = BClockPermVec(k1,k3,k2,mu,alpha0)
    c = BClockPermVec(k3,k2,k1,mu,alpha0)
    return  a+b+c

# @njit
def BIntVecPerm(k1,k2,k3,nu,alpha0=0):
    alpha123 = (k1+k2)/k3
    bools = alpha123 < alpha0

    result = 6./5.*np.power(3,7./2.-3.*nu) * (k1*k1 + k2*k2 + k3*k3) / np.power(k1+k2+k3,7./2.-3*nu) * np.power(k1*k2*k3,1./2.-nu) * A*A*((k1*k2*k3)**(-2.))
    result[bools] = 0
    return result

# @njit
def BIntVec(k1,k2,k3,nu,alpha0):
    a= BIntVecPerm(k1,k2,k3,nu,alpha0) 
    b= BIntVecPerm(k1,k3,k2,nu,alpha0)
    c= BIntVecPerm(k3,k2,k1,nu,alpha0)
    if alpha0 == 0: return a
    return a+b+c
