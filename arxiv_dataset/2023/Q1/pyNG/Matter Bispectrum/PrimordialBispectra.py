import numpy as np
from PrimordialPowerspectrum import *

def BLocal(P1prim,P2prim,P3prim):
    return 6./5.*(P1prim*P2prim + P1prim*P3prim + P2prim*P3prim)

def BOrtho(P1prim,P2prim,P3prim):
    return 18./5.*(-3.*(P1prim*P2prim+P1prim*P3prim+P2prim*P3prim)-8.*np.power(P1prim*P2prim*P3prim,2./3.) + 3.*(np.power(P1prim,1./3.)*np.power(P2prim,2./3.)*P3prim + np.power(P1prim,1./3.)*np.power(P3prim,2./3.)*P2prim + np.power(P2prim,1./3.)*np.power(P1prim,2./3.)*P3prim + np.power(P2prim,1./3.)*np.power(P3prim,2./3.)*P1prim + np.power(P3prim,1./3.)*np.power(P2prim,2./3.)*P1prim + np.power(P3prim,1./3.)*np.power(P1prim,2./3.)*P2prim))

def BOrthoAlt(k1,k2,k3,mu=0,alpha0=0):
    kt = k1 + k2 + k3
    delta =  (kt - 2*k1)*(kt - 2*k2)*(kt - 2*k3)
    gamma = 2.*(k1*k2 + k2*k3 + k3*k1)/3. - (k1**2 + k2**2 + k3**2)/3.
    p = 27. / (-21. + 743./(7.*(20.*np.pi**2 - 193.)))
    return 18./5. * A**2 *  ( (1.+p) * delta / (k1*k2*k3)**3. - p * gamma**3 / (k1*k2*k3)**4.)

def BEquil(P1prim,P2prim,P3prim):
    return 18./5.*(-(P1prim*P2prim+P1prim*P3prim+P2prim*P3prim)-2.*np.power(P1prim*P2prim*P3prim,2./3.) + np.power(P1prim,1./3.)*np.power(P2prim,2./3.)*P3prim + np.power(P1prim,1./3.)*np.power(P3prim,2./3.)*P2prim + np.power(P2prim,1./3.)*np.power(P1prim,2./3.)*P3prim + np.power(P2prim,1./3.)*np.power(P3prim,2./3.)*P1prim + np.power(P3prim,1./3.)*np.power(P2prim,2./3.)*P1prim + np.power(P3prim,1./3.)*np.power(P1prim,2./3.)*P2prim)

def BFlat(P1prim,P2prim,P3prim):
    return 18./5. * np.power(P1prim*P2prim*P3prim,2./3.)


def BClockPermVec(k1,k2,k3,mu,alpha0):
    alpha = (k1 + k2)/k3
    bools = alpha < alpha0

    results = alpha**(-1./2.) *  np.sin(mu*np.log(alpha/2))
    results[bools] *= 0
    return A*A*((k1*k2*k3)**(-2.)) * 3.**(9./2.)/10. * results

def BClockVec(k1,k2,k3,mu,alpha0):
    a = BClockPermVec(k1,k2,k3,mu,alpha0)
    b = BClockPermVec(k1,k3,k2,mu,alpha0)
    c = BClockPermVec(k3,k2,k1,mu,alpha0)
    return  a+b+c

def BIntVecPerm(k1,k2,k3,nu,alpha0=0):
    alpha123 = (k1+k2)/k3
    bools = alpha123 < alpha0

    result = 6./5.*np.power(3,7./2.-3.*nu) * (k1*k1 + k2*k2 + k3*k3) / np.power(k1+k2+k3,7./2.-3*nu) * np.power(k1*k2*k3,1./2.-nu) * A*A*((k1*k2*k3)**(-2.))
    result[bools] = 0
    return result

def BIntVec(k1,k2,k3,nu,alpha0):
    a= BIntVecPerm(k1,k2,k3,nu,alpha0) 
    b= BIntVecPerm(k1,k3,k2,nu,alpha0)
    c= BIntVecPerm(k3,k2,k1,nu,alpha0)
    if alpha0 == 0: return a
    return a+b+c
