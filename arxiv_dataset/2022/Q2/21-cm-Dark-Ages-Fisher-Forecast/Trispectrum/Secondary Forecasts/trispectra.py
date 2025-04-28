import numpy as np
from powerspectrum import *
from numba import njit

@njit
def trianglecondition(k1,k2,K12): #checks whether the triangle condition is VIOLATED
    bools = k1 + k2 < K12
    bools += k1 + K12< k2
    bools += k2 + K12< k1
    return bools

@njit
def TtauNL(k1,k2,k3,k4,K12,K14,K13,mu=0,alpha0=0):
    P1 = P(k1)
    P2 = P(k2)
    P3 = P(k3)
    P4 = P(k4)
    T12 = P(K12)*(P1*P3 + P1*P4 + P2*P3 + P2*P4)
    T13 = P(K13)*(P1*P2 + P1*P4 + P3*P2 + P3*P4)
    T14 = P(K14)*(P1*P2 + P1*P3 + P4*P2 + P4*P3)
    return (T12 + T13 + T14)

@njit
def TgNL(k1,k2,k3,k4,K12,K14,K13,mu=0,alpha0=0):
    P1 = P(k1)
    P2 = P(k2)
    P3 = P(k3)
    P4 = P(k4)
    return 54./25. * (P1*P2*P3 + P4*P1*P2 + P3*P4*P1 + P2*P3*P4)

def Teq1(k1,k2,k3,k4,K12,K14,K13,mu=0,alpha0=0):
    return 221184./25. * A**3. /(k1*k2*k3*k4) / (k1 + k2 + k3 + k4)**5.

@njit
def Teq2Perm(k1,k2,k3,k4,K12):
    K = k1 + k2 + k3 + k4
    kdot = (-k3*k3-k4*k4+K12*K12)/2.
    return (K**2. + 3.*(k3+k4)*K + 12.*k3*k4)/(k1*k2*k3**3. * k4**3. * K**5.) * kdot

@njit
def Teq2(k1,k2,k3,k4,K12,K14,K13,mu=0,alpha0=0):
    return -27648./325. * A**3 * (Teq2Perm(k1,k2,k3,k4,K12)+Teq2Perm(k1,k3,k2,k4,K13)+Teq2Perm(k1,k4,k2,k3,K14)+Teq2Perm(k2,k3,k1,k4,K14)+Teq2Perm(k2,k4,k3,k1,K13)+Teq2Perm(k3,k4,k1,k2,K12))

@njit
def Teq3Perm(k1,k2,k3,k4,K12):
    K = k1 + k2 + k3 + k4
    kdot1 = (-k1*k1 - k2*k2 + K12*K12)/2.
    kdot2 = (-k3*k3 - k4*k4 + K12*K12)/2.
    return kdot1*kdot2

@njit
def Teq3(k1,k2,k3,k4,K12,K14,K13,mu=0,alpha0=0):
    K = k1 + k2 + k3 + k4
    return 165888./2575. * A**3. * (2.*K**4. - 2.*K**2. * (k1**2. + k2**2. + k3**2. + k4**2.) + K*(k1**3. + k2**3. + k3**3. + k4**3.) + 12.*k1*k2*k3*k4) / (k1*k2*k3*k4)**3. / K**5. * (Teq3Perm(k1,k2,k3,k4,K12) + Teq3Perm(k1,k3,k2,k4,K13) + Teq3Perm(k1,k4,k2,k3,K14))

@njit
def g1(k1,k2,k3,k4,K12,K14):
    return K12*K12*K14*K14*(k1*k1+k2*k2+k3*k3+k4*k4-K12*K12-K14*K14) - K12*K12*(k2*k2-k3*k3)*(k1*k1-k4*k4) + K14*K14*(k1*k1-k2*k2)*(k3*k3-k4*k4) - (k1*k1*k3*k3-k2*k2*k4*k4)*(k1*k1-k2*k2 + k3*k3-k4*k4)

@njit
def TClockPerm(k1,k2,k3,k4,s,mu,alpha0):
    alpha = s*s/(k1+k2)/(k3+k4)

    alpha12 = (k1+k2)/s
    alpha34 = (k3+k4)/s
    bools = alpha12 < alpha0
    bools += alpha34 < alpha0
    result = 1./(k1*k2*k3*k4)**3. *  ((k1+k2)*(k3+k4))**(3./2.) * np.sin(mu * np.log(alpha))
    result[bools] = 0
    return result

@njit
def TClock(k1,k2,k3,k4,K12,K14,K13,mu,alpha0=2):
    return 3.**(3./2.)/16. * A**3. * (TClockPerm(k1,k2,k3,k4,K12,mu,alpha0) + TClockPerm(k1,k3,k2,k4,K13,mu,alpha0) + TClockPerm(k1,k4,k2,k3,K14,mu,alpha0))

@njit
def TintPerm(k1,k2,k3,k4,s,nu,alpha0):
    P1 = P(k1)
    P2 = P(k2)
    P3 = P(k3)
    P4 = P(k4)
    Ps = P(s)
    alpha = s*s/(k1+k2)/(k3+k4)

    alpha12 = (k1+k2)/s
    alpha34 = (k3+k4)/s
    bools = alpha12 < alpha0
    bools+= alpha34 < alpha0
    result = 4. * np.power(3.,3./2. - nu) * np.sqrt(P1*P2*P3*P4)*Ps*np.power(alpha,3./2.-nu)
    result[bools] = 0
    return result

@njit
def Tint(k1,k2,k3,k4,K12,K14,K13,mu,alpha0=2):
    return  (TintPerm(k1,k2,k3,k4,K12,mu,alpha0) + TintPerm(k1,k3,k3,k4,K13,mu,alpha0) + TintPerm(k1,k4,k2,k4,K14,mu,alpha0))