import numpy as np

def C_Love(Lns):
    lnL = np.log(Lns)
    return 0.360-0.0355*lnL+0.000705*lnL**2

L0 = np.logspace(0,5,1000)
C0 = C_Love(L0)

def Love_C(C):
    return np.interp(C,C0[::-1],L0[::-1])

