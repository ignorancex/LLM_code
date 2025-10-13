""" SWtools.py

AUTHOR: O. Melchert
DATE: 2023-2025
"""
import os, sys
import numpy as np
from SWtools import NSOM


def save_npz(out_path, **results):
    dir_name = os.path.dirname(out_path)
    file_basename = os.path.basename(out_path)
    file_extension = file_basename.split(".")[-1]

    try:
        os.makedirs(dir_name)
    except OSError:
        pass

    np.savez_compressed(out_path, **results)


def main(beta=1.):

    # -- (1) INITIALIZATION AND DECLARATION OF PARAMETERS 
    # ... COMPUTATIONAL DOMAIN
    x = np.linspace(-8, 8, 512)
    k = np.fft.fftfreq(x.size, d=x[1]-x[0])*2*np.pi
    nskip = 10
    # ... 1D GPE INGREDIENTS 
    c2 = -0.5
    Lk = c2*k**2
    V = lambda x: 0.5*x**2
    F = lambda I, x:  -V(x) - beta*I
    N = 1.0
    # ... INITIAL CONDITIONS 
    U0_0 = np.exp(-x**2/2)/np.pi**0.25 + 0j
    U0_1 = 2*x*np.exp(-x**2/2)/np.pi**0.25/np.sqrt(2) + 0j
    U0_2 = (4*x**2-2)*np.exp(-x**2/2)/np.pi**0.25/np.sqrt(8) + 0j
    U0_3 = (8*x**3-12*x)*np.exp(-x**2/2)/np.pi**0.25/np.sqrt(48) + 0j
    # ... INSTANTIATE METHOD 
    myS = NSOM(x, (0,c2,0,0), F, tol=1e-12, ORP=1.5, maxiter=10000, nskip=nskip, verbose=True)

    # -- SUCCESSIVE OVERRELAXATION 
    # ... GROUND-STATE
    U0, err, succ, msg = myS.solve(U0_0, N)
    kap0 = myS.kap
    acc0 = myS.acc_list
    # ... 1ST EXCITED STATE
    U1, err, succ, msg = myS.solve(U0_1, N, ortho_set = [U0])
    kap1 = myS.kap
    acc1 = myS.acc_list
    # ... 2ND EXCITED STATE
    U2, err, succ, msg = myS.solve(U0_2, N, ortho_set = [U0, U1])
    kap2 = myS.kap
    acc2 = myS.acc_list
    # ... 2ND EXCITED STATE
    U3, err, succ, msg = myS.solve(U0_3, N, ortho_set = [U0, U1, U2])
    kap3 = myS.kap
    acc3 = myS.acc_list

    # -- (3) POSTPROCESS RESULTS
    results = {
        'x': x,
        'N': N,
        'beta': beta,
        'c2': c2,
        'V': V(x),
        'U0_0': U0_0,
        'U0_1': U0_1,
        'U0_2': U0_2,
        'U0_3': U0_3,
        'nskip': nskip,
        'U0': U0,
        'kap0': kap0,
        'acc0': acc0,
        'U1': U1,
        'kap1': kap1,
        'acc1': acc1,
        'U2': U2,
        'kap2': kap2,
        'acc2': acc2,
        'U3': U3,
        'kap3': kap3,
        'acc3': acc3,
    }

    save_npz('./data_1DGPE/res_1DGPE_NSOM_beta%lf'%(beta), **results)


if __name__=="__main__":
    main(beta=1.)

