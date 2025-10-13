""" SWtools.py

AUTHOR: O. Melchert
DATE: 2023-2025
"""
import os, sys; sys.path.append('../../src/')
import numpy as np
from SWtools import SuccessiveOverrelaxationMethod as NSOM


def save_npz(out_path, **results):
    dir_name = os.path.dirname(out_path)
    file_basename = os.path.basename(out_path)
    file_extension = file_basename.split(".")[-1]

    try:
        os.makedirs(dir_name)
    except OSError:
        pass

    np.savez_compressed(out_path, **results)


def main(beta=3.1371, wORP=1.0):

    # -- (1) INITIALIZATION AND DECLARATION OF PARAMETERS 
    # ... SIMULATION GRID
    x_max =  16
    x = np.linspace(-x_max, x_max, int(2*x_max*32)+1, endpoint=True)
    # ... PROBLEM PARAMETERS
    c2 = -0.5
    F = lambda I, x: -(0.5*x**2 + beta*I)
    # ... NORMALIZATION CONSTRAINT
    N0 =  1
    # ... TRIAL SOLUTION
    U0 = np.exp(-x**2/2)/np.pi**0.25
    # ... SORM INSTANTIATION
    myS = NSOM(x, (0,c2,0,0), F, tol=1e-12, ORP=wORP, maxiter=100000, nskip=1, verbose=True)

    # -- (2) PERFORM CALCULATION
    U, acc, succ, msg = myS.solve(U0, N0)

    # -- (3) POSTPROCESS RESULTS
    results = {
        'x': x,
        'N0': N0,
        'beta': beta,
        'c2': c2,
        'U0': U0,
        'nit': myS.num_iter,
        'U': U,
        'acc': acc,
        'kap': myS.kap,
        'U_list': myS.U_list,
        'N_list': myS.N_list,
        'H_list': myS.H_list,
        'kap_list': myS.K_list,
        'acc_list': myS.acc_list,
        'iter_list': myS.iter_list
    }

    save_npz('./data_1DGPE/res_1DGPE_SOR_beta%lf_wORP%lf'%(beta,wORP), **results)


def wrapper():
    wORP = 1.5
    for beta in [3.1371, 31.371, 156.855, 627.42]:
        main(beta=beta, wORP=wORP)


if __name__=="__main__":
    wrapper()

