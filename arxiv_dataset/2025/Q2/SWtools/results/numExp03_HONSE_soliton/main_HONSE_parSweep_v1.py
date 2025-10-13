""" SWtools.py

AUTHOR: O. Melchert
DATE: 2023-2025
"""
import sys; sys.path.append('../../src/')
import sys, os
import numpy as np
import numpy.fft as nfft
from SWtools import SRM


# -- FFT ABREVIATIONS
FT = nfft.ifft
IFT = nfft.fft
FTFREQ = nfft.fftfreq
SHIFT  = nfft.fftshift


def save_npz(out_path, **results):
    dir_name = os.path.dirname(out_path)
    file_basename = os.path.basename(out_path)
    file_extension = file_basename.split(".")[-1]

    try:
        os.makedirs(dir_name)
    except OSError:
        pass

    np.savez_compressed(out_path, **results)

def main(kap=1.0):

    # -- (1) INITIALIZATION AND DECLARATION OF PARAMETERS 
    # ... SIMULATION GRID
    t_max = 20
    N_t = 2**10
    t, dt = np.linspace(-t_max, t_max, N_t, endpoint=True, retstep=True)
    w = FTFREQ(t.size, d=t[1]-t[0])*2*np.pi
    # ... PROBLEM PARAMETERS
    gamma = 1.0
    b2 = -0.2
    b3 = 0.2
    b4 = -1.0
    b1 = -0.07
    coeffs = (b1, b2/2, b3/6, b4/24)
    # ... INITIAL CONDITION AND EXACT SOLUTION
    psi0 = np.exp(-t**2)

    myS = SRM(t, coeffs, lambda I, t: gamma*I, tol=1e-12, maxiter=10000, nskip=1)
    U, acc, succ, msg = myS.solve(psi0, kap)

    # -- (3) POSTPROCESS RESULTS
    results = {
        't': t,
        'vinv': b1,
        'b2': b2,
        'b3': b3,
        'b4': b4,
        'kap': kap,
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

    save_npz('./data_HONSE/res_HONSE_SRM_kap%lf_vinv%lf'%(kap,b1), **results)


if __name__ == "__main__":
    main(kap=0.200)
    main(kap=0.867)
    main(kap=10.00)
