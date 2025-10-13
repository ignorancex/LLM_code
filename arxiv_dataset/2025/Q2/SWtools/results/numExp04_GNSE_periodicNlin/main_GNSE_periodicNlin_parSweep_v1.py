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



def main(alpha=-1.):

    # -- (1) INITIALIZATION AND DECLARATION OF PARAMETERS 
    # ... SIMULATION GRID
    t_max = 20
    N_t = 2**12
    t, dt = np.linspace(-t_max, t_max, N_t, endpoint=True, retstep=True)
    #w = FTFREQ(t.size, d=t[1]-t[0])*2*np.pi
    # ... PROBLEM PARAMETERS
    gamma = 1.0
    b2 = -1.
    #Lw = b2*w**2
    coeffs = (0,-1.,0,0)
    m = alpha*np.cos(2*np.pi*t*2)
    F = lambda I, t: (1+m)*I
    kap = 1.
    # ... INITIAL CONDITION AND EXACT SOLUTION
    psi0 = np.exp(-t**2)

    myS = SRM(t, coeffs, F , tol=1e-12, maxiter=100000, nskip=1, verbose=False)
    U, acc, succ, msg = myS.solve(psi0, kap)

    # -- (3) POSTPROCESS RESULTS
    results = {
        't': t,
        'b2': b2,
        'kap': kap,
        'alpha': alpha,
        'nit': myS.num_iter,
        'm': m,
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

    save_npz('./data_GNSE/res_GNSE_SRM_kap%lf_alpha%lf'%(kap, alpha), **results)


def wrapper():
    for alpha in [-0.3,-0.8,-3.,-8.]:
        main(alpha=alpha)


if __name__=="__main__":
    wrapper()

