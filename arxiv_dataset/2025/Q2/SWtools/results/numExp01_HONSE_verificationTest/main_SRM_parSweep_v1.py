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


def main(N_t=2**10):

    # -- (1) INITIALIZATION AND DECLARATION OF PARAMETERS 
    # ... SIMULATION GRID
    t_max = 20
    t, dt = np.linspace(-t_max, t_max, N_t, endpoint=True, retstep=True)
    #w = FTFREQ(t.size, d=t[1]-t[0])*2*np.pi
    # ... PROBLEM PARAMETERS
    gamma = 1.0
    b2 = -1.
    b4 = -1.
    coeffs = (0,b2/2,0,b4/24)
    #Lw = (b2/2)*w**2 + (b4/24)*w**4
    # ... EXACT SOLUTION
    P0 = 9*b2**2/(5*gamma*np.abs(b4))
    t0 = np.sqrt(5*b4/(3*b2))
    kap = 24/25*b2**2/np.abs(b4)
    U_exact = np.sqrt(P0)/np.cosh(t/t0)**2
    E0 = np.trapz(np.abs(U_exact)**2, dx=dt)
    # ... INITIAL CONDITION
    U0 = np.exp(-t**2)
    # ... SRM INSTANTIATION
    mySW = SRM(t, coeffs, lambda I, t: gamma*I, tol=1e-12, maxiter=10000, nskip=1)

    # -- (2) SPECTRAL RENORMALIZATION PROCEDURE 
    U, acc, succ, msg = mySW.solve(U0, kap)

    # -- (3) POST PROCESSING
    # ... ROOT-MEAN-SQUARE DEVIATION OF THE PULSE ENERGY
    err_glob = np.sqrt(np.trapz(np.abs(U_exact - U)**2, dx=dt)/E0)
    print(f"{N_t} {dt} {mySW.N} {np.abs(mySW.H)} {mySW.kap} {acc} {err_glob}")


def wrapper():
    for n in [8,9,10]:
        main(2**n)
        main(int(np.sqrt(2)*2**n))

if __name__=='__main__':
    wrapper()
