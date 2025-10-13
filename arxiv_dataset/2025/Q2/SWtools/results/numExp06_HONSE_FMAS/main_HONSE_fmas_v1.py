"""main_HONSE_fmas_v1.py

This module demonstrates how SWtools can be used with fmas [1]_, an open-source
Python package for the accurate simulation of the propagation dynamics of
optical pulses.  It implements one of the propagation scenarios considered in
[2]_.  This example is discussed as SWtools use-case in [2]_.

References
----------

.. [1] O. Melchert, A. Demircan, py-fmas: A python package for ultrashort
optical pulse propagation in terms of forward models for the analytic signal,
Computer Physics Communications 273 (2022) 108257,
https://doi.org/10.1016/j.cpc.2021.108257.

.. [2] O. Melchert, A. Demircan, Numerical investigation of solitary-wave
solutions for the nonlinear Schrödinger equation perturbed by third-order and
negative fourth-order dispersion, Phys. Rev. A 110 (2024) 043518,
https://doi.org/10.1103/PhysRevA.110.043518.

.. [3] O. Melchert, A. Demircan, https://doi.org/10.48550/arXiv.2504.10623.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from SWtools import SRM
from fmas.solver import LEM

# -- CONVENIENT ABBREVIATIONS
FT = np.fft.ifft
IFT = np.fft.fft
FTFREQ = np.fft.fftfreq

# -- SET UP DOMAIN AND MODEL 
xi = np.linspace(-50, 50, 2**13)
k = FTFREQ(xi.size, d=xi[1]-xi[0])*2*np.pi
c1, c2, c3, c4 = 0.458, -1/2, 1/12, -1/24
F = lambda I, xi: I
kap = 0.618

# -- SET UP AND SOLVE NEVP 
NEVP = SRM( xi, (c1,c2,c3,c4), F, verbose=True)
NEVP.solve(np.exp(-xi**2), kap)

# -- SET UP AND SOLVE IVP
u0 = FT(NEVP.U)*(1+np.exp(-5j*k+2.49826j))
Lk = 1j*(c2*k**2 + c3*k**3 + c4*k**4)
N = lambda U: F(np.abs(U)**2,xi)*U
Nk = lambda u: 1j*FT(N(IFT(u)))
IVP = LEM(Lk, Nk, del_G=1e-8)
IVP.set_initial_condition(k, u0)
IVP.propagate(
  z_range = 90.,
  n_steps = 1000,
  n_skip = 1
)

# -- POSTPROCESSING 
res = {
    'xi':  xi,
    'U0':  NEVP.U,
    'it':  NEVP.iter_list,
    'acc': NEVP.acc_list,
    'eta': IVP.z,
    'U':   IVP.utz,
    'del': IVP._del_rle,
    'h':   IVP._dz_a
}
np.savez_compressed('res.npz',**res)
