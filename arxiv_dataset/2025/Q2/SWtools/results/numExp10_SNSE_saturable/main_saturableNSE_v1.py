"""main_saturableNSE_v1.py

This module demonstrates an instance of a solitary wave for a one-dimensional
nonlinear Schrödinger equation with saturable nonlinearity, modeled after [1]_.
This example is discussed as SWtools use-case in [2]_.

References
----------

.. [1] L. Falsi, A. Villois, F. Coppini, A. J. Agranat, E. DelRe, S. Trillo,
Evidence of 1 + 1D Photorefractive Stripe Solitons Deep in the Kerr Limit,
Phys. Rev. Lett. 133 (2024) 183804,
https://doi.org/10.1103/PhysRevLett.133.183804.

.. [2] O. Melchert, A. Demircan, https://doi.org/10.48550/arXiv.2504.10623.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from SWtools import NSOM

# -- MODEL PARAMETRS
N0 = 0.8

# -- INSTANTIATE NSOM
myS = NSOM(
  # ... COMPUTATIONAL DOMAIN
  np.linspace(-40, 40, 2**10),
  # ... COEFFICIENTS OF LINEAR PART
  (0, -1, 0, 0),
  # ... NONLINEAR FUNCTIONAL
  lambda I, xi: -1/(1+I)**2,
  # ... OVERRELAXATION PARAMETER
  ORP = 1.9,
  nskip = 1,
  maxiter = 50000,
  verbose = True
)

# -- SOLUTION PROCEDURE
myS.solve(
  # ... INITIAL CONDITION
  np.exp(-myS.xi**2),
  # ... NORM CONSTRAINT
  N0
)

# -- POSTPROCESSING
print(f"max(U) = {np.max(myS.U)}")
print(f"sqrt(1+kap) = {np.sqrt(1+myS.kap)}")
res = {
    'xi': myS.xi,
    'U': myS.U,
    'it': myS.iter_list,
    'acc': myS.acc_list,
    'kap': myS.kap,
    'N0': N0
}
np.savez_compressed('res_N0%lf.npz'%(N0),**res)
print(myS); myS.show()
