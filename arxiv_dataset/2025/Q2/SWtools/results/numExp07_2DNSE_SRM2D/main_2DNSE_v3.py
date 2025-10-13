"""main_2DNSE_v3.py

This module demonstrates the calculation of a solitary wave for a
two-dimensional (2D) nonlinear Schrödinger equation with periodic potential,
modeled after [1]_ [2]_. As a technical basis it uses the 2D SRM implemented
with SWtools [3]_ [4]_.

References
----------

.. [1] J. Yang, Z. H. Musslimani, Fundamental and vortex solitons in a
two-dimensional optical lattice, Opt. Lett. 28 (21) (2003) 2094–2096,
https://doi.org/10.1364/JOSAB.21.000973.

.. [2] Z. Musslimani, J. Yang, Self-trapping of light in a two-dimensional
photonic lattice, J. Opt. Soc. Am. B 21 (2004) 973,
https://doi.org/10.1364/JOSAB.21.000973.

.. [3] O. Melchert, A. Demircan, https://doi.org/10.48550/arXiv.2504.10623.

.. [4] https://github.com/omelchert/SWtools

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from SWtools import SRM2D

# -- SETUP AND INITIALIZATION 
x1 = np.linspace(-20, 20, 2**8)
x2 = np.linspace(-20, 20, 2**8)
X1, X2 = np.meshgrid(x1, x2, indexing='ij')
# ... NEVP INGREDIENTS 
cL = (-1., -1.)
F = lambda I, xi: \
  3*(np.cos(xi[0])**2+np.cos(xi[1])**2) + I
kap = 3.7045
# ... INITIAL CONDITION 
U0 = np.exp(-(X1**2+X2**2))
# ... SRM INSTANTIATION
myS = SRM2D((X1, X2), cL, F, tol=1e-12, verbose=True, nskip=100)

# -- 2DSRM SOLUTION PROCEDURE
myS.solve(U0, kap)

# -- POSTPROCESSING
results = {
    'xi': myS.xi,
    'U': myS.U,
    'iter_list': myS.iter_list,
    'acc_list': myS.acc_list,
}
np.savez_compressed('./res_2DNSE_SRM2D_kap%lf'%(kap), **results)

print(f"max(U) = {np.max(myS.U):5.4F}")
print(myS); myS.show()
