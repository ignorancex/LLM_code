"""main_CQNSE_v1.py

This module demonstrates an instance of a well-dressed repulsive-core soliton,
discussed in the context of a cubic-quintic nonlinear Schrödinger equation
(CQNSE) in [1]_. This example is discussed as SWtools use-case in [2]_.

References
----------

.. [1] V. N. Serkin, T. L. Belyaeva, Well-dressed repulsive-core solitons and
    nonlinear optics of nuclear reactions, Optics Communications 549 (2023)
    129831, https://doi.org/10.1016/j.optcom.2023.129831.

.. [2] O. Melchert, A. Demircan, https://doi.org/10.48550/arXiv.2504.10623.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from SWtools import NSOM

# -- MODEL PARAMETRS
ds = 0.00291225
alpha = 3*(1-ds**2)/8
N0 = 16.

# -- INSTANTIATE NSOM
myS = NSOM(
  # ... COMPUTATIONAL DOMAIN
  np.linspace(-40, 40, 2**12),
  # ... COEFFICIENTS OF LINEAR PART
  (0, -0.5, 0, 0),
  # ... NONLINEAR FUNCTIONAL
  lambda I, xi: I - alpha*I*I,
  # ... OVERRELAXATION PARAMETER
  ORP = 1.8,
  nskip = 10,
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
res = {
    'xi': myS.xi,
    'U': myS.U,
    'it': myS.iter_list,
    'acc': myS.acc_list,
    'N0': N0,
    'ds': ds,
    'alpha': alpha
}
np.savez_compressed('res.npz',**res)
print(myS); myS.show()
