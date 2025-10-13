"""main_GPE_excitedStates_v2.py

This module demonstrates the calculation of nonlinear excited states, based on
a quantum harmonic oscillator. This example is discussed as SWtools use-case in
[1]_.

References
----------
.. [1] O. Melchert, A. Demircan, https://doi.org/10.48550/arXiv.2504.10623.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from SWtools import NSOM

# -- SETUP AND INITIALIZATION 
xi = np.linspace(-8, 8, 512)
# ... NEVP INGREDIENTS 
cL = (0,-0.5,0,0)
F = lambda I, xi: -0.5*xi**2 - I
N0 = 1.0
# ... INITIAL CONDITIONS 
UI0 = np.exp(-xi**2/2)/np.pi**0.25
UI1 = 2*xi*UI0/np.sqrt(2)
UI2 = (4*xi**2-2)*UI0/np.sqrt(8)
UI3 = (8*xi**3-12*xi)*UI0/np.sqrt(48)
# ... INSTANTIATE NSOM
nevp = NSOM(xi, cL, F, tol=1e-12, ORP=1.5, maxiter=10000, nskip=10, verbose=True)

# -- NSOM SOLUTION PROCEDURE
def prev_sols(sol, cache = []):
  """Caches solution `sol.U`."""
  cache.append(sol.U)
  return cache

# ... GROUND STATE
nevp.solve(UI0, N0)
print(nevp); nevp.show()

# ... 1ST EXCITED STATE
nevp.solve(UI1, N0, ortho_set=prev_sols(nevp))
print(nevp); nevp.show()

# ... 2ND EXCITED STATE
nevp.solve(UI2, N0, ortho_set=prev_sols(nevp))
print(nevp); nevp.show()

# ... 3RD EXCITED STATE
nevp.solve(UI3, N0, ortho_set=prev_sols(nevp))
print(nevp); nevp.show()
