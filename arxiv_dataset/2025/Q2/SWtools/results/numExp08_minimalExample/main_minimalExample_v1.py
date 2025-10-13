"""main_minimalExample_v1.py

This module demonstrates the calculation of a solitary wave solution for a
higher-order nonlinear Schrödinger equation, modeled after [1]_. This example
is discussed as SWtools use-case at [2]_.

References
----------

.. [1] Widjaja et al., Absence of Galilean invariance for pure-quartic
solitons, Phys. Rev. A 104 (2021) 043526,
https://doi.org/10.1103/PhysRevA.104.043526.

.. [2] https://github.com/omelchert/SWtools

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>

"""
import numpy as np
from SWtools import SRM

myS = SRM(
  np.linspace(-10, 10, 2**10),  # COMPUTATIONAL DOMAIN
  (0, 0, 0, -0.0917),           # NEVP - COEFFS. LINEAR PART
  lambda I, xi: 4.07*I,         # NEVP - NONLINEAR FUNCTIONAL
  verbose = True                # NEVP - LIST DETAILS DURING ITERATION
)

myS.solve(
  np.exp(-myS.xi**2),           # SRM - INITIAL CONDITION
  1.76                          # SRM - EIGENVALUE OF SOUGHT-FOR SOLUTION
)

myS.show('fig_minimalExample_v1.png')
