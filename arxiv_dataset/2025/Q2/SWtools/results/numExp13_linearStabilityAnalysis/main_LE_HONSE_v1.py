"""main_LE_HONSE_v1.py

This script demonstrates the linear stability analysis for a higher-order
nonlinear Schrödinger equation with quartic propagation constant, discussed in
[1]_. The example reproduces the linear eigenspectrum (LE) of the generalized
dispersion Kerr soliton in Sect. 8 and Fig. 9 of Ref. [2]_. This example is
discussed as SWtools use-case in [3]_.

References
----------

.. [1] O. Melchert, A. Demircan, "Optical Solitary Wavelets",
https://doi.org/10.48550/arXiv.2410.06867.

.. [2] K. K. K. Tam, T. J. Alexander, A. Blanco-Redondo, C. M. de Sterke,
Generalized dispersion Kerr solitons, Phys. Rev. A 101 (2020) 043822,
https://doi.org/10.1103/PhysRevA.101.043822.

.. [3] O. Melchert, A. Demircan, https://doi.org/10.48550/arXiv.2504.10623.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from SWtools import SRM, LE_HONSE, LE_dump

# -- SET UP DOMAIN AND MODEL 
xi = np.linspace(-25, 25, 2**10)
c2, c3, c4 = 0.35, 0., -0.04167
kap = 1.

# -- DETERMINE SW SOLUTION
NEVP = SRM(xi, (0,c2,c3,c4), lambda I, xi: I)
NEVP.solve(np.exp(-xi**2/2), kap)

# -- PERFORM LSA
res = LE_HONSE(xi, NEVP.U, kap, (c2,c3,c4))

# -- POSTPROCESS RESULTS
LE_dump(*res)
Lam_max, Lam, f, g = res
res = {
    'xi': xi,
    'U': NEVP.U,
    'betas': [c2,c3,c4],
    'kap': kap,
    'Lam_max': Lam_max,
    'Lam': Lam,
    'f': f,
    'g': g
}
np.savez_compressed('res_LE_HONSE.npz', **res)
