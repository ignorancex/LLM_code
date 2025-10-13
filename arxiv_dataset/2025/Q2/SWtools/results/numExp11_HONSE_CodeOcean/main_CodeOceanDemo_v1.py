"""main_CodeOceanDemo_v1.py

This module demonstrates the calculation of a solitary wave solution for a
higher-order nonlinear Schrödinger equation, modeled after [1]_. This example
forms the basis of the verification test discussed in section 4.1 of [2]_.

References
----------

.. [1] M. Karlsson, A. Höök, Soliton-like pulses governed by fourth order
dispersion in optical fibers, Optics Communications 104 (1994) 303,
https://doi.org/10.1016/0030-4018(94)90560-6.

.. [2] O. Melchert, A. Demircan, https://doi.org/10.48550/arXiv.2504.10623.

.. codeauthor:: Oliver Melchert <melchert@iqo.uni-hannover.de>
"""
import numpy as np
from scipy.fftpack import diff
from SWtools import SRM


def main():

    # -- SETUP AND INITIALIZATION 
    xi = np.linspace(-20, 20, 2**9)         # ... COMPUTATIONAL DOMAIN
    cL = (0.1, -1/2, 0., -1/24)             # ... LINEAR COEFFICIENTS
    kap = 0.96                              # ... SOLITON EIGENVALUE 
    F = lambda I, xi: I                     # ... NONLINEAR FUNCTIONAL
    U0 = np.exp(-xi**2)                     # ... INITIAL CONDITION
    nevp = SRM(xi, cL, F, verbose=True)     # ... INITIALIZE NEVP

    # -- SOLVE NEVP
    nevp.solve(U0, kap)

    # -- POSTPROCESSING
    # ... CALCULATE MOMENTUM VIA SPECTRAL DERIVATIVE
    U = nevp.U
    P = np.imag(np.trapz(np.conj(U)*diff(U, period=xi[-1]-xi[0]), x=xi))
    # ... LIST PROPERTIES
    print(f"N[U] = {nevp.N}")   # ... NORM
    print(f"H[U] = {nevp.H}")   # ... HAMILTONIAN
    print(f"P[U] = {P}")        # ... MOMENTUM
    # ... GENERATE FIGURE
    nevp.show()


if __name__=="__main__":
    main()
