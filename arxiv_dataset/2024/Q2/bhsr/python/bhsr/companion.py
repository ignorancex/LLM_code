##############################################
#  Functions related to BH companion effects #
##############################################

import numpy as np

from numba import njit
from .constants import *
from .kerr_bh import *

F_COMP0 = 114*np.sqrt(3)*np.pi*np.pi

@njit
def f_companion_infall(mu: float, mbh: float, astar: float, mcomp: float, tau: float) -> float:
   """
   Calculates the size of the companion BH effect on the BHSR due to infalls.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      mcomp (float): Companion BH mass in Msol.
      tau (float): Period of the companion BH in yr.

   Returns:
      float: Companion infall factor (unitless).

   Notes:
      - Ref.: Eq. (I6) in https://arxiv.org/pdf/2011.11646.pdf
   """
   alph = alpha(mu, mbh)
   ratio = mcomp/mbh
   x = mu*tau/inv_eVyr
   return F_COMP0*ratio / ((alph ** 7) * astar * (1 + ratio) * x * x)

@njit
def f_companion_resonance(mu: float, mbh: float, astar: float, tau: float) -> float:
   """
   Calculates the size of the companion BH effect on the BHSR due to resonance.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      tau (float): Period of the companion BH in yr.

   Returns:
      float: Companion resonance factor.

   Notes:
      - Ref.: Eq. (I7) in https://arxiv.org/pdf/2011.11646.pdf
   """
   alph = alpha(mu, mbh)
   x = mu*tau/inv_eVyr
   return (alph ** 5) * astar * x / 6.0
   