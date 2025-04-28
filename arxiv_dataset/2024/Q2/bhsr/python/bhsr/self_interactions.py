###############################################################
#  Functions to consider the effects of ULB self-interactions #
###############################################################

import numpy as np

from .constants import *
from .kerr_bh import *
from .bhsr import *

### Routines for the bosenova scenario.

@njit("float64(float64, float64, float64, uint8)")
def n_bose(mu: float, invf: float, mbh: float, n: int = 2) -> float:
   """
   Calculates the number of bosons in a black hole superradiant cloud that triggers a bosenova.

   Parameters:
      mu (float): Boson mass in eV.
      invf (float): Inverse of the boson decay constant in GeV^-1.
      mbh (float): Black hole mass in Msol.
      n (int, optional): Principal quantum number (default: 2).

   Returns:
      float: Number of bosons in the superradiant cloud.

   Notes:
      - Ref.: Eq. (9) in https://arxiv.org/pdf/1411.2263.pdf, derived from https://arxiv.org/pdf/1203.5070.pdf
   """
   alph = alpha(mu, mbh)
   x = n*n*(mbh/10)*1.0/(invf*mPred_in_GeV)
   return 1e78*c0_n_bose*x*x/alph**3

@njit("float64(float64, float64, uint8)")
def bosenova_fcrit(mu: float, mbh: float, n: int = 2) -> float:
   """
   Calculates the critical boson decay constant at which a bosenova in a black hole superradiant cloud
   is expected to happen before (most of the) black hole spin has been depleted.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      n (int, optional): Principal quantum number (default: 2).

   Returns:
      float: Critical boson decay constant in GeV.

   Notes:
      - Computed by equating n_bose() with n_fin(..., da = 0.1)
   """
   da0 = 0.1
   f0 = 2e16 * pow(alpha(mu, mbh)/(0.4*n), 1.5) * np.sqrt( (da0/0.1) * (5.0/c0_n_bose) / n )
   return f0

@njit("boolean(float64, float64, float64, float64, uint8, uint8, float64)")
def not_bosenova_is_problem(mu: float, invf: float, mbh: float, tbh: float, n: int, m: int, sr_rate: float) -> bool:
   """
   Check if bosenovae do not pose a problem for the parameter constraints.

   Parameters:
      mu (float): Boson mass in eV.
      invf (float): Inverse of the boson decay constant in GeV^-1.
      mbh (float): Black hole mass in Msol.
      tbh (float): Black hole timescale in yr.
      n (int): Principal quantum number.
      m (int): Magnetic quantum number
      sr_rate (float): Superradiance rate in eV.

   Returns:
      bool: True if bosenovae do not pose a problem for the parameter constraints.
   """
   nm = n_fin(mbh, m=m)
   nb = n_bose(mu, invf, mbh, n)
   inv_tbh = inv_eVyr/tbh
   res = sr_rate > inv_tbh*np.log(nb)*(nm/nb)
   if np.isnan(res):
      res = 0
   return res

@njit("float64(float64, float64, float64, uint8, uint8)")
def eta_bn(mu: float, invf: float, mbh: float, n: int = 2, m: int = 1) -> float:
   """
   Calculates the effective reduction factor in the superradiance rate due to bosenovae.

   Parameters:
      mu (float): Boson mass in eV.
      invf (float): Inverse of the boson decay constant in GeV^-1.
      mbh (float): Black hole mass in Msol.
      n (int, optional): Principal quantum number (default: 2).
      m (int, optional): Magnetic quantum number (default: 1).

   Returns:
      float: Reduction factor in the superradiance rate.
   """
   nm = n_fin(mbh, m=m)
   nb = n_bose(mu, invf, mbh, n)
   return nm*np.log(nb)/nb

### Routines for the equilibrium regime

@njit
def GammaSR_322xBH_211x211(mu: float, mbh: float, astar: float, invf: float) -> float:
   """
   Calculates the rate factor that corresponds to the damping rate of forced ocillations in the boson cloud,
   from the interacting |211> and |322> states and the BH.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      invf (float): Inverse boson decay constant in GeV^-1.

   Returns:
      float: The rate factor in eV.

   Notes:
      - Ref.: Table I, https://arxiv.org/pdf/2011.11646.pdf
   """
   al = alpha(mu, mbh)
   return 4.3e-7 * mu*(1.0 + np.sqrt(1.0-astar*astar))*pow(al, 11)*pow(mP_in_GeV*invf, 4)

@njit
def GammaSR_211xinf_322x322(mu: float, mbh: float, invf: float) -> float:
   """
   Calculates the rate factor that corresponds to scalar emissions from the interacting |211> and |322> states
   to infinity.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      invf (float): Inverse boson decay constant in GeV^-1.

   Returns:
      float: The rate factor in eV.

   Notes:
      - Ref.: Table I, https://arxiv.org/pdf/2011.11646.pdf
   """
   al = alpha(mu, mbh)
   return 1.1e-8 * mu*pow(al, 8)*pow(mP_in_GeV*invf, 4)

@njit
def n_eq_211(mu: float, mbh: float, astar: float, invf: float, sr_0_211: float) -> float:
   """
   Calculates the equilibrium occupation number of the |211> state in a self-interacting boson cloud.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      invf (float): Inverse boson decay constant in GeV^-1.
      sr_0_211 (float): Superradiance rate for the |211> level.

   Returns:
      float: The equilibrium occupation number.

   Notes:
      - Ref.: Eq. (55a) in https://arxiv.org/pdf/2011.11646.pdf
   """
   sr_3b22 = GammaSR_322xBH_211x211(mu, mbh, astar, invf)
   sr_2i33 = GammaSR_211xinf_322x322(mu, mbh, invf)
   return 2*np.sqrt(sr_0_211*sr_2i33/3.0)/sr_3b22

def GammaSR_nlm_eq(mu: float, mbh: float, astar: float, invf: float, n: int = 2, l: int = 1, m: int = 1, sr_function: callable = GammaSR_nlm_bxzh) -> float:
   """
   Calculates the effective SR rate of the |211> state in a self-interacting boson cloud.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      invf (float): Inverse of the boson decay constant in GeV.
      n (int): Principal quantum number (currently only n = 2).
      l (int): Orbital angular momentum quantum number (currently only l = 1).
      m (int): Magnetic quantum number (currently only m = 1).

   Returns:
      tuple(float): The equilibrium BHSR rate, the corresponding non-interacting rate (in eV).
   """
   if n != 2 or l != 1 or m != 1:
      raise ValueError("Only the |nlm> = |211> rate is currently supported for the equilibrium regime and GammaSR_nlm_eq(...).")
   sr0 = sr_function(mu, mbh, astar, n, l, m)
   neq = n_eq_211(mu, mbh, astar, invf, sr0)
   return neq*sr0, sr0