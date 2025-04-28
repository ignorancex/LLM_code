######################
#  Kerr BH functions #
######################

import numpy as np

from numba import njit
from .constants import GNewton, inv_eVyr, Msol_in_eV

@njit
def mbh_in_eV(mbh: float) -> float:
   """
   Convert black hole mass from Msol to eV.

   Parameters:
      mbh (float): Black hole mass in Msol.

   Returns:
      float: Black hole mass in eV.
   """
   return mbh*Msol_in_eV

@njit
def rg(mbh: float) -> float:
   """
   Calculate the "gravitational radius" of a Kerr black hole

   Parameters:
      mbh (float): Black hole mass in Msol.

   Returns:
      float: The gravitational radius of the black hole in eV^-1.
   """
   return GNewton*mbh

@njit
def alpha(mu: float, mbh: float) -> float:
   """
   Compute the dimensionless coupling constant alpha.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.

   Returns:
      float: The dimensionless coupling constant alpha.
   """
   return rg(mbh)*mu

@njit
def r_plus(mbh: float, astar: float) -> float:
   """
   Calculates the radius of the inner event horizon of a Kerr black hole.

   Parameters:
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.

   Returns:
      float: The radius of the inner event horizon in eV^-1.
   """
   return rg(mbh)*(1 + np.sqrt(1 - astar*astar))

@njit
def r_minus(mbh: float, astar: float) -> float:
   """
   Calculates the radius of the outer event horizon of a Kerr black hole.

   Parameters:
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.

   Returns:
      float: The radius of the outer event horizon in eV^-1.
   """
   return rg(mbh)*(1 - np.sqrt(1 - astar*astar))

@njit
def omH(mbh: float, astar: float) -> float:
   """
   Calculate the angular velocity of the event horizon of a Kerr black hole.

   Parameters:
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.

   Returns:
      float: The angular velocity of the event horizon in eV.
   """
   return 0.5*astar/r_plus(mbh, astar)

# TODO: -> (3n^2 - l*(l+1)) M / (2 mu alpha) ~ n^2 M / alpha^2 for l = n - 1
@njit
def r_cloud(mu: float, mbh: float, n: int) -> float:
   """
   Calculate the approximate boson "could distance" from a BH.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      n (int): Principal quantum number.

   Returns:
      float: The approximate boson "could distance" in eV^-1.

   Notes:
      - Ref.: Eq. (4) https://arxiv.org/pdf/1411.2263.pdf
   """
   x = n/alpha(mu, mbh)
   return x*x*rg(mbh)


@njit
def t_infall(mu: float, mbh: float, n: int = 2) -> float:
   """
   Calculate the approximate boson could infall time.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      n (int): Principal quantum number (default: 2).

   Returns:
      float: The approximate boson could infall time in years.
   """
   if n != 2:
      raise ValueError("Only n = 2 is currently supported.")
   c_bn = 16.0
   rc = r_cloud(mu, mbh, n)
   return c_bn*inv_eVyr*rc