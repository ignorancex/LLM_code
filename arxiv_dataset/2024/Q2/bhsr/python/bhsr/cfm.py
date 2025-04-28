#####################################################################
#  Computations of the BHSR rates via the continued fraction method #
#####################################################################

import numpy as np

from numba import njit
from qnm.angular import sep_consts
from .constants import *
from .kerr_bh import rg

# Approximation for the eigenvalues of the spin-weighted spheroidal(!) functions

@njit("float64(uint8, int16, uint8)")
def h_seidel(l: int, m: int = 1, s: int = 0) -> float:
   """
   Helper function for alm_approx.

   Parameters:
      l (int): Orbital angular momentum quantum number.
      m (int, optional): Azimuthal quantum number (default: 1).
      s (int, optional): Spin of the boson (default: 0).

   Returns:
      float: The \f$h(l)\f$ function.

   Notes:
      - Eq. (8) in https://doi.org/10.1088/0264-9381/6/7/012
   """
   num = l*(l*l - m*m)
   denom = 2*(l-0.5)*(l+0.5)
   if s > 0:
      mabs = np.abs(m)
      s1 = max(mabs, s)
      s2 = m*s/max(mabs, s)
      l2 = l*l
      num = (l2 - s1*s1)*(l2 - s*s)*(l2 - s2*s2)
      denom *= l2*l
   return num/denom

@njit("float64(uint8, int16, uint8)")
def flm_seidel_2(l: int, m: int = 1, s: int = 0) -> float:
   """
   Helper function for alm_approx.

   Parameters:
      l (int): Orbital angular momentum quantum number.
      m (int, optional): Azimuthal quantum number (default: 1).
      s (int, optional): Spin of the boson (default: 0).

   Returns:
      float: The \f$_sf_2^{lm}\f$ contribution.

   Notes:
      - Eq. (10c) in https://doi.org/10.1088/0264-9381/6/7/012
   """
   return h_seidel(l+1, m, s) - h_seidel(l, m, s) - 1

@njit("float64(uint8, int16, uint8)")
def flm_seidel_4(l: int, m: int = 1, s: int = 0) -> float:
   """
   Helper function for alm_approx.

   Parameters:
      l (int): Orbital angular momentum quantum number.
      m (int, optional): Azimuthal quantum number (default: 1).
      s (int, optional): Spin of the boson (default: 0).

   Returns:
      float: The \f$_sf_4^{lm}\f$ contribution.

   Notes:
      - Eq. (10e) in https://doi.org/10.1088/0264-9381/6/7/012
   """
   hl = h_seidel(l, m, s)
   hlp1 = h_seidel(l+1, m, s)
   hlp2 = h_seidel(l+2, m, s)
   twol = 2*l
   l2 = l*l
   lm1 = l-1
   lp1 = l+1
   lp2 = l+2
   res = (hlp1 - lp2*hlp2/(twol+3))*hlp1/(2*lp1)
   res += (hlp1/lp1 - hl)*hl/twol
   res += lm1*h_seidel(l-1, m, s)*hl/(twol*(twol-1))
   if s > 0:
      lp2sq = lp2*lp2
      lm1sq = lm1*lm1
      lp1sq = lp1*lp1
      res += 4*( hlp1/(lp1sq*lp2sq) - hl/(l2*lm1sq))*m*m*pow(s,4)/(l2*lp1sq)
   return res

@njit("complex128(complex128, uint8, int16, uint8)")
def alm_approx(c: complex , l: int, m: int = 1, s: int = 0) -> complex:
   """
   The eigenvalues \f$A_{lm}\f$ of the spin-weighted spheroidal(!) functions.

   Parameters:
      l (int): Orbital angular momentum quantum number.
      m (int, optional): Azimuthal quantum number (default: 1).
      s (int, optional): Spin of the boson (default: 0).

   Returns:
      float: The \f$A_{lm}\f$ eigenvalues.

   Notes:
      - Eq. (7) in https://doi.org/10.1088/0264-9381/6/7/012 with \f$A_{lm} \equiv _sE_l^m - s(s+1)\f$.
   """
   fvals = [l*(l+1), flm_seidel_2(l, m, s), flm_seidel_4(l, m, s)]
   expansion = [f*pow(c,2*i) for i,f in enumerate(fvals)]
   return sum(expansion)

# Compute continued fraction based on arXiv:0705.2880

def angular_ev(omega: complex, mbh: float, astar: float, mu: float, l: int, m: int) -> complex:
   """
   Compute the angular eigenvalue of the spin-weighted spheroidal(!) functions.

   Parameters:
      omega (complex): The frequency of the perturbation in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      mu (float): Boson mass in eV.
      l (int): Orbital angular momentum quantum number.
      m (int): Azimuthal quantum number.

   Returns:
      complex: The angular eigenvalue of the spin-weighted spheroidal(!) functions.
   """
   c = rg(mbh)*astar*np.sqrt(omega*omega - mu*mu)
   if np.abs(c) > 3:
      # Need to use the 'qnm' package here
      # print("WARNING. |c| > 3 detected. Use qlm.")
      return np.sort(sep_consts(s=0, c=c, m=m, l_max=l))[-1]
   return alm_approx(c, l, m, s=0)

@njit("UniTuple(complex128, 7)(complex128, float64, float64, float64, complex128, uint8)")
def cfunctions(omega: complex, mbh: float, astar: float, mu: float, alm: complex, m: int) -> tuple[complex, ...]:
   """
   Computes numerical coefficients for the continued fraction method.

   Parameters:
      omega (complex): The frequency of the perturbation in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      mu (float): Boson mass in eV.
      alm (complex): The eigenvalue of the angular equation \f$\mathcal{A}_{nlm}\f$.
      m (int): Azimuthal quantum number.
   
   Returns:
      tuple[complex]: The numerical coefficients for the continued fraction method.
   
   Notes:
      - Eqs (40)-44 in https://arxiv.org/pdf/0705.2880.pdf
      - The \f$\mathcal{A}_{nlm}\f$ should be computed using the `angular_ev` function to allow for optimisation with numba.
   """
   a = astar
   a2 = a*a
   b = np.sqrt(1 - a2)
   om = rg(mbh)*omega
   om2 = om*om
   mu_r = rg(mbh)*mu
   mu2 = mu_r*mu_r
   # Choose appropriate sign for bound states
   q = np.sqrt(mu2 - om2)
   # cN2 = 0.75 + (2*(b+1)*om2 - (2*b+1)*mu2)/q
   q = -np.sign(q.real)*q
   cN1 = 4*b*q
   cN2 = 0.75 + (2*(b+1)*om2 - (2*b+1)*mu2)/q
   q2 = q*q
   x = (om - 0.5*m*a)/b
   y = 2j*(om + x)
   z = om - 1j*q
   z2 = z*z/q
   # Compute the numerical coefficients
   c0 = 1 - y
   c1 = -4 + 2*y + 4*(b + 1)*q - 2*(q2 + om2)/q
   c2 = 3 - y - 2*(q2 - om2)/q
   c3 = 2j*z*z2 + a2*q2 + 2j*m*a*q + (z2 + 1)*(2j*x + 2*b*q - 1) - alm
   c4 = z2*z2 + 2j*z2*(om - x)
   return c0, c1, c2, c3, c4, cN1, cN2

@njit("complex128(uint8, complex128)")
def alpha_n(n: int, c0: complex) -> complex:
   """
   The \f$\alpha_n\f$ coefficients in the continued fraction.

   Parameters:
      n (int): The index of the coefficient.
      d0 (complex): The \f$c_0\f$ coefficient.

   Returns:
      complex: The \f$\alpha_n\f$ coefficient.

   Notes:
      - Eq. (37) in https://arxiv.org/pdf/0705.2880.pdf
   """
   return n*n + (c0 + 1)*n + c0

@njit("complex128(uint8, complex128, complex128)")
def beta_n(n: int, d1: complex, d3: complex) -> complex:
   """
   The \f$\beta_n\f$ coefficients in the continued fraction.

   Parameters:
      n (int): The index of the coefficient.
      d1 (complex): The \f$c_1\f$ coefficient.
      d3 (complex): The \f$c_3\f$ coefficient.

   Returns:
      complex: The \f$\beta_n\f$ coefficient.

   Notes:
      - Eq. (38) in https://arxiv.org/pdf/0705.2880.pdf
   """
   return -2*n*n + (d1 + 2)*n + d3

@njit("complex128(uint8, complex128, complex128)")
def gamma_n(n: int, c2: complex, c4: complex) -> complex:
   """
   The \f$\gamma_n\f$ coefficients in the continued fraction.

   Parameters:
      n (int): The index of the coefficient.
      c2 (complex): The \f$c_2\f$ coefficient.
      c4 (complex): The \f$c_4\f$ coefficient.

   Returns:
      complex: The \f$\gamma_n\f$ coefficient.

   Notes:
      - Eq. (39) in https://arxiv.org/pdf/0705.2880.pdf
   """ 
   return n*n + (c2 - 3)*n + c4

@njit
def continued_fraction(omega: complex, mbh: float, astar: float, mu: float, alm: complex, m: int, nmax: int = 2000) -> complex:
   """
   Compute the continued fraction equation, whose complex roots are the frequencies.

   Parameters:
      omega (complex): The frequency of the perturbation in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      mu (float): Boson mass in eV.
      alm (complex): The eigenvalue of the angular equation \f$\mathcal{A}_{nlm}\f$.
      m (int): Azimuthal quantum number.
      nmax (int, optional): The maximum number of iterations (default: 2000).
   
   Returns:
      complex: The value of the continued fraction equation.

   Notes:
      - Eq. (48) in https://arxiv.org/pdf/0705.2880.pdf
      - We set the residual terms \f$f_N\f$ to zero; see Sec. II C in https://arxiv.org/pdf/1410.7698.pdf for alternatives.
   """
   c0, c1, c2, c3, c4, cN1, cN2 = cfunctions(omega, mbh, astar, mu, alm, m)
   fr = (-1+0j) + cN1/np.sqrt(nmax) + cN2/nmax # Improved residual term
   fr0 = beta_n(0, c1, c3)/alpha_n(0, c0)
   flipped_range = [nmax-i for i in range(nmax)]
   for i in flipped_range:
      alph = alpha_n(i, c0)
      beta = beta_n(i, c1, c3)
      gam = gamma_n(i, c2, c4)
      fr = gam/(beta - alph*fr)
   return fr0/fr - 1

def root_equation(omega: complex, mbh: float, astar: float, mu: float, l: int, m: int) -> float:
   """
   Helper function for the root equation

   Parameters:
      omega (complex): The frequency of the perturbation in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      mu (float): Boson mass in eV.
      l (int): Orbital angular momentum quantum number.
      m (int): Azimuthal quantum number.
   
   Returns:
      float: The value of the root equation.
   
   Notes:
      - This wrapper allows continued_fraction(...) to be optimised with numba.
   """
   alm = angular_ev(omega, mbh, astar, mu, l, m)
   z = continued_fraction(omega, mbh, astar, mu, alm, m)
   return np.log(z+1)

def min_equation(x: list[float, float], mbh: float, astar: float, mu: float, l: int, m: int) -> float:
   """
   Wrapper function for the root-finding algorithm.

   Parameters:
      x (list[float, float]): The real and imaginary parts of the frequency.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      mu (float): Boson mass in eV.
      l (int): Orbital angular momentum quantum number.
      m (int): Azimuthal quantum number.
   
   Returns:
      float: The value of the root equation.
   """
   omega = x[0] + 1j*x[1]
   z = root_equation(omega, mbh, astar, mu, l, m)
   return np.abs(z)