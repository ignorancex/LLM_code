#####################################
#  BHSR rates and related functions #
#####################################

import numpy as np
import warnings

from fractions import Fraction
from iminuit import Minuit
from math import factorial
from numba import njit
from scipy.optimize import root_scalar
from scipy.special import gamma
from superrad.ultralight_boson import UltralightBoson

from .cfm import *
from .constants import *
from .kerr_bh import *

## Energy levels

@njit("float64(float64, float64, uint8)")
def omegaLO(mu: float, mbh: float, n: int) -> float:
   """
   Calculates the leading-order frequency of the superradiant mode for a given set of quantum numbers |n,l,m>.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      n (int): Principal quantum number.
      l (int): Orbital angular momentum quantum number (currently not used).
      m (int): Magnetic quantum number (currently not used).

   Returns:
      float: The frequency of the superradiant mode in eV.
   """
   x = alpha(mu, mbh)/n
   return mu*(1.0 - 0.5*x*x)

@njit("float64(float64, float64, float64, uint8, uint8, int16)")
def omegaHyperfine(mu: float, mbh: float, astar: float, n: int, l: int, m: int) -> float:
   """
   Calculates the  hyperfine frequency of the superradiant mode for a given set of quantum numbers quantum numbers |n,l,m>.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      n (int): Principal quantum number.
      l (int): Orbital angular momentum quantum number (currently not used).
      m (int): Magnetic quantum number (currently not used).

   Returns:
      float: The frequency of the superradiant mode in eV.

   Notes:
      - See Eq. (2.28) in https://arxiv.org/pdf/1908.10370.pdf
   """
   x = alpha(mu, mbh)/n
   x2 = x*x
   x4 = x2*x2
   fine = 1.875 - 6.0*n/(2*l+1) # = 2 - 1/8 - ...
   hyperfine = 8.0*m*n*n*astar/(l*(2*l+1)*(2*l+2))
   return mu*(1.0 - 0.5*x2 + fine*x4 + hyperfine*x*x4)

## BHSR rates using the "non-relativistic approximation"

def c_nl_factor(n: int, l: int) -> Fraction:
   """
   Helper function for computing a factor in GammaSR_nlm_nr.

   Parameters:
      n (int): Principal quantum number.
      l (int): Orbital angular momentum quantum number.

   Returns:
      Fraction: Multiplication factor.

   Notes:
      - The alternative quantum number n', with n = n' + l + 1, is used in some publications.
   """
   x = Fraction(factorial(n+l) * pow(2, 4*l+2), factorial(n-l-1) * pow(n, 2*l+4))
   y = Fraction(factorial(l), factorial(2*l+1)*factorial(2*l))
   return x*y*y

# Pre-compute c_nl for 2 <= n < 10 and l < n
c_nl_float = [[float(c_nl_factor(n,l)) for l in range(1, n)] for n in range(2, 10)]

def GammaSR_nlm_nr(mu: float, mbh: float, astar: float, n: int = 2, l: int = 1, m: int = 1) -> float:
   """
   Calculate the superradiance rate for a bosonic cloud around a Kerr black hole,
   using the "non-realtivisic approximation."

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter (astar = a/rg).
      n (int): Principal quantum number (n < 10 for numerical efficency).
      l (int): Orbital angular momentum quantum number.
      m (int): Magnetic quantum number.

   Returns:
      float: Superradiance rate in eV

   Notes:
      - See Eq. (18) in https://arxiv.org/pdf/1004.3558.pdf
      - Fig. 5 in https://arxiv.org/pdf/1004.3558.pdf
      - The authors of https://arxiv.org/pdf/1004.3558.pdf also use a semi-analytical method (see also https://arxiv.org/pdf/0912.1780.pdf)
   """
   if not(2 <= n < 10 and l < n):
      raise ValueError(f"Invalid quantum numbers for GammaSR_nlm_bxzh: |{m:d},{l:d},{m:d}> while only 2 <= n < 10, l < n are supported.")
   al = alpha(mu, mbh)
   x = 1.0 - astar*astar
   murp = al*(1 + np.sqrt(x)) # = mu*rg*rp
   y = m*astar - 2*murp
   c_nl = c_nl_float[n-2][l-1]
   factors = np.prod([(k*k)*x + y*y for k in range(1,l+1)], axis=0)
   return mu*y*c_nl*factors*pow(al, 4*l+4)

## BHSR rates using higher-order corrections

@njit("float64(float64, float64, uint8)")
def omega0_bxzh(mu: float, mbh: float, n: int) -> float:
   """
   Helper function to compute the leading-order (LO) frequency of the superradiant mode.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      n (int): Principal quantum number.

   Returns:
      float: Contribution to the LO frequency of the superradiant mode in eV.

   Notes:
      - See Eq. (16a) in https://arxiv.org/pdf/2201.10941.pdf
   """
   n2 = n*n
   al = alpha(mu, mbh)
   al2 = al*al
   x = 2*al2/(n2 + 4*al2 + n*np.sqrt(n2 + 8*al2))
   return mu*np.sqrt(1.0 - x)

@njit("float64(float64, float64, uint8)")
def omega1_bxzh(mu: float, mbh: float, n: int) -> float:
   """
   Helper function to compute the leading-order (LO) frequency of the superradiant mode.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      n (int): Principal quantum number.

   Returns:
      float: Contribution to the LO frequency of the superradiant mode in eV.

   Notes:
      - See Eq. (16b) in https://arxiv.org/pdf/2201.10941.pdf
   """
   om0 = omega0_bxzh(mu, mbh, n)
   if om0 > 0:
      om02 = om0*om0
      mu2 = mu*mu
      al = alpha(mu, mbh)
      al2 = al*al
      x = 1.0 + 4*al2*(2*om02/mu2 - 1.0)/(n*n)
      return (mu2 - om02)/(n*om0*x)
   return 0

def c_nl_bxzh(n: int, l: int) -> Fraction:
   """
   Factor used in the computation of the higher-order SR rate computation.

   Parameters:
      n (int): Principal quantum number.
      l (int): Orbital angular momentum quantum number.

   Returns:
      Fraction: Multiplication factor.

   Notes:
      - Compare to Eq. (15) in https://arxiv.org/pdf/2201.10941.pdf
   """
   x = Fraction(factorial(n+l) * pow(2, 4*l+2), factorial(n-l-1))
   y = Fraction(factorial(l), factorial(2*l+1)*factorial(2*l))
   return x*y*y

# Pre-compute c_nl for 2 <= n < 10 and l < n
c_nl_bxzh_float = [[float(c_nl_bxzh(n,l)) for l in range(1, n)] for n in range(2, 10)]

def GammaSR_nlm_bxzh(mu: float, mbh: float, astar: float, n: int = 2, l: int = 1, m: int = 1) -> float:
   """
   Compute the higher-order SR rate, following https://arxiv.org/pdf/2201.10941.pdf.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      n (int): Principal quantum number (default: n = 2).
      l (int): Orbital angular momentum quantum number (default: l = 1)
      m (int): Magnetic quantum number (default: m = 1)

   Returns:
      float: Superradiance rate in eV.
   """
   if not(n >= 2 and n < 10 and l < n):
      raise ValueError(f"Invalid quantum numbers for GammaSR_nlm_bxzh: |{n:d},{l:d},{m:d}> while only 2 <= n < 10, l < n are supported.")
   om0 = omega0_bxzh(mu, mbh, n)
   al = alpha(mu, mbh)
   z = pow(al*al*(1.0-om0*om0/(mu*mu)), l+0.5)
   om1 = omega1_bxzh(mu, mbh, n)
   c_nl = c_nl_bxzh_float[n-2][l-1]
   x = 1 - astar*astar
   rp = r_plus(mbh, astar)
   y = m*astar - 2*rp*om0
   factors = np.prod([k*k*x + y*y for k in range(1,l+1)], axis=0)
   return y*z*c_nl*factors*om1

def gam_pq_bxzh(p: float, q: float, eps: float, n: int, l: int) -> float:
   """
   Helper function to compute the next-to-leading-order (NLO) energy level of the superradiant mode.

   Parameters:
      p (float): Parameter p.
      q (float): Parameter q.
      eps (float): Parameter epsilon.
      n (int): Principal quantum number.
      l (int): Orbital angular momentum quantum number.

   Returns:
      float: Dimensionless facotor.

   Notes:
      - See Eqs. (22, 23) in https://arxiv.org/pdf/2201.10941.pdf
   """
   lp = l + eps
   ip = p*1j
   twolp = 2*lp
   g1 = gamma(twolp + 1)
   g2 = gamma(twolp + 2)
   g2n = gamma(twolp + 1 + n - l)
   gpmeps = gamma(1 + 2*eps)*gamma(1 - 2*eps)
   x1 = lp + 1 + ip
   x2 = np.sqrt(q - p*p + 0j)
   gabs = np.abs(gamma(x1 + x2)*gamma(x1 - x2))
   gmix = gamma(1.0 - ip - eps + x2)*gamma(1.0 - ip - eps - x2)
   gmix *= gamma(1.0 + ip + eps + x2)*gamma(1.0 + ip + eps - x2)
   num = g2n*gpmeps*gabs*gabs*pow(2, 4*lp + 2)
   denom = factorial(n-l-1)*g1*g1*g2*g2*gmix
   return num/denom

def omega_nlm_bxzh(mu: float, mbh: float, astar: float, n: int = 2, l: int = 1, m: int = 1) -> tuple[float, float]:
   """
   Compute the complex valued frequency of the superradiant mode for the |n,l,m> state, following https://arxiv.org/pdf/2201.10941.pdf.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      n (int, optional): Principal quantum number (default: n = 2).
      l (int, optional): Orbital angular momentum quantum number (default: l = 1)
      m (int, optional): Magnetic quantum number (default: m = 1)

   Returns:
      tuple[float, float]: Real and imaginary parts of the frequency in eV.
   """
   om0 = omega0_bxzh(mu, mbh, n)
   om1 = omega1_bxzh(mu, mbh, n)
   al = alpha(mu, mbh)
   al2 = al*al
   eps = -8.0*al2/(2*l+1)
   lp = l + eps
   if lp < 0:
      return 0, 0
   rp = r_plus(mbh, astar)
   rG = rg(mbh)
   x = np.sqrt(1.0 - astar*astar)
   y = mu*mu - om0*om0
   p = -0.5*(m*astar - 2.0*rp*om0)/x
   q = 4*om0*p*rG - 2*(3.0 - x)*al2
   gam_terms = gam_pq_bxzh(p, q, eps, n, l)
   kappab_term = pow(rG*rG*x*x*y, lp+0.5)
   delta1 = 0.5*(q/eps - eps - p*2j)*kappab_term*gam_terms
   om = om0 + (eps + delta1)*om1
   return om.real, om.imag

## BHSR rates from superrad Python package

bc0 = UltralightBoson(spin=0, model="relativistic")
def GammaSR_nlm_superrad(mu: float, mbh: float, astar: float, n: int = 2, l: int = 1, m: int = 1, bc: UltralightBoson = bc0) -> float:
   """
   Calculate the superradiance rate for a bosonic cloud around a Kerr black hole,
   using the superrad Python package (https://bitbucket.org/weast/superrad)

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      n (int): Principal quantum number (only n = 2).
      l (int): Orbital angular momentum quantum number (only l = m)
      m (int): Magnetic quantum number (only m = 1, 2)
      bc (UltralightBoson, optional): Type of boson to consider (default: relativistic scalar)

   Returns:
      float: Superradiance rate in eV

   Notes:
      - Ref.: https://arxiv.org/pdf/2211.03845.pdf
   """
   if (n!=2) and (l!=m) and (m!=1) and (m!=2):
      raise ValueError(f"The state |{n:d}{l:d}{m:d}> is currently not supported by SuperRad. Set n = 2 and l = m = 1,2.")
   try:
      """
      From the SuperRad documentation:
      evo_type can take values evo_type="full" or evo_type="matched".
      The "matched" evolution assumes that the boson cloud decays solely through gravitational radiation after reach its peak mass (by definition, t=0), and matches this on to a exponentially growing solution with constant growth rate before the peak mass is obtained (t<0).
      Hence, it is not very accurate around t=0, and in particular the derivative of the frequency will be discontinuous at this point.
      The "full" evolution option numerically integrates the ordinary differential equations describing the cloud mass and black hole mass and spin, and hence is more accurate for scenarios when the signal before the time when the cloud reaches saturation makes a non-negligible contribution.
      However, it is significantly more computationally expensive, and the numerical integration is not always robust.
      This option should currently be considered experimental.
      """
      # wf = bc.make_waveform(mbh, astar, mu, units="physical", evo_type="matched")
      # return inv_eVs/wf.efold_time()
      al = alpha(mu, mbh)
      rG = rg(mbh)
      return bc._cloud_model.omega_imag(m, al, astar)/rG
   except ValueError:
      return 0

@njit
def n_fin(mbh: float, da: float = 0.1, m: int = 1) -> float:
   """
   Calculate the threshold value for spin of the boson cloud occupation number for a given spin reduction.

   Parameters:
      mbh (float): Black hole mass in Msol.
      da (float, optional): Difference of dimensionless initial and final spin (default: 0.1).
      m (int, optional): Magnetic quantum number (default: 1).

   Returns:
      float: The maximum value of the boson occupation number.

   Notes:
      - Ref. Eq. (8) in https://arxiv.org/pdf/1411.2263.pdf
   """
   mbh_rel = mbh/10
   return 1e76 * (da/0.1) * mbh_rel*mbh_rel / m

@njit("boolean(float64, float64, float64)")
def can_grow_max_cloud(mbh: float, tbh: float, sr_rate: float):
   nm = n_fin(mbh)
   inv_tbh = inv_eVyr/tbh
   res = sr_rate > inv_tbh*np.log(nm)
   if np.isnan(res):
      res = 0
   return res
   
## Regge slope

def find_cf_root(mbh: float, astar: float, mu: float, n: int = 2, l: int = 1, m: int = 1, verbose: bool = False) -> complex:
   """
   Root-finding algorithm for the continued fraction method.
   
   Parameters:
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      mu (float): Boson mass in eV.
      n (int, optional): Principal quantum number (default: n = 2).
      l (int, optional): Orbital angular momentum quantum number (default: l = 1)
      m (int, optional): Magnetic quantum number (default: m = 1)
      verbose (bool, optional): Print additional information (default: False).

   Returns:
      complex: Complex frequency of the level in eV.
   """
   alph = alpha(mu, mbh)
   omR = omegaHyperfine(mu, mbh, astar, n, l, m)
   _, omI = omega_nlm_bxzh(mu, mbh, astar, n, l, m)
   if omR > 0 and omI > 0 and alph > 0:
      cost_oR = lambda x: np.log(np.abs(root_equation(x+1j*omI, mbh, astar, mu, l, m)))
      mR = Minuit(cost_oR, x=omR)
      mR.tol = 1e-10
      factor = 0.5*alph*alph
      om0 = mu*(1 - 0.5*factor*( 1.0/((n-1)*(n-1)) + 1.0/(n*n) ))
      om1 = mu*(1 - factor/(n*n))
      mR.limits["x"] = (om0, om1)
      mR.migrad()
      cost = lambda x, lgy: np.abs(root_equation(x+1j*pow(10,lgy), mbh, astar, mu, l, m))
      mRI = Minuit(cost, x=mR.values["x"], lgy=np.log10(omI))
      mRI.tol = 1e-10
      mRI.limits["x"] = (om0, om1)
      mRI.limits["lgy"] = (np.log10(0.7*omI), min(np.log10(10*omI), np.log10(0.1*omR)))
      mRI.migrad()
      om = mRI.values["x"] + 1j*pow(10, mRI.values["lgy"])
   else:
      if verbose:
         print("Estimates for real or imaginary part of omega are not positive:", om)
      return omR + 1j*omI
   return om

def GammaSR_nlm_cfm(mu: float, mbh: float, astar: float, n: int = 2, l: int = 1, m: int = 1) -> float:
   """
   Calculate the superradiance rate using the continued fraction method.

   Parameters:
      mu (float): Boson mass in eV.
      mbh (float): Black hole mass in Msol.
      astar (float): Dimensionless black hole spin parameter.
      n (int): Principal quantum number.
      l (int): Orbital angular momentum quantum number.
      m (int): Magnetic quantum number.

   Returns:
      float: Superradiance rate in eV.
   """
   om = find_cf_root(mbh, astar, mu, n, l, m)
   return om.imag

def compute_regge_slopes(mu: float, mbh_vals: list[float], states: list[tuple[int, int, int]], sr_function: callable = GammaSR_nlm_nr, inv_tSR: float = inv_tSR) -> np.ndarray:
   """
   Compute the Regge trajectories for several levels.

   Parameters:
      mu (float): Boson mass in eV.
      mbh_vals (list[float]): List of black hole masses in Msol.
      states (list[tuple[int, int, int]]): List of quantum states (n, l, m).
      inv_tSR (float, optional): Inverse of the superradiance timescale (default: inv_tSR).

   Returns:
      np.ndarray: Array of dimensionless black hole spins corresponding to mbh_vals.

   Notes:
      - The Regge slope is defined as the minimum value of the dimensionless spin parameter 'a' at which the superradiance rate equals the inverse of the superradiance timescale.
      - Note that the root finding may fail if no Regge slope exists. In this case, we set the corresponding BH spin value = NAN.
   """
   a_min_vals = []
   foo = lambda a, mbh, n, l, m: sr_function(mu, mbh, a, n, l, m) - inv_tSR
   for mbh in mbh_vals:
      temp = []
      for s in states:
         with warnings.catch_warnings(record=True) as w:
            res = root_scalar(foo, x0=0, x1=0.5, args=(mbh, *s))
            a_root = res.root if len(w) == 0 else np.nan
            temp.append(a_root)
      a_min_vals.append(temp)
   return np.array(a_min_vals)