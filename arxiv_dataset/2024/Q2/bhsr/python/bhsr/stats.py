#####################################
#  Utility functions for statistics #
#####################################

import numpy as np

from numba import njit
from .bhsr import GammaSR_nlm_bxzh, n_fin
from .self_interactions import *
from .constants import *
from .kerr_bh import alpha

@njit
def cdf_1d(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
   """
   Utility function to compute the cumulative distribution function (CDF) of a 1D array of data.

   Parameters:
      data (np.ndarray): 1D array of samples from the distribution.

   Returns:
      tuple(np.ndarray, np.ndarray): Sorted array of sample values, the corresponding CDF values.
   """
   cdf = []
   cdf_sum = 0
   if data.ndim == 1:
      avals, weights = np.sort(data), np.array(len(data)*[1], dtype='i')
   else:
      indices = np.argsort(data[:,0])
      dat = data[indices]
      avals, weights = dat[:,0], dat[:,1]
   for w in weights:
      cdf_sum += w
      cdf.append(cdf_sum)
   cdf = np.array(cdf, dtype='f')
   cdf /= cdf[-1]
   return avals, cdf

def p_mc_int_no_f(mu: float, samples: np.ndarray[(any,2), float], states: list[tuple[int,int,int]] = [(2,1,1)], tbh: float = tSR_in_yr, sr_function: callable = GammaSR_nlm_bxzh):
   """
   Monte Carlo integration to compute the marginal likelihood of the ULB's presence with Monte Carlo integration.
   This functions assumes that the boson does not have self-interactions.

   Parameters:
      mu (float): Boson mass in eV.
      samples (np.ndarray): Two-dimenstional array of samples from the (M,a*) posterior distribution.
      states (list[tuple[int,int,int]], optional): List of quantum numbers |n,l,m> to consider (default: [(2,1,1)]).
      tbh (float, optional): Black hole superradiance timescale in years (default: tSR_in_yr).
      sr_function (callable, optional): Function to compute the superradiance rate (default: GammaSR_nlm_bxzh).

   Returns:
      float: Marginal likelihood of the ultralight boson's existence.
   """
   inv_tbh = inv_eVyr/tbh
   n_samples = len(samples[:,0])
   p = n_samples
   for mbh, astar in samples:
      for s in states:
         n, l, m = s
         alph = alpha(mu, mbh)
         # Check if SR mode
         if alph/l <= 0.5:
            nfi = n_fin(mbh, m=m)
            srr = sr_function(mu, mbh, astar, n, l, m)
            if srr > np.log(nfi)*inv_tbh:
               p -= 1
               break
   return p/float(n_samples)

def p_mc_int_eq(mu: float, invf: float, samples: np.ndarray[(any,2), float], states: list[tuple[int,int,int]] = [(2,1,1)], tbh: float = tSR_in_yr, sr_function: callable = GammaSR_nlm_bxzh):
   if states != [(2,1,1)]:
      raise ValueError("Only the |nlm> = |211> level is currently supported.")
   inv_tbh = inv_eVyr/tbh
   n_samples = len(samples[:,0])
   p = n_samples
   for mbh, astar in samples:
      for s in states:
         n, l, m = s
         alph = alpha(mu, mbh)
         # Check if SR mode
         if alph/l <= 0.5:
            # Only now (for efficiency) compute BHSR rate and check if it is fast enough
            srr, srr0 = GammaSR_nlm_eq(mu, mbh, astar, invf, n, l, m, sr_function)
            nfi = n_fin(mbh, m=m)
            # Check if we reach the equilibrium regime, and the equilibrium rate is fast enough
            if srr > inv_tbh and srr0 > np.log(nfi)*inv_tbh:
               p -= 1
               break
   return p/float(n_samples)

def mc_integration_bosenova(mu: float, invf: float, samples: np.ndarray[(any,2), float], states: list[tuple[int,int,int]] = [(2,1,1)], tbh: float = tSR_in_yr, sr_function: callable = GammaSR_nlm_bxzh):
   """
   Monte Carlo integration to compute the marginal lieklihood of the ultralight boson's existence with Monte Carlo integration.

   Parameters:
      mu (float): Boson mass in eV.
      invf (float): Inverse of the boson decay time in GeV^-1.
      samples (np.ndarray): Two-dimenstional array of samples from the (M,a*) posterior distribution.
      states (list[tuple[int,int,int]], optional): List of quantum numbers |n,l,m> to consider (default: [(2,1,1)]).
      tbh (float, optional): Black hole superradiance timescale in years (default: tSR_in_yr).
      sr_function (callable, optional): Function to compute the superradiance rate (default: GammaSR_nlm_bxzh).

   Returns:
      float: Marginal likelihood of the ultralight boson's existence.
   """
   n_samples = len(samples[:,0])
   p = n_samples
   for mbh, astar in samples:
      for s in states:
         n, l, m = s
         alph = alpha(mu, mbh)
         # Check if SR mode
         if alph/l <= 0.5:
            # Only now (for efficiency) compute BHSR rate and check if it is fast enough
            srr = sr_function(mu, mbh, astar, n, l, m)
            check1 = can_grow_max_cloud(mbh, tbh, srr)
            check2 = not_bosenova_is_problem(mu, invf, mbh, tbh, n, m, srr)
            if check1 and check2:
               p -= 1
               break
   return p/float(n_samples)

@njit
def simple_grid_and_hpd(samples: np.ndarray, xlims: np.ndarray[float], ylims: np.ndarray[float], nbins_post: int = 50, nbins_grid: int = 200, thresh: float = 0.95):
   """
   Aa simple method to analyse 2D posterior samples, providing a grid and thresholds for the highest posterior density (HPD) region for plotting.

   Parameters:
      samples (np.ndarray): Two-dimensional array of samples from the (M,a*) posterior distribution.
      xlims (np.ndarray): Array of two floats representing the x-axis limits.
      ylims (np.ndarray): Array of two floats representing the y-axis limits.
      nbins_post (int, optional): Number of bins for analysing posterior grid (default: 50).
      nbins_grid (int, optional): Number of bins for the interpolated grid for plotting (default: 200).
      thresh (float, optional): Threshold for the HPD region (default: 0.95).

   Returns:
      tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float): Arrays of x and y coordinates, the posterior density grid, the x and y coordinates for the interpolated grid, the HPD threshold.
   """
   pdens = []
   nsamples = len(samples[:,0])
   dx = (xlims[-1] - xlims[0])/nbins_post
   dy = (ylims[-1] - ylims[0])/nbins_post
   for i in range(nbins_post):
      tmp = []
      for j in range(nbins_post):
         sel1 = (xlims[0] + i*dx < samples[:,0]) & (samples[:,0] < xlims[0] + (i+1)*dx)
         sel2 = (ylims[0] + j*dy < samples[:,1]) & (samples[:,1] < ylims[0] + (j+1)*dy)
         sel = sel1 & sel2
         tmp.append(np.sum(sel))
      pdens.append(tmp)
   pdens = np.array(pdens)/nsamples
   pdens_sorted = -np.sort(-pdens.flatten())
   psum = 0
   for p in pdens_sorted:
      pthresh = p
      psum += p
      if psum > thresh:
         break
   x0 = [xlims[0] + (i+0.5)*dx for i in range(nbins_post) for _ in range(nbins_post)]
   y0 = [ylims[0] + (j+0.5)*dy for _ in range(nbins_post) for j in range(nbins_post)]
   xi = np.linspace(xlims[0], xlims[1], nbins_grid)
   yi = np.linspace(ylims[0], ylims[1], nbins_grid)
   return x0, y0, pdens, xi, yi, pthresh

def compute_regge_slopes_211(mu: float, mbh_vals: list[float], tbh: float, invf: float = -1, sr_function: callable = GammaSR_nlm_bxzh) -> np.ndarray:
   """
   Compute the Regge slopes given a superradiance rate function.

   Parameters:
      mu (float): Boson mass in eV.
      mbh_vals (list[float]): List of black hole masses in Msol.
      sr_function (callable): Superradiance rate function with signature sr_function(mu, mbh, astar)
      tbh (float): Black hole timescale in yr.

   Returns:
      np.ndarray: Array of Regge slopes.

   Notes:
      - The Regge slope is defined as the minimum value of the dimensionless spin parameter 'a' at which the superradiance rate equals the inverse of the superradiance timescale.
      - Note that the root finding may fail if no Regge slope exists, or the user-defined function may produce an error, e.g. because of large spins or difficult BH masses. In any such case, we set the corresponding BH spin value = NAN.
   """
   inv_tbh = inv_eVs / (yr_in_s*tbh)
   a_min_vals = []
   for mbh in mbh_vals:
      nfi = n_fin(mbh)
      def foo(a, mbh):
         srr0 = sr_function(mu, mbh, a, 2, 1, 1)
         is_valid = alpha(mu, mbh) <= 0.5
         if invf > 0:
            srr = srr0*n_eq_211(mu, mbh, a, invf, srr0)
            is_valid *= srr > inv_tbh
         return is_valid*srr0 - np.log(nfi)*inv_tbh
      with warnings.catch_warnings(record=True) as w:
         try:
            res = root_scalar(foo, bracket=[0.01, 0.99], args=(mbh))
            a_root = res.root if len(w) == 0 else np.nan
         except ValueError:
            a_root = np.nan
      a_min_vals.append(a_root)
   return np.array(a_min_vals)

def is_box_allowed_211(mu: float, invf: float, bh_data: list[float], sr_function: callable = GammaSR_nlm_bxzh) -> bool:
   """
   Check if a n ULB model \f$(\mu, f^{-1})\f$ is allowed by superradiance, using the "box method".

   Parameters:
      mu (float): Boson mass in eV.
      invf (float): Inverse of the boson decay constant in GeV^-1.
      bh_data (list[float]): Black hole data (mbh_m, mbh_p, a_m, tbh).
      sr_function (callable): Superradiance rate function (default: GammaSR_nlm_bxzh).

   Returns:
      bool: True if the configuration is allowed by superradiance.

   Notes:
      - This function only considers the |211> level.
      - Setting invf = -1 will neglect self-interactions.
      - We only allow for the equilibrium when self-interactions are considered.
   """
   mbh_m, mbh_p, a_m, tbh = bh_data
   inv_tbh = inv_eVs / (yr_in_s*tbh)
   points_below_the_regge_trajectory = 2
   for mbh in [mbh_m, mbh_p]:
      is_sr = alpha(mu, mbh) <= 0.5
      nfi = n_fin(mbh)
      srr0 = sr_function(mu, mbh, a_m, 2, 1, 1)
      srr = srr0
      if invf > 0:
         srr = srr0*n_eq_211(mu, mbh, a_m, invf, srr0)
         is_sr *= srr > inv_tbh
      if srr0 > np.log(nfi)*inv_tbh and is_sr:
         points_below_the_regge_trajectory -= 1
   return points_below_the_regge_trajectory > 0

def is_box_allowed(mu: float, invf: float, bh_data: list[float], states: list[tuple[int,int,int]] = [(ell+1, ell, ell) for ell in range(1,6)], sr_function: callable = GammaSR_nlm_bxzh, assume_bosenova: bool = False) -> bool:
   """
   Check if a configuration is allowed by superradiance and bosenovae, using the `box method`.

   Parameters:
      mu (float): Boson mass in eV.
      invf (float): Inverse of the boson decay constant in GeV^-1.
      bh_data (tuple): Black hole data (bh name, tbh, mbh, mbh_err, a, a_err_p, a_err_m).
      states (list[tuple[int,int,int]]): List of levels \f$|nlm\rangle\f$ (default: all states with \f$n \leq 6\f$).
      sr_function (callable): Superradiance rate function (default: GammaSR_nlm_bxzh).
      assume_bosenova (bool): Check if assume that a bosenove happens (default: False).

   Returns:
      bool: True if the configuration is allowed by superradiance and bosenovae.
   """
   if states == [(2,1,1)]:
      return is_box_allowed_211(mu, invf, bh_data, sr_function)
   mbh_m, mbh_p, a_m, tbh = bh_data
   inv_tbh = inv_eVs / (yr_in_s*tbh)
   # Iterate over the black hole mass range to find points below the minimum of the Regge trajectories
   for mbh in np.linspace(mbh_m, mbh_p, 100):
      for s in states:
         above_regge_trajectory = []
         n, l, m = s
         # Check SR condition
         if alpha(mu, mbh)/l <= 0.5:
            srr = sr_function(mu, mbh, a_m, n, l, m)
            nfi = n_fin(mbh, m=m)
            if invf > 0:
               if assume_bosenova:
                  srr /= eta_bn(mu, invf, mbh, n, m)
               else:
                  srr *= n_eq_211(mu, mbh, a_m, invf, srr)
            check = srr > np.log(nfi)*inv_tbh
            above_regge_trajectory.append(check)
      if all(above_regge_trajectory):
         continue
      else:
         return 1
   return 0