from __future__ import print_function

import sys
import os
import numpy as np
import errno
import subprocess
import shlex

from astropy import units
from astropy.io import ascii

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import scipy.stats._multivariate as mv
from scipy.stats import norm, uniform


class ABCCovError(Exception):
   """ ABCCovError

   Generic error that is raised by scripts that import this module.

   """

   pass


def alpha_new(n_S, n_D):
    """Return precision matrix estimate bias prefactor alpha.
       IK17 (5).
    """

    return (n_S - n_D - 2.0)/(n_S - 1.0)


def read_ascii(in_name):
    """Read ascii file.

    Parameters
    ----------
    in_name: string
        input file name

    Returns
    ----
    dat: array of named columns
        file content
    """
    
    dat = np.genfromtxt(in_name, names=True, deletechars=['[]'])

    return dat


def write_ascii(file_base, cols, names):
    """Write ascii file.

    Parameters
    ----------
    file_base: string
        output file name base
    cols: matrix
        data
    names: array of string
        column names

    Returns
    -------
    None
    """

    header = ' '.join(names)[2:]
    data   = np.array(cols).transpose()
    np.savetxt('{}.txt'.format(file_base), data, header=header, fmt='%.10g')


def get_n_S_arr(n_S_min, n_D, f_n_S_max, n_n_S, n_S=None):
    """Return array of values of n_S=number of simulations.

    Parameters
    ----------
    n_S_min: int
        smallest n_S
    n_D: int
        number of data points
    f_n_S_max: float
        largest n_S = f_n_S_max * n_D
    n_n_S: int
        number of values for n_S
    n_S: array of int, optional, default=None
        array of number of simulations, overrides all other arguments if not None

    Returns
    -------
    n_S_arr: array of int
        array with n_S values
    n_n_S: int
        number of n_S values
    """

    if n_S != None:
        if min(n_S) > 1:
            n_S = [int(n) for n in n_S]
        else:
            # For the affine_off_diag model, n_S might be encoding
            # the float correlation coefficient r<1
            pass
        n_S_arr = np.array(n_S)
        n_n_S   = len(n_S_arr)
    else:
        start   = n_S_min
        stop    = n_D * f_n_S_max
        if sys.version_info.major == 3:
            n_S_arr = np.logspace(np.log10(start), np.log10(stop), n_n_S, dtype='int')
        elif sys.version_info.major == 2:
            n_S_arr = [int(nS) for nS in np.logspace(np.log10(start), np.log10(stop), n_n_S)]
        else:
            error('Invalid python version', sys.version_info)

    return n_S_arr, n_n_S


def no_bias(n, n_D, par):
    """Unbiased estimator of par.
       For example maximum-likelihood estimate of covariance normalised trace,
       TKJ13 (17).
       Or Fisher matrix errors from debiased estimate of inverse covariance.
    """

    return np.asarray([par] * len(n))


def alpha(n_S, n_D):
    """Return precision matrix estimate bias prefactor alpha.
    """

    return (n_S - 1.0)/(n_S - n_D - 2.0)


def A(n_S, n_D):
    """Return TJK13 (27)
    """

    A = alpha(n_S, n_D)**2 / ((n_S - n_D - 1.0) * (n_S - n_D - 4.0))

    return A


def std_fish_biased_TJ14(n, n_D, par):
    """Error on variance from the Fisher matrix. From TJ14 (12), with division by the de-biasing factor alpha.
    """

    n_P = 2  # Number of parameters

    return [np.sqrt(2 * (n_S - n_D + n_P - 1) / (n_S - n_D - 2)**2) / alpha(n_S, n_D) * par for n_S in n]


def std_fish_biased_TJK13(n, n_D, par):
    """0th-order error on variance from Fisher matrix with biased inverse covariance estimate.
       From TJK13 (49) with A (27) instead of A_corr (28) in (49).
    """

    return [np.sqrt(2 * A(n_S, n_D) / alpha(n_S, n_D)**4 * (n_S - n_D - 1)) * par for n_S in n]


def std_fish_deb(n, n_D, par):
    """Error on variance from Fisher matrix with debiased inverse covariance estimate.
       Square root of TJK13 (49, 50).
    """

    return [np.sqrt(2.0 / (n_S - n_D - 4.0)) * par for n_S in n]


def std_fish_Gupta(n, n_D, par):
    """Error on variance from Fisher matrix with debiased inverse covariance estimate.
       Following Gupta & Nagar (2000), Theorem 3.3.13
    """

    n_P = 2

    return [np.sqrt(2.0 / (n_S - n_D + n_P - 1)) * par for n_S in n]



def par_fish_SH(n, n_D, par):
    """Parameter RMS from Fisher matrix esimation of SH likelihood.
    """

    return [np.sqrt(alpha_new(n_S, n_D) * 2.0 * n_S / (n_S - 1.0)) * par for n_S in n]


def coeff_TJ14(n_S, n_D, n_P):
    """Square root of the prefactor for the variance of the parameter variance, TJ14 (12).
    """

    return np.sqrt(2 * (n_S - n_D + n_P - 1) / (n_S - n_D - 2)**2)


def std_fish_deb_TJ14(n, n_D, par):
    """Error on variance from the Fisher matrix. From TJ14 (12).
    """

    n_P = 2  # Number of parameters

    return [coeff_TJ14(n_S, n_D, n_P) * par for n_S in n]


def get_n_S_R_from_fit_file(file_base, npar=2):
    """Return array of number of simulations, n_n_S, and number of runs, n_R from fit output file.

    Parameters
    ----------
    file_base: string
        input file name base (without extension)
    npar: int
        number of parameters, default=2

    Returns
    -------
    n_S: array of int
        array of number of simulations
    n_R: int
        number of runs
    """

    in_name = '{}.txt'.format(file_base)
    try:
        dat = read_ascii(in_name)
    except IOError as exc:
        if exc.errno == errno.ENOENT:
            error('File {} not found'.format(in_name))
        else:
            raise

    n_S = dat['n_S']
    n_R   = int((len(dat.dtype) - 1) / 2 / npar)

    return n_S, n_R


def get_cov_ML(mean, cov, size):
    """Return maximum-likelihood estime of covariance matrix, from
       realisations of a multi-variate Normal
    
    Parameters
    ----------
    mean: array(double)
        mean of mv normal
    cov: array(double)
        covariance matrix of mv normal
    size: int
        dimension of data vector, cov is size x size matrix

    Returns
    -------
    cov_est: matrix of double
        estimated covariance matrix, dimension size x size
    """
            
    from scipy.stats import multivariate_normal

    y2 = multivariate_normal.rvs(mean=mean, cov=cov, size=size)
    # y2[:,j] = realisations for j-th data entry
    # y2[i,:] = data vector for i-th realisation

    # Calculate covariance matrix via np
    cov_est = np.cov(y2, rowvar=False)
    
    if size > 1:
        pass
    else:
        cov_est = [[cov_est]]
    
    return cov_est


def get_cov_SSC(ell, C_ell_obs, cov_SSC_fname, func='BKS17'):
    """Return SSC weak-lensing covariance, by reading it (or some function of it)
    from a file.

    Parameters
    ----------
    ell: array of double
         angular Fourier modes
    C_ell: array of double
         power spectrum
    cov_SSC_fname: string
        file name of relative SSC covariance, in column format
    func: string, optional, default='BKS17'
        function of relative covariance, one in 'BKS17', 'id'

    Returns
    -------
    cov_SSC: matrix of double
        covariance matrix
    """

    dat = ascii.read(cov_SSC_fname)
    n   = int(np.sqrt(len(dat)))

    ell_SSC = dat['col2'][0:n]

    # Check whether ell's match input ell's to this function
    drel = (ell_SSC - ell) / ell
    if np.where(drel > 1e-2)[0].any():
        print(ell)
        print(ell_SSC)
        print(drel)
        error('SSC ell values do not match input ell values')

    cov_SSC = np.zeros((n, n))
    c = 0
    for i in range(n):
        for j in range(n):
            cov_rel       = dat['col3'][c]

            if func == 'BKS17':
                # File stored Cov_SSC(l1, l2) / C(l1) / C(l2) * 10^4
                cov_SSC[i][j] = cov_rel / 1.0e4 * C_ell_obs[i] * C_ell_obs[j]
            elif func == 'id':
                cov_SSC[i][j] = cov_rel

            c = c + 1

    return cov_SSC


def get_cov_Gauss(ell, C_ell, f_sky, sigma_eps, nbar):
    """Return Gaussian weak-lensing covariance (which is
    a diagonal matrix).
    
    Parameters
    ----------
    ell: array of double
         angular Fourier modes
    C_ell: array of double
         power spectrum
    f_sky: double
        sky coverage fraction
    sigma_eps: double
        complex ellipticity dispersion
    nbar: double
        galaxy number density [rad^{-2}]

    Returns
    -------
    Sigma: matrix of double
        covariance matrix
    """

    # Total (signal + shot noise) E-mode power spectrum.
    # (The factor of 2 in the shot noise indicates one of two
    # components for the power spectrum, see Joachimi et al. (2008).

    C_ell_tot = C_ell + sigma_eps**2 / nbar


    # New 11/09/2018: Added Delta ell

    # The following seems complicated
    # To just use Delta_ell = diff(ell) would bias the Delta's,
    # since they would not correspond to the bin center ell's.
    
    # For log10-bins: Delta log10 ell = Delta ell / ell / log 10,
    # we ignore constant 1/log10 that would later be multiplied when
    # doing 10^.
    # Use mean of ell_i+1 and ell_i good to < 1% compared
    # to Delta ln ell
    ell_mode = get_ell_mode(ell)
    if ell_mode == 'log':
        Delta_ln_ell = np.diff(ell) / (ell[:-1]/2 + ell[1:]/2)
 
        # add last element again to restore length of Delta_ell 
        Delta_ln_ell = np.append(Delta_ln_ell, Delta_ln_ell[-1])

        Delta_ell = Delta_ln_ell * ell

    else:
        Delta_ell = np.diff(ell)
        Delta_ell = np.append(Delta_ell, Delta_ell[-1])

    D = 1.0 / (f_sky * (2.0 * ell + 1) * Delta_ell) * C_ell_tot**2
    Sigma = np.diag(D)

    return Sigma



def sample_cov_Wishart(cov, n_S):
    """Returns estimated coariance as sample from Wishart distribution
 
    Parameters
    ----------
    cov: matrix of double
         'true' covariance matrix (scale matrix)
    n_S: int
         number of simulations, dof = nu = n_S - 1

    Returns
    -------
    cov_est: matrix of double
         sampled matrix
    """

    # Sample covariance from Wishart distribution, with dof nu=n_S - 1
    W = mv.wishart(df=n_S - 1, scale=cov)

    # Mean of Wishart distribution is cov/dof = cov/(n_S - 1)
    cov_est = W.rvs() / (n_S - 1)

    return cov_est


def get_ell_mode(ell):
    """Return binning type, linear or logarithmic

    Parameters
    ----------
    ell: array of double
        ell-bins

    Returns
    -------
    mode: string
        'log' or 'lin'
    """

    eps = 0.001
    if np.fabs(ell[2]/ell[1] - ell[1]/ell[0]) < eps:
        return 'log'
    elif np.fabs((ell[2]-ell[1]) - (ell[1]-ell[0])) < eps:
        return 'lin'
    else:
        print(ell)
        print(ell[2]/ell[1] - ell[1]/ell[1])
        print((ell[2]-ell[1]) - (ell[1]-ell[0]))
        raise ValueError('Bins neither logarithmic nor linear')


def get_cov_WL(model, ell, C_ell_obs, nbar, f_sky, sigma_eps, nsim, d_SSC=0.75):
    """Compute true and estimated WL covariance.

    Parameters
    ----------
    model: string
        One in 'Gauss', 'Gauss+SSC_BKS17'
    ell: array of float
        ell-values
    C_ell_obs: array of float
        observed power spectrum
    nbar: float
        galaxy density in arcmin^{-2}
    f_sky: float
        Observed sky fraction
    sigma_eps: float
        ellipticity dispersion
    nsim: int
        number of simulations for covariance estimation
    d_SSC: double, optional, default=0.75
        if >0, increases SSC diagonal by d_SSC

    Returns
    -------
    cov: matrix of float
        'true' underlying covariance matrix
    cov_est: matrix of float
        estimated covariance
    """

    # Construct (true) covariance Sigma
    nbar_amin2  = units.Unit('{}/arcmin**2'.format(nbar))
    nbar_rad2   = nbar_amin2.to('1/rad**2')
    # We use the same C_ell as the 'observation', from above
    cov_G       = get_cov_Gauss(ell, C_ell_obs, f_sky, sigma_eps, nbar_rad2)

    if model == 'Gauss':
        cov = cov_G

    elif model == 'Gauss+SSC_BKS17':
        ell_mode = get_ell_mode(ell)
        if ell_mode == 'log':
            cov_SSC_base = 'cov_SSC_rel_log'
        elif ell_mode == 'lin':
            cov_SSC_base = 'cov_SSC_rel_lin'
        cov_SSC_path = '{}/{}.txt'.format('.', cov_SSC_base)
        func_SSC      = 'BKS17'
        cov_SSC       = get_cov_SSC(ell, C_ell_obs, cov_SSC_path, func_SSC)

        # Writing covariances to files for testing/plotting
        np.savetxt('cov_G.txt', cov_G)
        np.savetxt('cov_SSC.txt', cov_SSC)

        if d_SSC > 0:
            d = np.diag(cov_SSC) * d_SSC
            cov_SSC = cov_SSC + np.diag(d)

            print('diag factor d_SSC = {}'.format(d_SSC))
            print('Mean increase of diagonal wrt tot = {}'.format(np.mean(d/np.diag(cov_G+cov_SSC))))
        else:
            print('No d_SSC correction factor')

        cov = cov_G + cov_SSC

    else:

       error('Invalid covariance mode \'{}\''.format(mode))

    size = cov.shape[0]

    # if nu=nsim-1<p, or scale matrix is singular: create normal mrv
    # and compute cov.
    # Otherwise: sample cov from Wishart.

    if nsim - 1 >= size:
        try:
            cov_est = sample_cov_Wishart(cov, nsim)
        except np.linalg.LinAlgError:
            cov_est = get_cov_ML(C_ell_obs, cov, size)
        except:
            print('Warning: scale matrix not positive, sampling from normal instead of Wishart')
            raise
    else:
        cov_est = get_cov_ML(C_ell_obs, cov, size)

    return cov, cov_est



def weighted_std(data, weights):
    """Taken from http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf"""

    mean = np.average(data, weights=weights)
    c = sum([weights[i] > pow(10, -6) for i in range(weights.shape[0])])

    num = sum([weights[i] * pow(data[i] - mean, 2) for i in range(data.shape[0])])
    denom = (c - 1) * sum(weights)/float(c)

    return np.sqrt(num / denom)



def add(ampl):
    """Return the additive constant of the quadratic function
       from the amplitude fitting parameter
       (mimics power-spectrum normalisation s8)
    """

    # This provides a best-fit amp=0.827, but the 10% increased
    # spectrum (0.9097) gives a best-fit of 0.925
    # Changing the prefactor of amp or lg(amp) does not help...
    c = np.log10(ampl)*2 - 6.11568527 + 0.1649

    return c



def shift(tilt):
    """Return the shift parameter of the quadratic function
       from the tilt parameter (mimics matter density)
    """

    u0 = tilt * 1.85132114 / 0.306

    return u0



def quadratic(u, *params):
    """Used to fit quadratic function varying all three parameters
    """

    (ampl, tilt, a) = np.array(params)
    c  = add(ampl)
    u0 = shift(tilt)

    return c + a * (u - u0)**2



def quadratic_ampl_tilt(u, ampl, tilt):
    """Return quadratic function given coordinate 1 (u=logell), amplitude,
       and tilt.
    """

    param   = (ampl, tilt, -0.17586216)
    q       = quadratic(u, *param)

    return q



def model_quad(u, ampl, tilt):
    """Return model based on quadratic function. This should correspond
       to the WL power spectrum C_ell with u = lg ell.
       Since q(u) ~ lg[ ell C_ell], the model is
       y = C_ell = 10^q / ell = 10^q / 10^lg ell = 10^(q - u).
    """

    q = quadratic_ampl_tilt(u, ampl, tilt)

    y = 10**(q - u) 

    return y


def par_symbol(par, eq=True):
    """Par Symbol
    Return parameter symbol for given parameter name

    Parameters
    ----------
    par : string
        parameter name

    Returns
    -------
    res : string
        parameter symbol
    """

    symbol = {
        'tilt' : 't',
        'ampl' : 'A',
        'Omegam' : '\Omega_{\\rm m}',
        'sigma8' : '\sigma_8',
    }

    if par in symbol:
        res = symbol[par]
    else:
        res = par

    if eq:
        return f'${res}$'
    else:
        return res


def stat_notation(stat):
    """Stat Notation
    Returns format notation of a given statistical function, e.g. mean, std

    Parameters
    ----------
    stat : string
        (informal) name of statistical function
    """

    if stat == 'mean':
        return '$\\bar\\theta$'
    elif stat == 'std':
        return 'SE$(\\bar\\theta)$'
    elif stat == 'std_var':
        return 'SD$[$Var$(\\hat{\\theta})]$'
    else:
        raise ABCCovError(f'Invalid statistical function name f{stat}')

class param:
    """General class to store (default) variables
    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self, **kwds):
        return str(self.__dict__)

    def var_list(self, **kwds):
        return vars(self)


class Results:
    """Store results of Fisher matrix and MCMC sampling
    """

    def __init__(self, par_name, n_n_S, n_R, file_base='mean_std', yscale='linear', fct=None, n_D=-1, n_S_arr=[]):
        """Set arrays for mean and std storing all n_S simulation cases
           with n_R runs each
        """

        self.mean = {}
        self.std = {}
        self.par_name = par_name

        self.file_base = file_base

        if np.isscalar(yscale):
            self.yscale = [yscale, yscale]
        else:
            self.yscale = yscale

        self.fct = fct
        for p in par_name:
            self.mean[p] = np.zeros(shape = (n_n_S, n_R))
            self.std[p] = np.zeros(shape = (n_n_S, n_R))

        self.F  = np.zeros(shape = (n_n_S, n_R, 2, 2))
        self.fs = 12

        self.n_D = n_D
        self.n_S_arr = n_S_arr


    def set(self, par, i, run, which='mean'):
        """Set mean or std for all parameteres for simulation #i and run #run
        """

        for j, p in enumerate(self.par_name):
            w = getattr(self, which)
            w[p][i][run] = par[j]


    def get_std_var(self, p, ste=False, n_S_range='all'):
        """Return standard deviation (SD) of the variance or standard error over all runs

        Parameters
        ----------
        p : string
            parameter name
        ste : bool, optional, default=False
            if True (False), return SD of standard error (variance)
        n_S_range : str, optional, default='all'
            range of n_S for average, one in
             'all' : entire range
             'n_S>n_D' :  n_S>n_D, non-singular covariance
             'n_S<=n_D' : n_S<=n_D, singular covariance

        Returns
        -------
        std_var : numpy array of float
            SD of SE or variance for each n_s
        """

        std_var = []
        for i, n_S in enumerate(self.n_S_arr):
            if (
                (n_S_range == 'all')
                or ( (n_S_range == 'n_S>n_D') and (n_S > self.n_D) )
                or ( (n_S_range == 'n_S<=n_D') and (n_S <= self.n_D) )
            ): 
                if ste:
                    s = np.std(self.std[p][i])
                else:
                    s = np.std(self.std[p][i]**2)
                std_var.append(s)

        return np.array(std_var)

    def get_mean(self, p, n_S_range='all'):
        """Return mean parameter per simulation, averaged over realisations
        """

        mean = []
        for i, n_S in enumerate(self.n_S_arr):
            m = np.mean(self.mean[p][i])
            if (
                (n_S_range == 'all')
                or ( (n_S_range == 'n_S>n_D') and (n_S > self.n_D) )
                or ( (n_S_range == 'n_S<=n_D') and (n_S <= self.n_D) )
            ):
                mean.append(m)

        return np.array(mean)

    def get_std(self, p):
        """Return std of the parameter per simulation, averaged over realisations
        """

        n_n_S = self.mean[self.par_name[0]].shape[0]
        std = np.zeros(shape = n_n_S)
        for i in range(n_n_S):
            std[i] = np.mean(self.std[p][i])

        return std

    def get_mean_std_all(self, p, ste=False, n_S_range='all'):
        """Get Mean Std All
        Return mean, standard error (SE), std, and std of SE/variance,
        all averaged over simulations and runs

        Parameters
        ----------
        p : string
            parameter name
        ste : bool, optional, default=False
            if True (False), return SD of standard error (variance)
        n_S_range : str, optional, default='all'
            range of n_S for average, one in
             'all' : entire range
             'n_S>n_D' :  n_S>n_D, non-singular covariance
             'n_S<=n_D' : n_S<=n_D, singular covariance

        Returns
        -------
        std_var : array of float
            SD of SE or variance for each n_s
        """

        # Mean over all runs and simulations
        mean = self.get_mean(p, n_S_range=n_S_range)
        m = np.mean(mean)

        # Create array of mean for each run and simulation 
        mean2 = []
        n_n_S, n_R = self.mean[self.par_name[0]].shape
        for i, n_S in enumerate(self.n_S_arr):
            for run in range(n_R):
                this_m = self.mean[p][i][run]
                if (
                    (n_S_range == 'all')
                    or ( (n_S_range == 'n_S>n_D') and (n_S > self.n_D) )
                    or ( (n_S_range == 'n_S<=n_D') and (n_S <= self.n_D) )
                ): 
                    mean2.append(this_m)
        # Compute SD of mean
        s = np.std(mean2)

        # Get standard error (SE) for each run and simulation
        std = []
        for i, n_S in enumerate(self.n_S_arr):
            for run in range(n_R):
                this_s = self.std[p][i][run]
                if (
                    (n_S_range == 'all')
                    or ( (n_S_range == 'n_S>n_D') and (n_S > self.n_D) )
                    or ( (n_S_range == 'n_S<=n_D') and (n_S <= self.n_D) )
                ): 
                    std.append(this_s)
        # Compute mean standard error
        s2 = np.mean(std)

        # Compute standard deviation of SE or SE^2
        std_ste = self.get_std_var(p, ste=ste, n_S_range=n_S_range)
        s_e = np.mean(std_ste)

        return m, s, s2, s_e


    def read_mean_std(self, npar=2, update=False, verbose=False):
        """Read mean and std from file.

        Parameters
        ----------
        npar: int, optional, default=2
            number of parameters
        update: bool, optional, default=False
            if True updates values from init call;
            if False exits with error if values different
        verbose: bool, optional, defaut=False
            verbose output if True

        Returns
        -------
        n_S_arr: array of int
            list of value of n_S, number of simulations
        None
        """

        if not update:
            dummy_n_n_S, n_R = self.mean[self.par_name[0]].shape

        in_name = '{}.txt'.format(self.file_base)
        if verbose:
            print('Reading file {}'.format(in_name))
        try:
            dat = read_ascii(in_name)
            my_n_S_arr, my_n_R = get_n_S_R_from_fit_file(self.file_base, npar=npar)

            if update:
                my_n_n_S = len(my_n_S_arr)
                for p in self.par_name:
                    self.mean[p] = np.zeros(shape = (my_n_n_S, my_n_R))
                    self.std[p] = np.zeros(shape = (my_n_n_S, my_n_R))
                n_R = my_n_R
            else:
                if my_n_R != n_R:
                    raise IOError('File {} has n_R={}, not {}'.format(in_name, my_n_R, n_R))


            for p in self.par_name:
                for run in range(n_R):
                    col_name = 'mean[{0:s}]_run{1:02d}'.format(p, run)
                    self.mean[p].transpose()[run] = dat[col_name]
                    col_name = 'std[{0:s}]_run{1:02d}'.format(p, run)
                    self.std[p].transpose()[run] = dat[col_name]

        except IOError as exc:
            if exc.errno == errno.ENOENT:
                if verbose == True:
                    warning('File {} not found'.format(in_name))
                pass
            else:
                raise

        n_S_arr = [int(n_S) for n_S in my_n_S_arr]
        return n_S_arr


    def write_mean_std(self, n, format='ascii'):
        """Write mean and std to file
        """

        n_n_S, n_R = self.mean[self.par_name[0]].shape
        if format == 'ascii':
            cols  = [n]
            names = ['# n_S']
            for p in self.par_name:
                for run in range(n_R):
                    cols.append(self.mean[p].transpose()[run])
                    names.append('mean[{0:s}]_run{1:02d}'.format(p, run))
                    cols.append(self.std[p].transpose()[run])
                    names.append('std[{0:s}]_run{1:02d}'.format(p, run))

            write_ascii(self.file_base, cols, names)


    def read_Fisher(self, format='ascii'):
        """Read Fisher matrix
        """

        n_n_S, n_R = self.mean[self.par_name[0]].shape
        if format == 'ascii':
            fname = 'F_{}.txt'.format(self.file_base)
            if os.path.isfile(fname):
                dat = read_ascii(fname)
                for run in range(n_R):
                    for i in (0,1):
                        for j in (0,1):
                            col_name = 'F[{0:d},{1:d}]_run{2:02d}'.format(i, j, run)
                            self.F[:, run, i, j] = dat[col_name].transpose()


    def write_Fisher(self, n, format='ascii'):
        """Write Fisher matrix.
        """

        n_n_S, n_R = self.mean[self.par_name[0]].shape
        if format == 'ascii':
            cols  = [n]
            names = ['# n_S']
            for run in range(n_R):
                for i in (0,1):
                    for j in (0,1):
                        Fij = self.F[:, run, i, j]
                        cols.append(Fij.transpose())
                        names.append('F[{0:d},{1:d}]_run{2:02d}'.format(i, j, run))
            write_ascii('F_{}'.format(self.file_base), cols, names)


    def append(self, new, verbose=False):
        """Append new result to self.

        Parameters
        ----------
        self, new: class Result
            previous and new result
        verbose: bool, optional
            verbose mode

        Returns
        -------
        res: bool
            True for success
        """

        n_n_S, n_R = self.mean[self.par_name[0]].shape
        n_n_S_new, n_R_new = new.mean[new.par_name[0]].shape
        if n_n_S != n_n_S_new:
            error( \
                'Number of simulations different for previous ({}) and new ({}), skipping append...'.format( \
                n_n_S, n_n_S_new), stop=False, verbose=verbose)

            return False

        for p in self.par_name:
            mean      = self.mean[p]
            std       = self.std[p]

            self.mean[p]   = np.zeros(shape = (n_n_S, n_R + n_R_new))
            self.std[p]    = np.zeros(shape = (n_n_S, n_R + n_R_new))
            for n_S in range(n_n_S):
                self.mean[p][n_S]   = np.append(mean[n_S], new.mean[p][n_S])
                self.std[p][n_S]    = np.append(std[n_S], new.std[p][n_S])

        F = self.F
        self.F = np.zeros(shape = (n_n_S, n_R + n_R_new, 2, 2))
        for n_S in range(n_n_S):
            for r in range(n_R):
                self.F[n_S][r] = F[n_S][r]
            for r in range(n_R_new):
                self.F[n_S][n_R + r] = new.F[n_S][r]

        return True


    def plot_mean_std(self, par=None, boxwidth=None, xlog=False, ylim=None, model='affine'):
        """Plot mean and std versus number of realisations n

        Parameters
        ----------
        par: dictionary of array of float, optional
            input parameter values and errors, default=None
        boxwidth: float, optional
            box width for box plots, default: None, width is determined from n
        xlog: bool, optional
            logarithmic x-axis, default False
        ylim: array of two floats, optional, default None
            y-limits
        model: string
            model, one in 'affine', 'quadratic'

        Returns
        -------
        None
        """

        n = self.n_S_arr
        n_R = self.mean[self.par_name[0]].shape[1]

        marker = ['.', 'D']
        markersize = [6] * len(marker)
        color = ['b', 'g']
        linestyle = ['-', '--']

        plot_sth = False
        fs = 13
        plot_init(self.n_D, n_R, fs=fs)

        box_width = set_box_width(boxwidth, xlog, n)
        if model == 'affine':
            rotation = 'vertical'
        else:
            rotation = 'horizontal'

        if xlog == True:
            fac_xlim = 1.6
            xmin = n[0]/fac_xlim
            xmax = n[-1]*fac_xlim
        else:
            fac_xlim   = 1.05
            xmin = (n[0]-5)/fac_xlim**5
            xmax = n[-1]*fac_xlim

        leg2 = {}
        lab2 = {}
        loc = {'mean': 'center', 'std': 'upper'}

        # Set the number of required subplots (1 or 2)
        n_panel = 1

        j_panel = {}
        for j, which in enumerate(['mean', 'std']):

            leg2[which] = []
            lab2[which] = []

            for i, p in enumerate(self.par_name):
                y = getattr(self, which)[p]   # mean or std for parameter p
                if y.any():
                    n_panel = 2
                    j_panel[which] = j+1

        if len(j_panel) == 1:   # Only one plot to do: Use entire canvas
            n_panel = 1
            j_panel[list(j_panel.keys())[0]] = 1

        if xlog == True:
            width   = lambda p, box_width: 10**(np.log10(p)+box_width/2.)-10**(np.log10(p)-box_width/2.)
            flinlog = lambda x: np.log(x)
        else:
            width   = lambda p, box_width: np.zeros(len(n)) + float(box_width)
            flinlog = lambda x: x

        for j, which in enumerate(['mean', 'std']):
            for i, p in enumerate(self.par_name):
                y = getattr(self, which)[p]   # mean or std for parameter p
                if y.any():
                    ax = plt.subplot(1, n_panel, j_panel[which])

                    if y.shape[1] > 1:
                        bplot = plt.boxplot(y.transpose(), positions=n, sym='.', widths=width(n, box_width), patch_artist=True)
                        for key in bplot:
                            plt.setp(bplot[key], color=color[i], linewidth=2)
                        plt.setp(bplot['whiskers'], linestyle='-', linewidth=2)
                        plt.setp(bplot['boxes'], facecolor=lighten_color(color[i], amount=0.25))
                        leg2[which].append(bplot['boxes'][0])
                    else:
                        pl = plt.plot(n, y.mean(axis=1), marker[i], ms=markersize[i], color=color[i],
                                      linestyle=linestyle[i])
                        leg2[which].append(pl)
                    lab2[which].append(fr'$\bar {par_symbol(p, eq=False)}$')

                    if xlog == True:
                        ax.set_xscale('log')

                    # Define high-resolution x array for smooth lines
                    if len(n) > 1:
                        n_fine = np.arange(n[0], n[-1]+len(n)/20.0, len(n)/20.0)
                    else:
                        x0 = n[0] / 4
                        x1 = n[0] * 4
                        n_fine = np.arange(x0, x1, 10)
                    my_par = par[which]
                    p_sym = par_symbol(p, eq=False)

                    # True input or Fisher-matrix normal
                    if which == 'mean':
                        label = f'true ${p_sym}$'
                    else:
                        label=rf'$\mathbf{{\hat F}}({p_sym})$'
                    plt.plot(n_fine, no_bias(n_fine, self.n_D, my_par[i]), '{}{}'.format(color[i], linestyle[i]),
			                 label=label, linewidth=2)

                    # Additional theoretical line
                    if self.fct is not None and which in self.fct:
                        plt.plot(
                            n_fine,
                            self.fct[which](n_fine, self.n_D, my_par[i]),
                            '{}{}'.format(color[i], linestyle[i]),
                            label=rf'$\mathbf{{\hat F}}_{{T^2}}({p_sym})$',
                            linewidth=1
                        )

                    
        # Finalize plot
        for j, which in enumerate(['mean', 'std']):
            if which in j_panel:

                # Get main axes
                ax = plt.subplot(1, n_panel, j_panel[which])

                # Dashed vertical line at n_S = self.n_D
                plt.plot([self.n_D, self.n_D], [-1e2, 1e2], ':', linewidth=1)
                plt.plot([self.n_D, self.n_D], [1e-5, 1e2], ':', linewidth=1)

                # Main-axes settings
                plt.xlabel('$n_{{\\rm s}}$')
                ylabel = stat_notation(which)
                plt.ylabel(ylabel)
                ax.set_yscale(self.yscale[j])
                leg = ax.legend(loc=f'{loc[which]} right', frameon=False, handlelength=1.3)
                plt.gca().add_artist(leg)
                if which in leg2:
                    ax.legend(leg2[which], lab2[which], loc=f'{loc[which]} left', frameon=False, handlelength=0.9)
                
                plt.xlim(xmin, xmax)

                # x-ticks
                ax = plt.gca().xaxis
                ax.set_major_formatter(ScalarFormatter())
                plt.ticklabel_format(axis='x', style='sci')

	            # For MCMC: Remove second tick label due to text overlap if little space
                x_loc = []
                x_lab = []
                for i, n_S in enumerate(n):
                    x_loc.append(n_S)
                    if n_panel == 1 or i != 1 or len(n)<10 or n_S<self.n_D:
                        lab = '{}'.format(n_S)
                    else:
                        lab = ''
                    x_lab.append(lab)
                plt.xticks(x_loc, x_lab, rotation=rotation)
                ax.label.set_size(fs)

	            # Second x-axis
                ax2 = plt.twiny()
                x2_loc = []
                x2_lab = []
                for i, n_S in enumerate(n):
                    if n_S > 0:
                        if n_panel == 1 or i != 1 or len(n)<10 or n_S<self.n_D:
                            frac = float(self.n_D) / float(n_S)
                            if frac > 100:
                                lab = '{:.3g}'.format(frac)
                            else:
                                lab = '{:.2g}'.format(frac)
                        else:
                            lab = ''
                        x2_loc.append(flinlog(n_S))
                        x2_lab.append(lab)
                plt.xticks(x2_loc, x2_lab)
                ax2.set_xlabel('$p / n_{\\rm s}$', size=fs)
                if model == 'affine':
                    for tick in ax2.get_xticklabels():
                        tick.set_rotation(90)
                plt.xlim(flinlog(xmin), flinlog(xmax))

                plot_sth = True

            # Set y limits by hand to be the same for all sampling plot (which have two panels)
            if n_panel == 2:
                if which == 'mean':
                    if model == 'affine':
                        plt.ylim(-2, 2)
                    else:
                        plt.ylim(0.2, 0.9)
                if which == 'std':
                    if model == 'affine':
                        plt.ylim(1e-4, 3e-1)
                    elif model == 'wl':
                        plt.ylim(2e-3, 2.9e-2)
                    else:
                        plt.ylim(5e-4, 5e-2)

            if ylim is not None:
                plt.ylim(ylim)


        if plot_sth == True:
            plt.tight_layout(h_pad=5.0)
            plt.savefig('{}.pdf'.format(self.file_base), bbox_inches="tight")

    def plot_std_var(self, par=None, sig_var_noise=None, xlog=False, title=False, model='affine', ste=False):
        """Plot standard deviation (SD) of parameter standard error or variance

        Parameters 
        ---------- 
        par: dictionary of array of float, optional
            input parameter values and errors, default=None
        xlog: bool, optional
            logarithmic x-axis, default False
        title: bool, optional, default=False
            if True, print title with n_d, n_r
        model: string
            model, one in 'affine', 'quadratic'
        ste : bool, optional, default=False
            if True (False), plot SD of standard error (variance)
            
        Returns 
        -------     
        None    
        """         

        n = self.n_S_arr
        n_R = self.mean[self.par_name[0]].shape[1]

        color = ['b', 'g']
        linestyle = ['-', '--']
        marker = ['o', 's']

        plot_sth = False
        fs = self.fs * 0.95
        plot_init(self.n_D, n_R, fs=fs, title=title, figsize=(3.0, 4.8))

        # For output ascii file
        cols  = [n]
        names = ['# n_S']

        for i, p in enumerate(self.par_name):
            y = self.get_std_var(p, ste=ste)
            if y.any():
                plt.plot(n, y, marker=marker[i], color=color[i], label=par_symbol(p), linestyle='None')
                cols.append(y)
                names.append('sigma(sigma^2_{})'.format(p))

                if sig_var_noise != None:
                    plt.plot(n, y - sig_var_noise[i], marker=marker[i], mfc='none', color=color[i], \
                             label='$\sigma(\sigma^2_{0}) - \sigma_n(\sigma^2_{0})$'.format(p), linestyle='None')

        for i, p in enumerate(self.par_name):
            y = self.get_std_var(p, ste=ste)
            if y.any():
                if par is not None:
                    n0 = self.n_D + 2
                    n_fine = np.arange(n0, n[-1], len(n)/40.0)
                    if 'std_var_TJK13' in self.fct:
                        yy = self.fct['std_var_TJK13'](n_fine, self.n_D, par[i])
                        plot_add_legend(True, n_fine, yy, ':', color=color[i], label='TJK13')
                        cols.append(self.fct['std_var_TJK13'](n, self.n_D, par[i]))
                        names.append('TJK13({})'.format(p))

                    if 'std_var_TJ14' in self.fct:
                        yy = self.fct['std_var_TJ14'](n_fine, self.n_D, par[i])
                        p_sym = par_symbol(p, eq=False)
                        plot_add_legend(True, n_fine, yy, linestyle=linestyle[i],
                                        color=color[i], label=fr'$\mathbf{{\hat F}}({p_sym})$', linewidth=2)
                        cols.append(self.fct['std_var_TJ14'](n, self.n_D, par[i]))
                        names.append('TJ14({})'.format(p))

                    plot_sth = True

        # Finalize plot

        # Get main axes
        ax = plt.subplot(1, 1, 1)

        if xlog == True:
            fac_xlim = 1.6
            xmin = n[0]/fac_xlim
            xmax = n[-1]*fac_xlim
            ax.set_aspect('auto')
            ax.set_xscale('log')
            flinlog = lambda x: np.log(x)
        else:
            flinlog = lambda x: x
            add_xlim = 5
            xmin = max(n[0] - add_xlim, 0)
            xmax = n[-1] + add_xlim
        # Check whether this is still ok for xlog=True
        plt.xlim(xmin, xmax)

        # Main-axes settings
        plt.xlabel('$n_{\\rm s}$')
        plt.ylabel(stat_notation('std_var'))
        ax.set_yscale('log')
        ax.legend(loc='best', numpoints=1, frameon=False, handlelength=1.3)
        #ax.set_aspect(aspect=1)

	    # x-ticks
        ax = plt.gca().xaxis
        ax.set_major_formatter(ScalarFormatter())
        plt.ticklabel_format(axis='x', style='sci')

        # Dashed vertical line at n_S = n_D
        plt.plot([self.n_D, self.n_D], [8e-9, 1e-1], ':', linewidth=1)

	    # Second x-axis
        x_loc, x_lab = plt.xticks()
        if model in ('quadratic', 'wl'):
            x_loc = [2, 5, 10, 20, 40]
        ax2 = plt.twiny()
        x2_loc = []
        x2_lab = []
        for i, n_S in enumerate(x_loc):
            if n_S > 0:
                x2_loc.append(flinlog(n_S))
                frac = float(self.n_D) / float(n_S)
                if frac > 100:
                    lab = '{:.0f}'.format(frac)
                else:
                    lab = '{:.2g}'.format(frac)
                x2_lab.append(lab)
        ax = plt.gca().xaxis
        # MKDEBUG: The following does not produce labels for model=='wl'
        plt.xticks(x2_loc, x2_lab)
        ax2.set_xlabel('$p / n_{\\rm s}$', size=fs)
        if xlog:
            ax2.set_xscale('log')
            plt.xlim(self.n_D/xmin, self.n_D/xmax)

        # y-scale
        if model == 'wl':
            plt.ylim(1e-6, 1e-3)
        elif model == 'quadratic':
            if ste:
                plt.ylim(1e-5, 1e-2)
            else:
                plt.ylim(1e-7, 1e-3)
        else:
            plt.ylim(2e-8, 1.5e-2)

        ### Output
        if ste:
            outbase = 'std_1{}'.format(self.file_base)
        else:
            outbase = 'std_2{}'.format(self.file_base)

        if plot_sth == True:
            plt.savefig('{}.pdf'.format(outbase), bbox_inches='tight')

        write_ascii(outbase, cols, names)


def set_box_width(boxwidth, xlog, n):
    if boxwidth == None:
        if xlog == False:
            if len(n) > 1:
                box_width = (n[1] - n[0]) / 2
            else:
                box_width = 50
        else:
            if len(n) > 1:
                box_width = np.sqrt(n[1]/n[0]) # ?
            else:  
                box_width = 0.1

    else:
        box_width = boxwidth

    return box_width


def plot_init(n_D, n_R, title=False, fs=9, figsize=None):

    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    fig.subplots_adjust(bottom=0.16)

    ax = plt.gca()
    ax.yaxis.label.set_size(fs)
    ax.xaxis.label.set_size(fs)

    plt.rcParams['font.size'] = fs

    plt.tick_params(axis='both', which='major', labelsize=fs)

    if n_R>0 and title:
        add_title(n_D, n_R, fs, raise_title=True)


def add_title(n_D, n_R, fs, raise_title=False):
    """Adds title to plot."""

    if raise_title == True:
        y = 1.1
    else:
        y = 1

    plt.suptitle('$p$ data points, $n_{{\\rm r}}={}$ runs'.format(n_D, n_R), fontsize=fs, y=y)



def plot_add_legend(do_legend, x, y, linestyle, color='b', label='', linewidth=1):

    if do_legend:
        label = label
    else:
        label = None

    plt.plot(x, y, linestyle, color=color, label=label, linewidth=linewidth)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])



def error(str, val=1, stop=True, verbose=True):
    """Print message str and exits program with code val

    Parameters
    ----------
    str: string
        message
    val: integer
        exit value, default=1
    stop: boolean
        stops program if True (default), continues if False
    verbose: boolean
        verbose output if True (default)

    Returns
    -------
    None
    """

    if stop == True:
        col = '31' # red
    else:
        col = '33' # orange

    if verbose is True:
        #print>>sys.stderr, "\x1b[{}m{}\x1b[0m".format(col, str),
        print("\x1b[{}m{}\x1b[0m".format(col, str), file=sys.stderr, end='')

    if stop is False:
        if verbose is True:
            #print>>sys.stderr,  "\x1b[{}m, continuing...\x1b[0m".format(col),
            #print>>sys.stderr, ''
            print("\x1b[{}m, continuing...\x1b[0m".format(col), file=sys.stderr, end='')
            print(file=sys.stderr)
    else:
        if verbose is True:
            print(file=sys.stderr)
        sys.exit(val)



def check_error_stop(ex_list, verbose=True, stop=False):
    """Check error list and stop if one or more are != 0 and stop=True

    Parameters
    ----------
    ex_list: list of integers
        List of exit codes
    verbose: boolean
        Verbose output, default=True
    stop: boolean
        If False (default), does not stop program

    Returns
    -------
    s: integer
        sum of absolute values of exit codes
    """

    if ex_list is None:
        s = 0
    else:
        s = sum([abs(i) for i in ex_list])


    # Evaluate exit codes
    if s > 0:
        n_ex = sum([1 for i in ex_list if i != 0])
        if verbose is True:
            if len(ex_list) == 1:
                print_color('red', 'The last command returned sum|exit codes|={}'.format(s), end='')
            else:
                print_color('red', '{} of the last {} commands returned sum|exit codes|={}'.format(n_ex, len(ex_list), s), end='')
        if stop is True:
            print_color('red', ', stopping')
        else:
            print_color('red', ', continuing')

        if stop is True:
            sys.exit(s)


    return s



def print_color(col, txt, end='\n'):
    print(txt, end=end)



def warning(str):
    """Prints message to stderr
        """

    error('Warning: ' + str, val=None, stop=False, verbose=True)



def detF(n_D, sig2, delta):
    """Return exact determinant of Fisher matrix.
    """

    det = (n_D/sig2)**2 * delta**2 / 12.0
    return det


def Fisher_error_ana(x, sig2, delta, mode=-1):
    """Return Fisher matrix parameter errors (std), and Fisher matrix determinant, for affine function parameters (a, b)
    """

    n_D = len(x)

    # The four following ways to compute the Fisher matrix errors are statistically equivalent,
    # for a digonal input covariance matrix cov = diag(sigma^2).
    # Note that mode==-1,0 uses the statistical properties mean and variance of the uniform
    # distribution, whereas mode=1,2 uses the actual sample x.

    if mode != -1:

        if mode == 2:
            # numerically using uniform vector x

            Psi = np.diag([1.0 / sig2 for i in range(n_D)])

            # Seems not to work well for a
            F_11 = np.einsum('i,ij,j', x, Psi, x)
            F_22 = np.einsum('i,ij,j', np.ones(shape=n_D), Psi, np.ones(shape=n_D))
            F_12 = np.einsum('i,ij,j', x, Psi, np.ones(shape=n_D))

        elif mode == 1:
            F_11 = sum(x*x) / sig2
            F_12 = sum(x) / sig2
            F_22 = n_D / sig2

        elif mode == 0:
            F_11 = n_D * delta**2 / 12.0 / sig2
            F_12 = 0
            F_22 = n_D / sig2

        det = F_11 * F_22 - F_12**2
        da2 = F_22 / det
        db2 = F_11 / det

    else:
        # mode=-1
        det = detF(n_D, sig2, delta)
        da2 = 12 * sig2 / (n_D * delta**2)
        db2 = sig2 / n_D

    return np.sqrt([da2, db2]), det



def Fisher_ana_quad(ell, f_sky, sigma_eps, nbar_rad2, tilt_fid, ampl_fid, cov_model,
                    ellmode='log', mode=1, templ_dir='.'):
    """Return Fisher matrix for quadratic model with parameters t (tilt) and A (amplitude).
    """


    if mode == 0:

        # Numerical derivatives for testing.
        # 1/2/2019: Gives the same result as mode=1 to 1%

        h   = 0.01
        yp1 = model_quad(np.log10(ell), ampl_fid + h, tilt_fid)
        ym1 = model_quad(np.log10(ell), ampl_fid - h, tilt_fid)
        yp2 = model_quad(np.log10(ell), ampl_fid, tilt_fid + h)
        ym2 = model_quad(np.log10(ell), ampl_fid, tilt_fid - h)

        dy_dt = (yp2 - ym2) / (2*h)
        dy_dA = (yp1 - ym1) / (2*h)

        y = model_quad(np.log10(ell), ampl_fid, tilt_fid)
        D = get_cov_Gauss(ell, y, f_sky, sigma_eps, nbar_rad2)
        D = np.diag(D)
        
    else:

        if ellmode == 'log':
            # Delta_ln_ell = const

            Delta_ln_ell = np.diff(ell) / (ell[:-1]/2 + ell[1:]/2)
            Delta_ln_ell = np.append(Delta_ln_ell, Delta_ln_ell[-1])
            Delta_ell = Delta_ln_ell * ell
        else:
            # Delta_ell = const

            Delta_ell = np.diff(ell)
            Delta_ell = np.append(Delta_ell, Delta_ell[-1])

        # Covariance = diagonal shot-/shape-noise term
        y = model_quad(np.log10(ell), ampl_fid, tilt_fid)
        B = sigma_eps**2 / (2.0 * nbar_rad2)

        N = 1.0 / (f_sky * (2.0 * ell) * Delta_ell)
        D = N * (y + B)**2

        u = np.log10(ell)

        c0    = -6.11568527 + 0.1649            # 5.95 in the paper
        t0    = 1.0 / (1.85132114 / 0.306)      # 1/6.05 in the paper
        a     = -0.17586216
        u0    = shift(tilt_fid)

        # The following two lines are equivalent:
        # dy_dA = dy_dq  * dq_dc * dc_dA
        #       = y ln10 * 1     * 2/ln10 A
        dy_dA = 2 * ampl_fid * 10**(c0 + a * (u-u0)**2 - u)
        #dy_dA = 2.0 * y / ampl_fid

        # dy_dt = dy_dq * dq_du0 * du0_dt
        #       = y ln10 * (-2) a (u -u0) 1/t0
        dy_dt = 1.0 / t0 * (-2.0) * a * (u - u0) * y * np.log(10)

    if cov_model == 'Gauss':
        Psi = np.diag([1.0 / d for d in D])
    elif cov_model == 'Gauss+SSC_BKS17':
        ell_mode = get_ell_mode(ell)
        if ell_mode == 'log':
            cov_SSC_base = 'cov_SSC_rel_log'
        elif ell_mode == 'lin':
            cov_SSC_base = 'cov_SSC_rel_lin'
        cov_SSC_path = '{}/{}.txt'.format(templ_dir, cov_SSC_base)
        func_SSC = 'BKS17'
        cov_SSC = get_cov_SSC(ell, y, cov_SSC_path, 'BKS17')
        cov = np.diag(D) + cov_SSC
        Psi = np.linalg.inv(cov)

    [da2, db2], det = Fisher_num(dy_dt, dy_dA, Psi)

    return np.sqrt([da2, db2]), det


def Fisher_num(y1, y2, Psi):
    """Fisher numerical

    Return Fisher matrix.

    Parameters
    ----------
    y1 : array of float
        d[y_obs] / d[theta_1]
    y2 :  array of float
        d[y_obs] / d[theta_2] 
    Psi : matrix of float
        precision matrix

    Returns
    da2_db2 : array(2) of float
        variance for theta_1, theta_2
    det : double
        Fisher matrix determinant
    """

    # Fisher matrix elements
    F_11   = np.einsum('i,ij,j', y1, Psi, y1)
    F_22   = np.einsum('i,ij,j', y2, Psi, y2)
    F_12   = np.einsum('i,ij,j', y1, Psi, y2)

    # Cramer-Rao, invert Fisher
    det = F_11 * F_22 - F_12**2
    da2 = F_22 / det
    db2 = F_11 / det

    return [da2, db2], det


def my_string_split(string, num=-1, verbose=False, stop=False):
    """Split a *string* into a list of strings. Choose as separator
        the first in the list [space, underscore] that occurs in the string.
        (Thus, if both occur, use space.)

    Parameters
    ----------
    string: string
        Input string
    num: int
        Required length of output list of strings, -1 if no requirement.
    verbose: bool
        Verbose output
    stop: bool
        Stop programs with error if True, return None and continues otherwise

    Returns
    -------
    list_str: string, array()
        List of string on success, and None if failed.
    """

    if string is None:
        return None

    has_space      = string.find(' ')
    has_underscore = string.find('_')

    if has_space != -1:
        # string has white-space
        sep = ' '
    else:
        if has_underscore != -1:
        # string has no white-space but underscore
            sep = '_'
        else:
            # string has neither, consists of one element
            if num == -1 or num == 1:
                # one-element string is ok
                sep = None
                pass
            else:
                error('Neither \' \' nor \'_\' found in string \'{}\', cannot split'.format(string))

    #res = string.split(sep=sep) # python v>=3?
    res = string.split(sep)

    if num != -1 and num != len(res):
        if verbose:
            print >>std.styerr, 'String \'{}\' has length {}, required is {}'.format(string, len(res), num)
        if stop is True:
            sys.exit(2)
        else:
            return None

    return res



def log_command(argv, name=None, close_no_return=True):
    """Write command with arguments to a file or stdout.
       Choose name = 'sys.stdout' or 'sys.stderr' for output on sceen.

    Parameters
    ----------
    argv: array of strings
        Command line arguments
    name: string
        Output file name (default: 'log_<command>')
    close_no_return: bool
        If True (default), close log file. If False, keep log file open
        and return file handler

    Returns
    -------
    log: filehandler
        log file handler (if close_no_return is False)
    """

    if name is None:
        name = 'log_' + os.path.basename(argv[0])

    if name == 'sys.stdout':
        f = sys.stdout
    elif name == 'sys.stderr':
        f = sys.stderr
    else:
        f = open(name, 'w')

    for a in argv:

        # Quote argument if special characters
        if ']' in a or ']' in a:
            a = '\"{}\"'.format(a)

        print(a, file=f, end='')
        print(' ', end='', file=f)

    print('', file=f)

    if close_no_return == False:
        return f

    if name != 'sys.stdout' and name != 'sys.stderr':
        f.close()



def run_cmd(cmd_list, run=True, verbose=True, stop=False, parallel=True, file_list=None, devnull=False):
    """Run shell command or a list of commands using subprocess.Popen().

    Parameters
    ----------

    cmd_list: string, or array of strings
        list of commands
    run: bool
        If True (default), run commands. run=False is for testing and debugging purpose
    verbose: bool
        If True (default), verbose output
    stop: bool
        If False (default), do not stop after command exits with error.
    parallel: bool
        If True (default), run commands in parallel, i.e. call subsequent comands via
        subprocess.Popen() without waiting for the previous job to finish.
    file_list: array of strings
        If file_list[i] exists, cmd_list[i] is not run. Default value is None
    devnull: boolean
        If True, all output is suppressed. Default is False.

    Returns
    -------
    sum_ex: int
        Sum of exit codes of all commands
    """

    if type(cmd_list) is not list:
        cmd_list = [cmd_list]

    if verbose is True and len(cmd_list) > 1:
        print('Running {} commands, parallel = {}'.format(len(cmd_list), parallel))

    ex_list   = []
    pipe_list = []
    for i, cmd in enumerate(cmd_list):

        ex = 0

        if run is True:

            # Check for existing file
            if file_list is not None and os.path.isfile(file_list[i]):
                if verbose is True:
                    print('Skipping command \'{}\', file \'{}\' exists'.format(cmd, file_list[i]))
            else:
                if verbose is True:
                        print('Running command \'{0}\''.format(cmd))

                # Run command
                try:
                    cmds = shlex.split(cmd)
                    if devnull is True:
                        pipe = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
                    else:
                        pipe = subprocess.Popen(cmds)

                    if parallel is False:
                        # Wait for process to terminate
                        pipe.wait()

                    pipe_list.append(pipe)

                    # If process has not terminated, ex will be None
                    #ex = pipe.returncode
                except OSError as e:
                    print('Error: {0}'.format(e.strerror))
                    ex = e.errno

                    check_error_stop([ex], verbose=verbose, stop=stop)

        else:
            if verbose is True:
                print('Not running command \'{0}\''.format(cmd))

        ex_list.append(ex)


    if parallel is True:
        for i, pipe in enumerate(pipe_list):
            pipe.wait()

            # Update exit code list
            ex_list[i] = pipe.returncode

    s = check_error_stop(ex_list, verbose=verbose, stop=stop)

    return s


def linear_dist_data(d2, p):
    """Distance between observed and simulated catalogues using
       least squares between observed and simulated data points y.

    Parameters
    ----------
    d2: array(double, 2)
        simulated catalogue
    p: dictionary
        input parameters

    Returns
    -------
    dist: double
        distance
    """

    if bool(p['xfix']) == False:
        raise ValueError('Parameter xfix needs to be 1 for linear_dist_data distance')

    y_sim = d2[:,1]
    y_obs = p['dataset1'][:,1]

    y_delta = y_sim - y_obs

    # Unweighted distances
    dist    = np.sqrt(sum(y_delta**2))

    return np.atleast_1d(dist)


def linear_dist_data_true_prec(d2, p):
    """Distance between observed and simulated catalogues using
       true inverse covariance.

    Parameters
    ----------
    d2: array(double, 2)
        simulated catalogue
    p: dictionary
        input parameters

    Returns
    -------
    dist: double
        distance
    """

    C_ell_sim = d2[:,1]
    C_ell_obs = p['dataset1'][:,1]

    dC = C_ell_sim - C_ell_obs

    # Unweighted distances
    #dist    = np.sqrt(sum(dC**2))

    # Least squares weighted by covariance
    if 'cov_true_inv' in p:
        cov_inv = p['cov_true_inv']
    else:
        cov_inv = np.loadtxt('cov_true_inv.txt')

    dist = np.einsum('i,ij,j', dC, cov_inv, dC)
    dist = np.sqrt(dist)

    return np.atleast_1d(dist)


def acf_one(C, di, mean, count_zeros=False, mean_std_t=False):
    """Return one value of the auto-correlation function xi(x) of C at argument x=di

    Parameters
    ----------
    C: array(float)
        observed power spectrum
    di: int
        difference of ell-mode indices
    mean: float
        mean value of C, only used if mean_std_t is False
    mean_std_t: bool, optional, default=False
        if True, mean and std depend on t in acf

    Returns
    -------
    xi: float
        auto-correlation function value
    """

    n_D  = len(C)
    # Shift signal and keep to same length (loose entries at high-ell end)
    C1 = C[:n_D - di]
    C2 = C[di:]

    # Normalisation pre-factor, if count_zeros is True, normalisation factor
    # accounts for entire original input data vector length, those which
    # are 'shifted out' are interpreted as zero
    if count_zeros:
        norm = float(n_D)
    else:
        norm = float(n_D - di)

    # Estimate ACF
    if mean_std_t:
        mean1 = C1.mean()
        mean2 = C2.mean()
        std1 = C1.std()
        std2 = C2.std()
        #fac = norm * std1 * std2
        fac = norm
        if fac > 0:
            xi = sum((C1 - mean1) * (C2 - mean2)) / fac
        else:
            xi = 0
    else:
        xi = sum((C1 - mean) * (C2 - mean)) / norm

    return xi


def acf(C, norm=False, count_zeros=False, mean_std_t=False):
    """Return auto-correlation function of C.

    Parameters
    ----------
    C: array(float)
        observed power spectrum
    di: int
        difference of ell-mode indices
    norm: bool, optional, default=False
        if True, acf is normalised by the variance

    Returns
    -------
    xi: array of float
        auto-correlation function value
    """

    mean = C.mean()

    xi = []
    for di in range(len(C)):
        xi.append(acf_one(C, di, mean, count_zeros=count_zeros,
                          mean_std_t=mean_std_t))

    #if norm and not mean_std_t:
    if norm:
        # Var = < C_ell C_ell> = xi(0)

        if xi[0] != 0:
            xi = xi / xi[0]

    return xi


def linear_dist_data_acf2_lin_diag(d2, p):
    """ For testing."""

    return linear_dist_data_acf2(d2, p, mode='xi', diag=True)


def linear_dist_data_acf2_lin(d2, p):

    return linear_dist_data_acf2(d2, p, mode='xi')


def linear_dist_data_acf2_sqr(d2, p):

    return linear_dist_data_acf2(d2, p, mode='xi_square')


def linear_dist_data_acf2_cub(d2, p):

    return linear_dist_data_acf2(d2, p, mode='xi_cub')


def linear_dist_data_acf2(d2, p, mode='xi_square', diag=False):
    """New distance between observed and simulated catalogues using
       the auto-correlation function of the observation"

    Parameters
    ----------
    d2: array(double, 2)
        simulated catalogue
    p: dictionary
        parameters
    mode: bool, optional, default='xi_square'
        mode, one in 'xi', 'xi_square', 'xi_cub''
    diag: bool, optional, default=False
        if True only uses 'diagonal' xi_i=j}

    Returns
    -------
    dist: double
        distance
    """

    C_ell_sim = d2[:,1]
    C_ell_obs = p['dataset1'][:,1]
    if 'xi' in p:
        xi = p['xi']
    else:
        raise ValueError('xi not found in parameter dict')

    d = 0
    n_D = len(C_ell_obs)
    for i in range(n_D):
        for j in range(n_D):
            if diag and i != j:
                continue
            xi_ij = xi[np.abs(i-j)]
            if mode == 'xi':
                term = (C_ell_sim[i] - C_ell_obs[i]) * xi_ij * (C_ell_sim[j] - C_ell_obs[j])
            elif mode == 'xi_square':
                term = (C_ell_sim[i] - C_ell_obs[i]) * xi_ij**2 * (C_ell_sim[j] - C_ell_obs[j])
            elif mode == 'xi_cub':
                term = (C_ell_sim[i] - C_ell_obs[i]) * xi_ij**3 * (C_ell_sim[j] - C_ell_obs[j])
            d = d + term

    if mode == 'xi_square':
        d = np.sqrt(d)
    elif mode == 'xi' or mode == 'xi_cub':
        d = np.abs(d)
    d = np.atleast_1d(d)
    return d


def linear_dist_data_dummy(d2, p):
    """Distance between observed and simulated catalogues using
       least squares between observed and simulated data points y.

    Parameters
    ----------
    d2: array(double, 2)
        simulated catalogue
    p: dictionary
        input parameters

    Returns
    -------
    dist: double
        distance
    """

    if bool(p['xfix']) == False:
        raise ValueError('Parameter xfix needs to be 1 for linear_dist_data distance')

    y_sim = d2[:,1]
    y_obs = p['dataset1'][:,1]

    y_delta = y_sim - y_obs

    # Unweighted distances
    dist    = np.sqrt(sum(y_delta**2))

    return np.atleast_1d(dist)

