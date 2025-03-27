#!/usr/bin/env python3

import numpy as np
from scipy.stats import norm, uniform, multivariate_normal
import pylab as plt
import sys
import os
import re
import time
import copy
from astropy.table import Table, Column
from astropy.io import ascii
from optparse import OptionParser
from optparse import OptionGroup

sys.path.append('../..')
from CorrMatrix_ABC.covest import *


"""The following functions correspond to the expectation values for Gaussian
   distributions. See Taylor, Joachimi & Kitching (2013), Taylor & Joachimi (2014).
"""

def A_corr(n_S, n_D):
    """Return TJK13 (28), this is A/alpha^2.
    """
    
    A_c =  1.0 / ((n_S - n_D - 1.0) * (n_S - n_D - 4.0))

    return A_c



def tr_N_m1_ML(n, n_D, par):
    """Maximum-likelihood estimate of inverse covariance normalised trace.
       TJK13 (24).
       This is 1/alpha.
    """

    #return [alpha(n_S, n_D) * par for n_S in n]
    return [1/alpha_new(n_S, n_D) * par for n_S in n]



def par_fish(n, n_D, par):
    """Parameter RMS from Fisher matrix using biased precision matrix.
       Square root of expectation value of TJK13 (43), follows from (25).
    """

    #return [np.sqrt(1.0 / alpha(n_S, n_D)) * par for n_S in n]
    return [np.sqrt(alpha_new(n_S, n_D)) * par for n_S in n]
 

def params_default():
    """Set default parameter values.

    Parameters
    ----------
    None

    Returns
    -------
    p_def: class mkstuff.param
        parameter values
    """

    n_D = 750

    p_def = param(
        n_D = n_D,
        n_R = 50,
        spar = '1.0 0.0',
        sig2 = 5.0,
        mode  = 's',
        do_fit_stan = False,
        do_fish_ana = False,
        likelihood = 'norm_deb',
        n_jobs = 1,
        random_seed = False,
        plot_style = 'talk'
    )

    return p_def



def parse_options(p_def):
    """Parse command line options.

    Parameters
    ----------
    p_def: class mkstuff.param
        parameter values

    Returns
    -------
    options: tuple
        Command line options
    args: string
        Command line string
    """

    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    parser.add_option('-D', '--n_D', dest='n_D', type='int', default=p_def.n_D,
        help='Number of data points, default={}'.format(p_def.n_D))
    parser.add_option('-R', '--n_R', dest='n_R', type='int', default=p_def.n_R,
        help='Number of runs per simulation, default={}'.format(p_def.n_R))
    parser.add_option('', '--n_S', dest='str_n_S', type='string', default=None,
        help='Array of n_S, default=None. If not given, 10 log numbers between n_D+5 and 10 n_D are used')


    parser.add_option('-p', '--par', dest='spar', type='string', default=p_def.spar,
        help='list of parameter values, default=\'{}\''.format(p_def.spar))
    parser.add_option('-s', '--sig2', dest='sig2', type='float', default=p_def.sig2,
        help='variance of Gaussian, default=\'{}\''.format(p_def.sig2))

    parser.add_option('', '--fit_stan', dest='do_fit_stan', action='store_true',
        help='Run stan for MCMC fitting, default={}'.format(p_def.do_fit_stan))
    parser.add_option('', '--fish_ana', dest='do_fish_ana', action='store_true',
        help='Calculate analytical Fisher matrix, default={}'.format(p_def.do_fish_ana))
    parser.add_option('-L', '--like', dest='likelihood', type='string', default=p_def.likelihood,
        help='Likelihood for MCMC, one in \'norm_deb\'|\'norm_biased\'|\'SH\', default=\'{}\''.format(p_def.likelihood))
    parser.add_option('', '--sig_var_noise', dest='sig_var_noise', type='string',
        help='MCMC \'noise\' to be subtracted from sigma(var) plots for fits, default=None')

    parser.add_option('-m', '--mode', dest='mode', type='string', default=p_def.mode,
        help='Mode: \'s\'=simulate, \'r\'=read, default={}'.format(p_def.mode))
    parser.add_option('-a', '--add_simulations', dest='add_simulations', action='store_true', help='add simulations to existing files')
    parser.add_option('-r', '--random_seed', dest='random_seed', action='store_true', help='random seed')

    parser.add_option('-n', '--n_jobs', dest='n_jobs', type='int', default=p_def.n_jobs,
        help='Number of parallel jobs, default={}'.format(p_def.n_jobs))

    parser.add_option('-b', '--boxwidth', dest='boxwidth', type='float', default=None,
        help='box width for box plot, default=None, determined from n_S array')
    parser.add_option('', '--plot_style', dest='plot_style', type='string', default=p_def.plot_style,
        help='plot style, one in \'paper\'|\'talk\' (default)')

    parser.add_option('-v', '--verbose', dest='verbose', action='store_true', help='verbose output')

    options, args = parser.parse_args()

    return options, args



def check_options(options):
    """Check command line options.

    Parameters
    ----------
    options: tuple
        Command line options

    Returns
    -------
    erg: bool
        Result of option check. False if invalid option value.
    """

    if options.add_simulations == True:
        if options.random_seed is False:
            error('Adding simulations (-a) only possible with random seed (-r)')
        if re.search('r', options.mode) is not None:
            error('Adding simulations (-a) is not  possible in read mode (-m r)')
        if not os.path.isfile('sigma_ML.txt'):
            error('Previous file \'sigma_ML.txt\' necessary for option -a (adding simulations)')

    if re.search('s', options.mode) is not None and re.search('r', options.mode) is not None:
        error('Simulation and read mode (-m rs) not possible simultaneously') 

    see_help = 'See option \'-h\' for help.'

    return True



def update_param(p_def, options):
    """Return default parameter, updated and complemented according to options.
    
    Parameters
    ----------
    p_def:  class mkstuff.param
        parameter values
    optiosn: tuple
        command line options
    
    Returns
    -------
    param: class mkstuff.param
        updated paramter values
    """

    param = copy.copy(p_def)

    # Update keys in param according to options values
    for key in vars(param):
        if key in vars(options):
            setattr(param, key, getattr(options, key))

    # Add remaining keys from options to param
    for key in vars(options):
        if not key in vars(param):
            setattr(param, key, getattr(options, key))

    # Do extra stuff if necessary
    par = my_string_split(param.spar, num=2, verbose=param.verbose, stop=True)
    options.par = [float(p) for p in par]
    options.a = options.par[0]
    options.b = options.par[1]

    tmp = my_string_split(param.sig_var_noise, num=2, verbose=param.verbose, stop=True)
    if tmp != None:
        param.sig_var_noise = [float(s) for s in tmp]
    else:
        param.sig_var_noise = None

    if options.str_n_S == None:
        param.n_S = None
    else:
        str_n_S_list = my_string_split(options.str_n_S, verbose=False, stop=True)
        param.n_S = [int(str_n_S) for str_n_S in str_n_S_list]


    return param



def numbers_from_file(file_base, npar):
    """Return number of simulation and runs as read from simulation file"""

    dat = ascii.read('{}.txt'.format(file_base))

    # Number of simulation cases = number of rows in file
    n_n_S = len(dat['n_S'])

    # Number of runs: number of columns, subtract one (n_S), divide by 2 (mean+std),
    # devide by number of parameters
    n_R   = (len(dat.keys()) - 1) / 2 / npar

    return n_n_S, n_R



def Fisher_ana_ele(r, s, x, Psi):
    """Return analytical Fisher matrix element (r, s).

    Parameters
    ----------
    r, s: integer
        indices of matrix, r,s in {0,1}
    x: array of float
        abcissa for data vector y(x)=a*x+b
    Psi: matrix
        precision matrix

    Returns
    -------
    f_rs: float
        Fisher matrix element (r, s)
    """

    n_D = len(x)
    v = np.zeros(shape = (2, n_D))
    for i in (r, s):
        if i == 0:
            v[i] = x
        elif i == 1:
            v[i] = np.ones(shape=n_D)
        else:
            print('Invalid index {}'.format(i))
            sys.exit(1)

    f_rs = np.einsum('i,ij,j', v[r], Psi, v[s])

    return f_rs


def Fisher_num_ele(r, s, x, a, b, Psi):
    """Return numerical Fisher matrix element (r, s).

    Parameters
    ----------
    r, s: integer
        indices of matrix, r,s in {0,1}
    x: array of float
        x-values on which data vector is evaluated
    a, b: double
        parameters
    Psi: matrix
        precision matrix

    Returns
    -------
    f_rs: float
        Fisher matrix element (r, s)
    """


    if r == 0:
        h = 0.1 * a
        dy_dr = ((a+h) * x + b - ((a-h) * x + b)) / (2*h)
    else:
        h = 0.1
        dy_dr = (a * x + b+h - (a * x + b-h)) / (2*h)

    if s == 0:
        h = 0.1 * a
        dy_ds = ((a+h) * x + b - ((a-h) * x + b)) / (2*h)
    else:
        h = 0.1
        dy_ds = (a * x + b+h - (a * x + b-h)) / (2*h)

    f_rs = np.einsum('i,ij,j', dy_dr, Psi, dy_ds)

    return f_rs



def Fisher_error(F):
    """Return errors (Cramer-Rao bounds) from Fisher matrix

    Parameters
    ----------
    F: matrix of float
        Fisher matrix

    Returns
    -------
    d: array of float
        vector of parameter errors
    """

    Finv = np.linalg.inv(F)
    d    = np.sqrt(np.diag(Finv))

    return d



def Fisher_ana(y, Psi):
    """Return analytical Fisher matrix

    Parameters
    ----------
     y: array of float
        data vector
    Psi: matrix
        precision matrix

    Returns
    -------
    f: matrix
        Fisher matrix
    """

    f = np.zeros(shape = (2, 2))
    for r in (0, 1):
        for s in (0, 1):
            f[r,s] = Fisher_ana_ele(r, s, y, Psi)

    return f



def Fisher_num(x, a, b, Psi):
    """Return numerical Fisher matrix

    Parameters
    ----------
    x: array of float
        x-values of data vector
    a, b: double
        parameters
    Psi: matrix
        precision matrix

    Returns
    -------
    f: matrix
        Fisher matrix
    """

    f = np.zeros(shape = (2, 2))
    for r in (0, 1):
        for s in (0, 1):
            f[r,s] = Fisher_num_ele(r, s, x, a, b, Psi)

    return f



def get_cov_ML_by_hand(y2, ymean, cov, n_S, n_D):
    """ Element-by-element compuation of covariance, for checking. Use:
        ymean = np.mean(y2, axis=0)
        cov_est = get_cov_ML_by_hand(y2, ymean, cov, size, n_D)
        to get same result as `get_cov_ML`.
    """

    cov_est = np.zeros(shape=(n_D, n_D))
    if n_D > 1:

        #didj = (y2 - ymean)*(y2 - ymean)
        #cov_est = didj.sum(axis=0)

        for i in range(n_D):
            for j in range(i, n_D):
                cov_est[i,j] = sum((y2[:,i] - ymean[i]) * (y2[:,j] - ymean[j]))
        for i in range(n_D):
            for j in range(0, i):
                cov_est[i,j] = cov_est[j,i]
        cov_est = cov_est / (n_S - 1.0)

    else:

        cov_est[0,0] = sum((y2 - ymean) * (y2 - ymean)) / (n_S - 1.0)

    return cov_est



def get_cov_ML(mean, cov, size):

    y2 = multivariate_normal.rvs(mean=mean, cov=cov, size=size)
    # y2[:,j] = realisations for j-th data entry
    # y2[i,:] = data vector for i-th realisation

    cov_est = np.cov(y2, rowvar=False)

    if size > 1:
        pass
    else:
        cov_est = [[cov_est]]

    return cov_est



def debias_cov(cov_est_inv, n_S):
    """Return debiased inverse covariance estimate
    """

    n_D   = cov_est_inv.shape[0]
    if n_S - n_D - 2 <= 0:
        print('Number of simulations {} too small for data dimension = {}, resulting in singular covariance'.format(n_S, n_D))
    alpha = (n_S - n_D - 2.0) / (n_S - 1.0)

    return alpha * cov_est_inv



def bias_cov(cov_est, n_S):
    """Bias covariance matrix such that the inverse of the returned matrix is debiased.
       This multiplies cov_est with the inverse factor compared to bias_inv_cov.
    """

    n_D   = cov_est.shape[0]
    if n_S - n_D - 2 <= 0:
        print('Number of simulations {} too small for data dimension = {}, resulting in singular covariance'.format(n_S, n_D))
    alpha = (n_S - n_D - 2.0) / (n_S - 1.0)

    return 1/alpha * cov_est


def fit_corr_inv_true(x1, cov_true, sig2, n_jobs=3):
    """
    Generates one draw from a multivariate normal distribution
    and performs the linear fit using the true inverse 
    diagonal covariance

    input:  x1, mean of multivariate normal distribution - vector of floats
            sig2, variance
            n_jobs, number of parallel jobs

    output: fit, Stan fitting object
    """

    # Fit
    toy_data = {}                  # build data dictionary
    toy_data['nobs'] = len(x1)     # sample size = n_D
    toy_data['x'] = x1             # explanatory variable

    # cov = covariance of the data!
    y = multivariate_normal.rvs(mean=x1, cov=cov_true, size=1)
    toy_data['y'] = y              # response variable, here one realisation

    # set estimated covariance matrix for fitting
    cov_inv = np.diag([1.0/sig2 for i in range(len(x1))])
    toy_data['cov_inv'] = cov_inv

    # STAN code
    # the fitting code does not believe that observations are correlated!
    stan_code = """
    data {
        int<lower=0> nobs;
        vector[nobs] x;
        vector[nobs] y; 
        matrix[nobs, nobs] cov_inv;
    }
    parameters {
        real a;
        real b;                                                              
    }
    model {
        vector[nobs] mu;
        real chi2;

        mu = b + a * x;

        chi2 = (y - mu)' * cov_inv * (y - mu);

        target += -chi2/2;
    }
    """

    import pystan
    start = time.time()
    fit = pystan.stan(model_code=stan_code, data=toy_data, iter=2000, chains=n_jobs, verbose=False, n_jobs=n_jobs)

    end = time.time()

    #elapsed = end - start
    #print 'elapsed time = ' + str(elapsed)

    return fit



def fit_corr(x1, cov_true, cov_est, n_jobs=3):
    """
    Generates one draw from a multivariate normal distribution
    and performs the linear fit taking the correlation estimated 
    from simulations into consideration.

    input:  x1, mean of multivariate normal distribution - vector of floats
            cov_true, square covariance matrix for the simulation
            cov_est, square covariance matrix for the fitting
            n_jobs, number of parallel jobs

    output: fit, Stan fitting object
    """

    # Fit
    toy_data = {}                  # build data dictionary
    toy_data['nobs'] = len(x1)     # sample size = n_D
    toy_data['x'] = x1             # explanatory variable

    # cov = covariance of the data!
    y = multivariate_normal.rvs(mean=x1, cov=cov_true, size=1)
    toy_data['y'] = y              # response variable, here one realisation

    # set estimated covariance matrix for fitting
    toy_data['cov_est'] = cov_est

    # STAN code
    # the fitting code does not believe that observations are correlated!
    stan_code = """
    data {
        int<lower=0> nobs;                                 
        vector[nobs] x;                       
        vector[nobs] y;   
        matrix[nobs, nobs] cov_est;                    
    }
    parameters {
        real a;
        real b;                                                              
    }
    model {
        vector[nobs] mu;

        mu = b + a * x;

        y ~ multi_normal(mu, cov_est);             // Likelihood function
    }
    """

    import pystan
    start = time.time()
    fit = pystan.stan(model_code=stan_code, data=toy_data, iter=2000, chains=n_jobs, verbose=False, n_jobs=n_jobs)

    # Testing: fast call to pystan
    #fit = pystan.stan(model_code=stan_code, data=toy_data, iter=1, chains=1, verbose=False, n_jobs=n_jobs)

    end = time.time()

    #elapsed = end - start
    #print 'elapsed time = ' + str(elapsed)

    return fit


def fit_corr_SH(x1, cov_true, cov_est_inv, n_jobs=3):
    """
    Generates one draw from a multivariate student-t distribution
    (see Sellentin & Heavens 2015)
    and performs the linear fit taking the correlation estimated 
    from simulations into consideration.

    input:  x1, mean of multivariate normal distribution - vector of floats
            cov_true, square covariance matrix for the simulation
            cov_est_inv, inverse square covariance matrix for the fitting
            n_jobs, number of parallel jobs

    output: fit, Stan fitting object
    """

    # Fit
    toy_data = {}                  # build data dictionary
    toy_data['nobs'] = len(x1)     # sample size = n_D
    toy_data['x'] = x1             # explanatory variable

    # cov = covariance of the data!
    y = multivariate_normal.rvs(mean=x1, cov=cov_true, size=1)
    toy_data['y'] = y              # response variable, here one realisation

    # set estimated inverse covariance matrix for fitting
    toy_data['cov_est_inv'] = cov_est_inv

    # STAN code
    # the fitting code does not believe that observations are correlated!
    stan_code = """
    data {
        int<lower=0> nobs;                                 
        vector[nobs] x;                       
        vector[nobs] y;   
        matrix[nobs, nobs] cov_est_inv;                    
    }
    parameters {
        real a;
        real b;                                                              
    }
    model {
        vector[nobs] mu;
        real chi2;

        mu = b + a * x;

        # Sellentin & Heavens debiasing scheme:
        # Replace normal likelihood with t-distribution
        # y ~ (1 + (log_normal(mu, cov_est)) / (1 + n_S))^(-n_S/2)  

        chi2 = (y - mu)' * cov_est_inv * (y - mu);

        # target += log-likelihood
        target += log(1.0 + chi2/(1.0 + nobs)) * -nobs/2.0;
    }
    """


    import pystan
    start = time.time()
    fit = pystan.stan(model_code=stan_code, data=toy_data, iter=2000, chains=n_jobs, verbose=False, n_jobs=n_jobs)
    end = time.time()

    #elapsed = end - start
    #print 'elapsed time = ' + str(elapsed)

    return fit



def set_fit_MCMC(fit, res, i, run):
    """Set mean and std of fit according to stan output res.
    
    Parameters
    ----------
    fit: class Results
        fit data
    res: stan fitting object
        fit output
    i: int
        current n_S index
    run: int
        current run number

    Returns
    -------
    None
    """

    la  = res.extract(permuted=True)
    par  = []
    dpar = []
    for p in fit.par_name:
        par.append(np.mean(la[p]))
        dpar.append(np.std(la[p]))

    fit.set(par, i, run, which='mean')
    fit.set(dpar, i, run, which='std')



def simulate(x1, yreal, n_S_arr, sigma_ML, sigma_m1_ML, sigma_m1_ML_deb, fish_ana, fish_num, fish_deb, fit_norm, fit_SH, options):
    """Simulate data"""
        
    if options.verbose == True:
        print('Creating {} simulations with {} runs each'.format(len(n_S_arr), options.n_R))

    cov = np.diag([options.sig2 for i in range(options.n_D)])            # *** cov of the data in the same catalog! ***

    # Go through number of simulations
    for i, n_S in enumerate(n_S_arr):

        if options.verbose == True:
            print('{}/{}: n_S={}'.format(i+1, len(n_S_arr), n_S))

        # Loop over realisations
        for run in range(options.n_R):

            cov_est = get_cov_ML(yreal, cov, n_S)

            # Normalised trace
            this_sigma_ML = np.trace(cov_est) / options.n_D
            sigma_ML.set([this_sigma_ML], i, run, which='mean')

            # Biased inverse
            cov_est_inv = np.linalg.inv(cov_est)

            this_sigma_m1_ML = np.trace(cov_est_inv) / options.n_D
            sigma_m1_ML.set([this_sigma_m1_ML], i, run, which='mean')

            # Debiased inverse
            cov_est_inv_deb  = debias_cov(cov_est_inv, n_S)
            this_sigma_m1_ML = np.trace(cov_est_inv_deb) / options.n_D
            sigma_m1_ML_deb.set([this_sigma_m1_ML], i, run, which='mean')


            ### Fisher matrix ###
            # analytical
            if options.do_fish_ana == True:
                F = Fisher_ana(x1, cov_est_inv)   # Bug fix 05/10, 1st arg should be x1, not yreal
                dpar = Fisher_error(F)
                fish_ana.set(dpar, i, run, which='std')

            # numerical
            F = Fisher_num(x1, options.a, options.b, cov_est_inv)
            dpar = Fisher_error(F)
            fish_num.set(dpar, i, run, which='std')
            fish_num.F[i,run] = F

            # using debiased inverse covariance estimate
            cov_est_inv_debiased = debias_cov(cov_est_inv, n_S)
            F = Fisher_num(x1, options.a, options.b, cov_est_inv_debiased)
            dpar = Fisher_error(F)
            fish_deb.set(dpar, i, run, which='std')

            # MCMC fit of Parameters
            if options.do_fit_stan == True:
                if re.search('norm_biased', options.likelihood) is not None:
                    if options.verbose == True:
                        print('Running MCMC with mv normal likelihood and biased inverse covariance')

                    res = fit_corr(x1, cov, cov_est, n_jobs=options.n_jobs)
                    set_fit_MCMC(fit_norm, res, i, run)

                elif re.search('norm_deb', options.likelihood) is not None:
                    if options.verbose == True:
                        print('Running MCMC with mv normal likelihood and debiased inverse covariance')

                    cov_est_biased  = bias_cov(cov_est, n_S)
                    res = fit_corr(x1, cov, cov_est_biased, n_jobs=options.n_jobs)

                    # For testing sampling noise: Using true covariance in likelihood
                    #res = fit_corr(x1, cov, cov, n_jobs=options.n_jobs)

                    # For testing sampling noise: Using true inverse covariance in likelihood
                    #print('MKDEBUG: cc branch, use true *inverse* cov for testing')
                    #res = fit_corr_inv_true(x1, cov, options.sig2, n_jobs=options.n_jobs)

                    set_fit_MCMC(fit_norm, res, i, run)

                if re.search('SH', options.likelihood) is not None:
                    if options.verbose == True:
                        print('Running MCMC with Sellentin&Heavens (SH) likelihood')
                    res = fit_corr_SH(x1, cov, cov_est_inv, n_jobs=options.n_jobs)
                    set_fit_MCMC(fit_SH, res, i, run)

                

def write_to_file(n_S_arr, sigma_ML, sigma_m1_ML, sigma_m1_ML_deb, fish_ana, fish_num, fish_deb, \
                  fit_norm_num, fit_norm_deb, fit_SH, options):
    """Write simulated runs to files"""

    if options.add_simulations == True:
        if options.verbose == True:
            print('Reading previous simulations from disk')

        # Initialise results
        n_S, n_R         = get_n_S_R_from_fit_file(sigma_ML.file_base, npar=1)
        n_n_S            = len(n_S)

        sigma_ML_prev    = Results(sigma_ML.par_name, n_n_S, n_R, file_base=sigma_ML.file_base)
        sigma_m1_ML_prev = Results(sigma_m1_ML.par_name, n_n_S, n_R, file_base=sigma_m1_ML.file_base)
        sigma_m1_ML_deb_prev = Results(sigma_m1_ML_deb.par_name, n_n_S, n_R, file_base=sigma_m1_ML_deb.file_base)
        fish_ana_prev    = Results(fish_ana.par_name, n_n_S, n_R, file_base=fish_ana.file_base, fct=fish_ana.fct)
        fish_num_prev    = Results(fish_num.par_name, n_n_S, n_R, file_base=fish_num.file_base, fct=fish_num.fct, yscale='linear')
        fish_deb_prev    = Results(fish_deb.par_name, n_n_S, n_R, file_base=fish_deb.file_base, fct=fish_deb.fct)
        fit_norm_num_prev = Results(fit_norm_num.par_name, n_n_S, n_R, file_base=fit_norm_num.file_base, fct=fit_norm_num.fct)
        fit_norm_deb_prev = Results(fit_norm_deb.par_name, n_n_S, n_R, file_base=fit_norm_deb.file_base, fct=fit_norm_deb.fct)
        fit_SH_prev      = Results(fit_SH.par_name, n_n_S, n_R, file_base=fit_SH.file_base, fct=fit_SH.fct)

        # Fill results from files
        read_from_file(sigma_ML_prev, sigma_m1_ML_prev, sigma_m1_ML_deb_prev, fish_ana_prev, fish_num_prev, fish_deb_prev, \
                       fit_norm_num_prev, fit_norm_deb_prev, fit_SH_prev, options)

        # Add new results
        sigma_ML.append(sigma_ML_prev)
        sigma_m1_ML.append(sigma_m1_ML_prev)
        sigma_m1_ML_deb.append(sigma_m1_ML_deb_prev)
        fish_ana.append(fish_ana_prev)
        fish_num.append(fish_num_prev)
        fish_deb.append(fish_deb_prev)
        fit_norm_num.append(fit_norm_num_prev)
        fit_norm_deb.append(fit_norm_deb_prev)
        fit_SH.append(fit_SH_prev)

    if options.verbose == True:
        print('Writing simulations to disk')

    if options.do_fish_ana == True:
        fish_ana.write_mean_std(n_S_arr)

    fish_num.write_mean_std(n_S_arr)
    fish_num.write_Fisher(n_S_arr)
    fish_deb.write_mean_std(n_S_arr)
    #fish_deb.write_Fisher(n_S_arr)

    if options.do_fit_stan == True:
        if re.search('norm_biased', options.likelihood) is not None:
            fit_norm_num.write_mean_std(n_S_arr)
        if re.search('norm_deb', options.likelihood) is not None:
            fit_norm_deb.write_mean_std(n_S_arr)
        if re.search('SH', options.likelihood) is not None:
            fit_SH.write_mean_std(n_S_arr)

    sigma_ML.write_mean_std(n_S_arr)
    sigma_m1_ML.write_mean_std(n_S_arr)
    sigma_m1_ML_deb.write_mean_std(n_S_arr)



def read_from_file(sigma_ML, sigma_m1_ML, sigma_m1_ML_deb, fish_ana, fish_num, fish_deb, \
                   fit_norm_num, fit_norm_deb, fit_SH, param):
    """Read simulated runs from files"""

    if param.verbose == True:
        print('Reading simulations from disk')

    if param.do_fish_ana == True:
        fish_ana.read_mean_std()
    fish_num.read_mean_std(verbose=param.verbose)
    fish_num.read_Fisher()
    fish_deb.read_mean_std(verbose=param.verbose)
    #fish_deb.read_Fisher()

    if param.do_fit_stan:
        if re.search('norm_biased', param.likelihood) is not None:
            fit_norm_num.read_mean_std(verbose=param.verbose)
        if re.search('norm_deb', param.likelihood) is not None:
            fit_norm_deb.read_mean_std(verbose=param.verbose)
        if re.search('SH', param.likelihood) is not None:
            fit_SH.read_mean_std(verbose=param.verbose)

    sigma_ML.read_mean_std(npar=1, verbose=param.verbose)
    sigma_m1_ML.read_mean_std(npar=1, verbose=param.verbose)
    sigma_m1_ML_deb.read_mean_std(npar=1, verbose=param.verbose)

    

# Main program
def main(argv=None):
    """Main program.
    """


    # Set default parameters
    p_def = params_default()

    # Command line options
    options, args = parse_options(p_def)
    # Without option parsing, this would be: args = argv[1:]

    if check_options(options) is False:
        return 1

    param = update_param(p_def, options)


    # Save calling command
    log_command(argv)
    if options.verbose:
        log_command(argv, name='sys.stderr')


    if options.verbose is True:
        print('Start program {}'.format(os.path.basename(argv[0])))


    ### Start main program ###

    if options.random_seed is False:
        np.random.seed(1056)                 # set seed to replicate example


    ## Parameters

    # Fixed default parameters

    # Parameter name list (for plots)
    par_name = ['a', 'b']

    # Name of 'trace' function (for plots of the covariance)
    tr_name  = ['tr']

    # Width of uniform distribution for ordinate x
    delta    = 200


    # Default parameters, can be changed by command-line arguments

    # Smallest number of simulation, min(n_s)
    n_S_min_default = param.n_D + 5

    # Largest number of simulations, max(n_S)
    f_n_S_max_default = 10.0

    # Number of cases of simulations, #{n_S}
    n_n_S_default = 10


    # Return number of simulations
    n_S_arr, n_n_S = get_n_S_arr(n_S_min_default, param.n_D, f_n_S_max_default, n_n_S_default, n_S=param.n_S)

    # Display n_S array and exit
    if re.search('d', param.mode) is not None:
        print('n_S =', n_S_arr)
        return 0


    # Initialisation of results
    sigma_ML = Results(tr_name, n_n_S, options.n_R, file_base='sigma_ML')
    sigma_m1_ML = Results(tr_name, n_n_S, options.n_R, file_base='sigma_m1_ML', yscale='log', fct={'mean': tr_N_m1_ML})
    sigma_m1_ML_deb = Results(tr_name, n_n_S, options.n_R, file_base='sigma_m1_ML_deb', fct={'mean': no_bias})

    fish_ana = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_ana', yscale='log', fct={'std': par_fish})

    fish_num = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_num', yscale='log', \
                       fct={'std': par_fish, 'std_var_TJK13': std_fish_biased_TJK13, 'std_var_TJ14': std_fish_biased_TJ14})
    fish_deb = Results(par_name, n_n_S, options.n_R, file_base='std_Fisher_deb', yscale='log', \
                       fct={'std': no_bias, 'std_var_TJK13': std_fish_deb, 'std_var_TJ14': std_fish_deb_TJ14})
    fit_norm_num = Results(par_name, n_n_S, options.n_R, file_base='mean_std_fit_norm', yscale=['linear', 'log'],
                       fct={'std': par_fish, 'std_var_TJK13': std_fish_biased_TJK13, 'std_var_TJ14': std_fish_biased_TJ14})
    fit_norm_deb = Results(par_name, n_n_S, options.n_R, file_base='mean_std_fit_norm_deb', yscale=['linear', 'log'],
                       fct={'std': no_bias, 'std_var_TJ14': std_fish_deb_TJ14})
    fit_SH = Results(par_name, n_n_S, options.n_R, file_base='mean_std_fit_SH', yscale=['linear', 'log'],
                       fct={'std': par_fish_SH})

    # Data
    x1 = uniform.rvs(loc=-delta/2, scale=delta, size=options.n_D)        # exploratory variable
    x1.sort()
    yreal = options.a * x1 + options.b

    # Create simulations
    if re.search('s', options.mode) is not None:

        if re.search('norm_biased', options.likelihood) is not None:
            fit_norm = fit_norm_num
        elif re.search('norm_deb', options.likelihood) is not None:
            fit_norm = fit_norm_deb
        else:
            fit_norm = fit_norm_num  # Dummy variable, could be changed to make more error-proof
 
        simulate(x1, yreal, n_S_arr, sigma_ML, sigma_m1_ML, sigma_m1_ML_deb, fish_ana, fish_num, fish_deb, fit_norm, fit_SH, options) 

        write_to_file(n_S_arr, sigma_ML, sigma_m1_ML, sigma_m1_ML_deb, fish_ana, fish_num, fish_deb, \
                      fit_norm_num, fit_norm_deb, fit_SH, options)


    # Read simulations
    if re.search('r', options.mode) is not None:

        read_from_file(sigma_ML, sigma_m1_ML, sigma_m1_ML_deb, fish_ana, fish_num, fish_deb, \
                       fit_norm_num, fit_norm_deb, fit_SH, param)


    mode = -1
    dpar_exact, det = Fisher_error_ana(x1, options.sig2, delta, mode=mode)


    # Make plots
    if options.verbose == True:
        print('Creating plots')

    if options.do_fish_ana == True:
        fish_ana.plot_mean_std(n_S_arr, options.n_D, par={'std': dpar_exact})
    fish_num.plot_mean_std(n_S_arr, options.n_D, par={'std': dpar_exact})
    fish_deb.plot_mean_std(n_S_arr, options.n_D, par={'std': dpar_exact})

    if options.do_fit_stan == True:
        if re.search('norm_biased', options.likelihood) is not None:
            fit_norm_num.plot_mean_std(n_S_arr, options.n_D, par={'mean': options.par, 'std': dpar_exact}, boxwidth=param.boxwidth)
        if re.search('norm_deb', options.likelihood) is not None:
            fit_norm_deb.plot_mean_std(n_S_arr, options.n_D, par={'mean': options.par, 'std': dpar_exact}, boxwidth=param.boxwidth)
        if re.search('SH', options.likelihood) is not None:
            fit_SH.plot_mean_std(n_S_arr, options.n_D, par={'mean': options.par, 'std': dpar_exact}, boxwidth=param.boxwidth)

    dpar2 = dpar_exact**2

    if options.plot_style == 'talk':
        title = True
    else:
        title = False

    fish_num.plot_std_var(n_S_arr, options.n_D, par=dpar2, title=title)

    fish_deb.plot_std_var(n_S_arr, options.n_D, par=dpar2, title=title)
    if options.do_fit_stan == True:
        if re.search('norm_biased', options.likelihood) is not None:
            fit_norm_num.plot_std_var(n_S_arr, options.n_D, par=dpar2, sig_var_noise=param.sig_var_noise, title=title)
        if re.search('norm_deb', options.likelihood) is not None:
            fit_norm_deb.plot_std_var(n_S_arr, options.n_D, par=dpar2, sig_var_noise=param.sig_var_noise, title=title)
        if re.search('SH', options.likelihood) is not None:
            fit_SH.plot_std_var(n_S_arr, options.n_D, par=dpar2, sig_var_noise=param.sig_var_noise, title=title)

    sigma_ML.plot_mean_std(n_S_arr, options.n_D, par={'mean': [options.sig2]}, boxwidth=param.boxwidth, ylim=[4.9, 5.1])

    # Precision matrix trace
    f_mean = 1/options.sig2

    sigma_m1_ML.plot_mean_std(n_S_arr, options.n_D, par={'mean': [f_mean]}, boxwidth=param.boxwidth)
    sigma_m1_ML_deb.plot_mean_std(n_S_arr, options.n_D, par={'mean': [f_mean]}, boxwidth=param.boxwidth)

    ### End main program

    if options.verbose is True:
        print('Finish program {}'.format(os.path.basename(argv[0])))

    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv))

