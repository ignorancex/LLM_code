##### Measuring Diagnostic Test Performance Using Imperfect Reference Tests: A Partial Identification Approach #####
### Functions script
## Developed by: Filip Obradović. Email: obradovicfilip@u.northwestern.edu

"""This script contains all functions needed to produce the results"""

############################################# Import Modules ###########################################################

import numpy as np
from scipy.stats import norm
import statsmodels.stats.proportion as stats
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from joblib import Parallel, delayed
from scipy.spatial import ConvexHull
from itertools import product
import pandas as pd

# Graph Options
font = font_manager.FontProperties(family='Times New Roman', style='normal', size=20)  # Set font for graphs
tol = 10**-10 # Tolerance for maximum

# Other Options
np.seterr(divide='ignore', invalid='ignore') # Quiets error messages for individual bootstrap draws that may have 0/0


############################################ Defining functions ########################################################


def estimates(s1, s0, t1r1,t1r0,t0r1,t0r0):
    """
    Returns the point estimates of probabilities needed to form the bounds.

    :param t1r1: Number of positives on both tests
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test
    :param t0r0: Number of negatives on both tests
    :param s1: Known sensitivity of the reference test. Float scalar
    :return: Tuple of needed probability estimates
    """
    N = (t0r1 + t0r0 + t1r0 + t1r1)
    p11 = (t1r1) / N
    p10 = (t1r0) / N
    p01 = (t0r1) / N
    p00 = (t0r0) / N
    pr1 = p11+p01
    pr0 = 1-pr1
    pt1 = p11+p10
    pt0 = 1-pt1
    py = (pr1-1+s0)/(s1-1+s0)
    return (p11, p10, p01, p00, pr1, pr0, pt1, pt0, py)

def probabilities(s1,s0, params):
    """
    Auxiliary function. Returns needed marginal probabilities and prevalence for given s1, s0 and joint probabilities
    as a tuple.
    :param s1: Sensitivity of the reference test. Float scalar
    :param s0: Specificity of the reference test. Float scalar
    :param params: Joint probabilities (p11, p10, p01, p00). Tuple of floats
    :return: Tuple of joint probabilities, marginal and prevalence.
    """
    (p11, p10, p01, p00) = params
    # Marginals
    pr1 = p11+p01
    pt1 = p11+p10

    # Prevalence
    py = (pr1-1+s0)/(s1-1+s0)
    return (p11, p10, p01, p00, pr1, py)

def intersection(a, b):
    """
    Auxiliary function. Returns intersection of intervals a and b.

    :param a: Interval. List
    :param b: Interval. List
    :return: List
    """
    return np.array([max(a[0], b[0]), min(a[1], b[1])])

def thibodeau(s1,s0, t1r1, t1r0, t0r1, t0r0, projection = False, grid_steps = 1000):
    """
    Yields the estimate of the bounds according to Thibodeau (1981) when s1, s0 are known.

    :param s1: Sensitivity of the reference test. Float scalar
    :param s0: Specificity of the reference test. Float scalar
    :param t1r1: Number of positives on both tests. Integer scalar
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
    :param t0r0: Number of negatives on both tests. Integer scalar
    :param projection: If True, returns projection boudns, otherwise returns the joint identified set. Boolean scalar
    :param grid_steps: Number of points in the grid in EACH dimension for theta1 and theta 0.
                       Total number grid_steps^2. Default 1000. Integer scalar

    :return: List of bounds. List of lists
    """
    (p11, p10, p01, p00, pr1, pr0, pt1, pt0, py) = estimates(s1, s0, t1r1, t1r0, t0r1, t0r0)
    theta1_bounds = [p11/py-(1-s0)*(1-py)/py,(p11*s0-p10*(1-s0))/(s0*pr1-(1-s0)*pr0)]
    theta1_bounds = intersection(theta1_bounds,[0,1]) # Intersect with [0,1] if the bounds take value outside of it
    theta0_bounds = [p00/(1-py)-(1-s1)*py/(1-py),(p00*s1-p01*(1-s1))/(s1*pr0-(1-s1)*pr1)]
    theta0_bounds = intersection(theta0_bounds,[0,1]) # Intersect with [0,1] if the bounds take value outside of it

    if projection:
        return [theta1_bounds, theta0_bounds]
    else:
        theta_grid = np.linspace(theta1_bounds[0], theta1_bounds[1], grid_steps)  # Fine grid over the bounds
        joint_region = []
        return [theta1_bounds, theta0_bounds]


def emerson(s1, s0, t1r1,t1r0,t0r1,t0r0):
    """
    Yields the estimate of the joint identified set according to Emerson et al. (2018) when s1, s0 are known.

    :param s1: Sensitivity of the reference test. Float scalar
    :param s0: Specificity of the reference test. Float scalar
    :param t1r1: Number of positives on both tests. Integer scalar
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
    :param t0r0: Number of negatives on both tests. Integer scalar

    :return: List of endpoints of the identified sets for sensitivity and specificity
    """

    (p11, p10, p01, p00, pr1, pr0, pt1, pt0, py) = estimates(s1, s0, t1r1,t1r0,t0r1,t0r0)

    theta1_bounds = [(max(0,p11-pr1+s1*py)+max(0,py-s1*py-p00))/py
                                      ,(min(p10,(py-s1*py))+min(p11,s1*py))/py] # Bounds on sensitivity
    theta0_bounds = [(max(0,p00-pr0+s0*(1-py))+max(0,(1-py)-s0*(1-py)-p11))/(1-py)
                                      ,(min(p01,(1-py)-s0*(1-py))+min(p00,s0*(1-py)))/(1-py)] # Bounds on sensitivity

    return [theta1_bounds, theta0_bounds]

def moments(theta1, theta0, s1, s0, t1r1, t1r0, t0r1, t0r0, wrongly_agree_0=False, wrongly_agree_1=False):
    """
    Returns calculated moment functions for given parameters theta1, theta0, s1, s0, at each observation.

    :param theta1: Sensitivity of the index test. Float scalar
    :param theta0: Specificity of the index test. Float scalar
    :param s1: Sensitivity of the reference test. Float scalar
    :param s0: Specificity of the reference test. Float scalar
    :param t1r1: Number of positives on both tests. Integer scalar
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
    :param t0r0: Number of negatives on both tests. Integer scalar
    :param wrongly_agree_0: Tests have a tendency to wrongly agree for y=0. Default False. Boolean scalar
    :param wrongly_agree_1: Tests have a tendency to wrongly agree for y=1. Default False. Boolean scalar
    :return: Moment functions for each observation. List of lists
    """

    N = (t0r1 + t0r0 + t1r0 + t1r1)
    ti1 = t1r1 + t1r0
    ti = np.append(np.ones(ti1), np.zeros(N - ti1)).astype(int)  # Create a vector of data for ti
    t = np.array([1] * t1r1 + [0] * t1r0 + [1] * t0r1 + [0] * t0r0)  # Create a conformal t
    y = (t - 1 + s0) / (s1 - 1 + s0)  # Create a y variable as a shorthand

    # Calculate the moment inequality functions
    m1 = ti * t - t + s1 * y - theta1 * y
    m2 = y * (1 - s1) - (1 - ti) * (1 - t) - theta1 * y
    m3 = y + ti - 1 - theta1 * y

    if wrongly_agree_1 == True and wrongly_agree_0 == False:
        m4 = theta1 * y - ti
        m5 = theta1 * y - ti * (1 - t) - s1 * y
        m6 = theta1 * y - y * (1 - s1) / 2 - ti * t
        meq = theta0 * (1 - y) - theta1 * y - (
                1 - y) + ti

        return m1, m2, m3, m4, m5, m6, meq

    elif wrongly_agree_0 == True and wrongly_agree_1 == False:
        m1 = (-theta0 + s0) * (1 - y) + ti * (t - 1)
        m2 = (-theta0 + 1 - s0) * (1 - y) - ti * t
        m3 = (-theta0 + 1) * (1 - y) - ti
        m4 = theta0 * (1 - y) + (ti - 1)
        m5 = (theta0 - s0) * (1 - y) - t * (1 - ti)
        m6 = (theta0 + (-1 + s0) / 2) * (1 - y) - (1 - ti) * (1 - t)
        meq = theta0 * (1 - y) - theta1 * y - (
                1 - y) + ti

        return m1, m2, m3, m4, m5, m6, meq

    elif wrongly_agree_1 == True and wrongly_agree_0 == True:
        m4 = theta1 * y - ti + 1 / 2 * (t - s1 * y)
        m5 = theta1 * y - ti * (1 - t) - s1 * y
        m6 = (theta1 + (-1 + s1) / 2) * y - t * ti + 1 / 2 * (t - s1 * y)
        meq = theta0 * (1 - y) - theta1 * y - (1 - y) + ti

        return m1, m2, m3, m4, m5, m6, meq

    else:
        m4 = theta1 * y - ti
        m5 = theta1 * y - ti * (1 - t) - s1 * y
        m6 = theta1 * y - y * (1 - s1) - ti * t
        meq = theta0 * (1 - y) - theta1 * y - (
                1 - y) + ti

        return m1, m2, m3, m4, m5, m6, meq



def bootstrap(t1r1, t1r0, t0r1, t0r0, boot_samples=500, seed = False):
    """
   Auxiliary function for making non-parametric boostrap draws in Romano, Shaikh and Wolf (2014).
   Returns the list of tuples of bootstrap samples.

   :param t1r1: Number of positives on both tests. Integer scalar
   :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
   :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
   :param t0r0: Number of negatives on both tests. Integer scalar
   :param boot_samples: Number of bootstrap samples. Default 500. Integer scalar
   :param seed: Seed number. Default False. Boolean or integer scalar
   :return: List of tuples
   """

    if seed!=False:
        np.random.seed(seed)

    N = (t0r1 + t0r0 + t1r0 + t1r1)
    ti1 = t1r1 + t1r0
    ti = np.append(np.ones(ti1), np.zeros(N - ti1)).astype(int)  # Create a vector of data for ti
    t = np.array([1] * t1r1 + [0] * t1r0 + [1] * t0r1 + [0] * t0r0)  # Create a conformal t
    sample = np.transpose([ti, t])
    idx = np.random.randint(0, len(sample), size=(boot_samples, len(sample)))  # Create indices for boostrap
    resampled_data = sample[idx]  # Use indices to get boostrap samples
    bootstrap_samples = []
    for i in resampled_data:
        bootstrap_samples.append((np.sum(i[:, 0] * i[:, 1]), np.sum(i[:, 0] * (1 - i[:, 1])),
                                  np.sum((1 - i[:, 0]) * i[:, 1]), np.sum((1 - i[:, 0]) * (1 - i[:, 1]))))

    return bootstrap_samples


def rsw(theta1, theta0, s1, s0, t1r1, t1r0, t0r1, t0r0, alpha=0.05, method='2', boot_samples=500,tol=10 ** (-15),
        wrongly_agree_0=False, wrongly_agree_1=False, seed=False):
    """
    Returns the Romano, Shaikh and Wolf (2014) (RSW henceforth) test result for given parameters theta1,
    theta0, s1 and s0.

    :param theta1: Sensitivity of the index test. Float scalar
    :param theta0: Specificity of the index test. Float scalar
    :param s1: Sensitivity of the reference test. Float scalar
    :param s0: Specificity of the reference test. Float scalar
    :param t1r1: Number of positives on both tests. Integer scalar
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
    :param t0r0: Number of negatives on both tests. Integer scalar
    :param alpha: Significance level. Default 0.05. Float scalar
    :param method: Test statistic type. Default "2". Possible values: "1", "2"
    :param boot_samples: Number of bootstrap samples for critical value calculations. Default 500. Integer scalar
    :param tol: Tolerance for censoring to zero to avoid float calculation errors. Default: 10**(-15). Float scalar
    :param wrongly_agree_0: Tests have a tendency to wrongly agree for y=0. Default False. Boolean scalar
    :param wrongly_agree_1: Tests have a tendency to wrongly agree for y=1. Default False. Boolean scalar
    :param seed: Seed number. Default False. Boolean or integer scalar
    :return: True (reject) or False (do not reject)
    """

    ## First check the parameter space limitation
    if wrongly_agree_1 == True and theta1>(1+s1)/2:
        return False

    if wrongly_agree_0 == True and theta0>(1+s0)/2:
        return False
    #
    ## If in parameter space, proceed to test
    beta = alpha/10 # Significance level of the first step. Default alpha/10 following RSW (2014)
    data = np.stack(moments(theta1, theta0, s1, s0, t1r1, t1r0, t0r1, t0r0,
                            wrongly_agree_0=wrongly_agree_0, wrongly_agree_1=wrongly_agree_1),
                            axis=1)  # stack all moment function values

    # Calculate the full sample statistics
    N = (t0r1 + t0r0 + t1r0 + t1r1)
    mbars = np.sum(data, axis=0) / N
    mbars[abs(mbars) <= tol] = 0  # Set to 0 all those that are below precision tolerance
    sigmashats = np.sqrt(np.sum((data - mbars) ** 2, axis=0) / N)

    # Create set of bootstrap samples
    bootstrap_samples = bootstrap(t1r1, t1r0, t0r1, t0r0, boot_samples, seed = seed)

    # Calculate boostrap statistics for c1
    boot_dataset = []
    boot_stats = []
    mbar_boots = []
    sigmashat_boots = []

    for boot in bootstrap_samples:
        boot_data = np.stack(moments(theta1, theta0, s1, s0, *boot,
                                     wrongly_agree_0=wrongly_agree_0, wrongly_agree_1=wrongly_agree_1), axis=1) # stack all moment function values

        mbar_boot = np.sum(boot_data, axis=0) / N
        if method == '1': # RSW1 test from Bai, Shaikh, Santos (2019)
            sigmashat_boot = np.sqrt(np.sum((boot_data - mbar_boot) ** 2, axis=0) / N)
            boot_stat = max(np.sqrt(N)*(mbars[:-1] - mbar_boot[:-1])/sigmashat_boot[:-1])# Omit data corresponding to moment equality
            sigmashat_boots.append(sigmashat_boot)
        elif method == '2': # RSW2 test from Bai, Shaikh, Santos (2019)
            boot_stat = max(np.sqrt(N) * (mbars [:-1]- mbar_boot[:-1]) / sigmashats[:-1]) # Omit data corresponding to moment equality
        boot_dataset.append(boot_data)
        mbar_boots.append(mbar_boot)
        boot_stats.append(boot_stat)

    c1 = np.quantile(boot_stats,1-beta) # First critical value
    uhat = np.min([mbars[:-1]+sigmashats[:-1]*c1/np.sqrt(N), np.zeros(len(mbars[:-1]))], axis=0) # Following notation from Bai, Shaikh, Santos (2019)


    # Calculate boostrap statistics for c2
    boot_stats = []
    for i in range(boot_samples):
        mbar_boot = mbar_boots[i]
        if method == '1':
            sigmashat_boot = sigmashat_boots[i]
            boot_stat = np.max(max(np.sqrt(N)*(-mbars[:-1] + mbar_boot[:-1] + uhat)/sigmashat_boot[:-1]), 0)
        elif method == '2':
            stat_inequalities = max(np.sqrt(N)*(-mbars[:-1] + mbar_boot[:-1] + uhat)/sigmashats[:-1]) # Correspond to inequalities
            boot_stat = max(np.abs(np.sqrt(N)*(-mbars[-1] + mbar_boot[-1])/sigmashats[-1]),stat_inequalities,0) # Corresponds to equality - See Bai, Shaikh, Santos (2021)
        boot_stats.append(boot_stat)

    c2 = np.quantile(boot_stats,1-alpha+beta) # Second critical value

    # Calculate the test statistic and do the test
    tn = max(np.abs(np.sqrt(N)*(mbars[-1])/sigmashats[-1]),max(np.sqrt(N)*(mbars[:-1])/sigmashats[:-1]),0)
    keep = tn <= c2 # perform the test

    return keep


def form_conf_sets(s1, s0, t1r1, t1r0, t0r1, t0r0, alpha = 0.05, grid_steps=1000, method = '2',
                   boot_samples = 500, wrongly_agree_0=False, wrongly_agree_1=False, parallel = True, num_threads = -1,
                   seed = False):
    """
    Function that returns the confidence set using RSW for a known s1 and s0.

    :param s1: Sensitivity of the reference test. Float scalar
    :param s0: Specificity of the reference test. Float scalar
    :param t1r1: Number of positives on both tests. Integer scalar
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
    :param t0r0: Number of negatives on both tests. Integer scalar
    :param alpha: Significance level. Float scalar
    :param grid_steps: Number of points in the grid in EACH dimension for theta1 and theta 0.
                       Total number grid_steps^2. Default 1000. Integer scalar
    :param method: Test statistic type. Default "2". Possible values: "1", "2"
    :param boot_samples: Number of bootstrap samples for critical value calculations. Default 500. Integer scala
    :param wrongly_agree_0: Tests have a tendency to wrongly agree for y=0. Default False. Boolean scalar
    :param wrongly_agree_1: Tests have a tendency to wrongly agree for y=1. Default False. Boolean scalarr
    :param parallel: Parallelization indicator. Boolean scalar
    :param num_threads: Number of threads used in calculation. Default -1 - all available. Integer scalar
    :param seed: Seed number. Default False. Boolean or integer scalar
    :return: List of points in confidence set
    """
    conf_set = []
    theta_grid = np.array(np.meshgrid(np.linspace(0,1,grid_steps), np.linspace(0,1,grid_steps),indexing='ij')).reshape(2,-1).T
    if parallel:
        sol = Parallel(n_jobs=num_threads)(delayed(rsw)(thetas[0], thetas[1], s1, s0, t1r1, t1r0, t0r1, t0r0,
                                                        alpha=alpha, method = method, boot_samples=boot_samples,
                                                        wrongly_agree_0=wrongly_agree_0, wrongly_agree_1=wrongly_agree_1,
                                                        seed = seed)
                                            for thetas in theta_grid)
        conf_set = theta_grid[sol]
    else:
        for thetas in theta_grid:
            if rsw(thetas[0], thetas[1], s1, s0, t1r1, t1r0, t0r1, t0r0,
                   alpha=alpha, method = method, boot_samples=boot_samples,
                   wrongly_agree_0=wrongly_agree_0, wrongly_agree_1=wrongly_agree_1):
                conf_set.append(thetas)

    conf_set = np.array(conf_set)
    # try:
    #     return np.array(sens_cf)[[1,-1]], np.array(spec_cf)[[1,-1]]
    # except:
    #     return np.array(sens_cf), np.array(spec_cf)
    return conf_set

def form_conf_sets_rsw_unknown(s1, s0, t1r1, t1r0, t0r1, t0r0, alpha = 0.05, grid_steps=1000, gridsteps_s=1000,
                               method = '2',boot_samples = 500, wrongly_agree_0=False,
                               wrongly_agree_1=False, parallel = True, num_threads = -1, seed=False):
    """
    Function that returns the confidence set using RSW when s1 and s0 in some set S. Assumed S is rectangular.

    :param s1: Bounds on sensitivity of the reference test. List of floats
    :param s0: Bounds on specificity of the reference test. List of floats
    :param t1r1: Number of positives on both tests. Integer scalar
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
    :param t0r0: Number of negatives on both tests. Integer scalar
    :param alpha: Significance level. Float scalar
    :param grid_steps: Number of points in the grid in EACH dimension for theta1 and theta 0.
                       Total number grid_steps^2. Default 1000. Integer scalar
    :param grid_steps_s: Number of points in the grid in EACH dimension for S.
                         Total number grid_steps_s^2 if both s1 and s0 not known exactly, gridsteps_s otherwise.
                         Default 1000. Integer scalar
    :param method: Test statistic type. Default "2". Possible values: "1", "2"
    :param boot_samples: Number of bootstrap samples for critical value calculations. Default 500. Integer scala
    :param wrongly_agree_0: Tests have a tendency to wrongly agree for y=0. Default False. Boolean scalar
    :param wrongly_agree_1: Tests have a tendency to wrongly agree for y=1. Default False. Boolean scalar
    :param parallel: Parallelization indicator. Boolean scalar
    :param num_threads: Number of threads used in calculation. Default -1 - all available. Integer scalar
    :param seed: Seed number. Default False. Boolean or integer scalar
    :return: List of points in confidence set
    """
    conf_set = []
    # Makes 4d gridspace with grid_steps**2+gridsteps_s**2 points
    theta_grid = np.array(np.meshgrid(np.linspace(0,1,grid_steps), np.linspace(0,1,grid_steps),
                                      np.linspace(s1[0],s1[1],gridsteps_s), np.linspace(s0[0],s0[1],gridsteps_s)
                                      ,indexing='ij')).reshape(4,-1).T
    theta_grid = np.unique(theta_grid,axis=0) # Drop duplicates in case s1 or s0 has singleton values

    if parallel:
        sol = Parallel(n_jobs=num_threads)(delayed(rsw)(thetas[0], thetas[1], thetas[2], thetas[3], t1r1, t1r0, t0r1,
                                                        t0r0, alpha=alpha, method = method, boot_samples=boot_samples,
                                                        wrongly_agree_0=wrongly_agree_0, wrongly_agree_1=wrongly_agree_1)
                                            for thetas in theta_grid)
        conf_set = theta_grid[sol]
    else:
        for thetas in theta_grid:
            if rsw(thetas[0], thetas[1], s1, s0, t1r1, t1r0, t0r1, t0r0,
                                                        alpha=alpha, method = method, boot_samples=boot_samples):
                conf_set.append(thetas)

    conf_set = np.array(conf_set)
    # try:
    #     return np.array(sens_cf)[[1,-1]], np.array(spec_cf)[[1,-1]]
    # except:
    #     return np.array(sens_cf), np.array(spec_cf)
    return conf_set


def bounds_estimate(s1, s0, t1r1,t1r0,t0r1,t0r0, wrongly_agree_0=False, wrongly_agree_1=False, grid_steps = 1000):
    """
    Yields the estimate of the joint identified set when s1, s0 are known.

    :param s1: Sensitivity of the reference test. Float scalar
    :param s0: Specificity of the reference test. Float scalar
    :param t1r1: Number of positives on both tests. Integer scalar
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
    :param t0r0: Number of negatives on both tests. Integer scalar
    :param wrongly_agree_0: Tests have a tendency to wrongly agree for y=0. Default False. Boolean scalar
    :param wrongly_agree_1: Tests have a tendency to wrongly agree for y=1. Default False. Boolean scalar
    :param grid_steps: Number of points in the grid in EACH dimension for theta1 and theta 0.
                       Total number grid_steps^2. Default 1000. Integer scalar
    :return: List of points in estimated identified set
    """

    (p11, p10, p01, p00, pr1, pr0, pt1, pt0, py) = estimates(s1, s0, t1r1,t1r0,t0r1,t0r0)
    b = wrongly_agree_0
    a = wrongly_agree_1

    theta1_bounds = [(max(0,p11-pr1+s1*py)+max(0,py-s1*py-p00))/py
                                      ,(min(p10,(py-s1*py)/(1+a))+min(p11-(pr1-s1*py)/(1+b)*b,s1*py))/py] # Bounds on sensitivity
    theta_grid = np.linspace(theta1_bounds[0], theta1_bounds[1], grid_steps) # Fine grid over the bounds
    joint_region = []
    for theta1 in theta_grid:
        theta0 = (theta1*py+1-py-pt1)/(1-py)
        joint_region.append([theta1,theta0])
    joint_region = np.array(joint_region)

    # joint_region = np.round([joint_region[:,0],joint_region[:,1]], int(np.log10(grid_steps)))

    return joint_region




def bounds_estimate_unknown(s1, s0, t1r1, t1r0, t0r1, t0r0, wrongly_agree_0=False, wrongly_agree_1=False,
                           grid_steps=1000, gridsteps_s=1000):
    """
    Yields the estimate of the joint identified set when s1, s0 are in some set S. Assumed S is rectangular.

    :param s1: Bounds on sensitivity of the reference test. List of floats
    :param s0: Bounds on specificity of the reference test. List of floats
    :param t1r1: Number of positives on both tests. Integer scalar
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
    :param t0r0: Number of negatives on both tests. Integer scalar
    :param wrongly_agree_0: Tests have a tendency to wrongly agree for y=0. Default False. Boolean scalar
    :param wrongly_agree_1: Tests have a tendency to wrongly agree for y=1. Default False. Boolean scalar
    :param grid_steps: Number of points in the grid in EACH dimension for theta.
                       Total number grid_steps^2. Default 1000. Integer scalar
    :param grid_steps_s: Number of points in the grid in EACH dimension for S.
                         Total number grid_steps^2 if both s1 and s0 not known exactly, gridsteps_s otherwise.
                         Default 1000. Integer scalar
    :return: List of points in estimated identified set
    """

    b = wrongly_agree_0
    a = wrongly_agree_1

    SeSp_grid = np.array(np.meshgrid(np.linspace(s1[0],s1[1],gridsteps_s), np.linspace(s0[0],s0[1],gridsteps_s)
                         ,indexing='ij')).reshape(2,-1).T
    SeSp_grid = np.unique(SeSp_grid,axis=0) # Drop duplicates in case s1 or s0 has singleton values

    joint_region = []
    temp_region = []
    for s1,s0 in SeSp_grid:
        (p11, p10, p01, p00, pr1, pr0, pt1, pt0, py) = estimates(s1, s0, t1r1, t1r0, t0r1, t0r0)

        theta1_bounds = [(max(0, p11 - pr1 + s1 * py) + max(0, py - s1 * py - p00)) / py
                        ,(min(p10, (py - s1 * py) / (1 + a)) + min(p11 - (pr1 - s1 * py) / (1 + b) * b,s1 * py)) / py]  # Bounds on sensitivity

        theta_grid = np.linspace(theta1_bounds[0], theta1_bounds[1], grid_steps)  # Fine grid over the bounds
        temp_region = []
        for theta1 in theta_grid:
            theta0 = (theta1 * py + 1 - py - pt1) / (1 - py)
            temp_region.append([theta1, theta0])

        joint_region = joint_region+temp_region
    joint_region = np.array(joint_region)
    joint_region = np.unique(joint_region,axis=0)
    # joint_region = np.round([joint_region[:,0],joint_region[:,1]], int(np.log10(grid_steps)))

    return joint_region

def graphing(conf_set, estimated_set, t1r1,t1r0,t0r1,t0r0, alpha=0.05, filename = "conf_set.png", transparent=False,
             unknown = False, font_size = 18, include_apparent = True):
    """
    Drawing graphs for the confidence sets and estimates.

    :param conf_set: Array of points in the confidence set. Float array
    :param estimated_set: Array of points in the estimated identified set. Float array
    :param t1r1: Number of positives on both tests. Integer scalar
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
    :param t0r0: Number of negatives on both tests. Integer scalar
    :param alpha: Significance level. Default 0.05. Float scalar.
    :param filename: Name of the output graph. Default "conf_set". String
    :param transparent: If True, makes a transparent graph for slide embedding. Default False. Boolean scalar
    :param unknown: Set to True if s1 or s0 are not known exactly. Default False. Boolean scalar
    :param font_size: Font size for label ticks and labels. Default 18. Integer scalar
    :param include_apparent: If True, draws apparent estimates and corresponding projection confidence intervals.
                             Default True. Boolean scalar
    :return: Plots and saves graphs so the Graphs folder.
    """
    # Define convex hull
    hull = ConvexHull(conf_set)
    hull_indices = np.unique(hull.simplices.flat)

    # Apparent estmates
    point = [t1r1/(t1r1+t0r1),t0r0/(t0r0+t1r0)]

    ## Corrected alpha CP - yields at least 1-alpha joint coverage

    alpha = 1-np.sqrt(1-alpha) # Correct for joint coverage of projection confidence interval
    CP = np.asarray([stats.proportion_confint(t1r1, (t1r1 + t0r1), alpha=alpha, method='beta'),
          stats.proportion_confint(t0r0, (t0r0 + t1r0), alpha=alpha, method='beta')])
    normal = np.asarray([stats.proportion_confint(t1r1, (t1r1 + t0r1), alpha=alpha, method='normal'),
            stats.proportion_confint(t0r0, (t0r0 + t1r0), alpha=alpha, method='normal')])

    box_cp = np.stack([CP[0][[0,1,1,0]],CP[1][[0,0,1,1]]])
    box_normal = np.stack([normal[0][[0,1,1,0]],normal[1][[0,0,1,1]]])

    ### Plot

    fig, ax = plt.subplots();
    fig.set_figheight(10)  # Set figure size when using a single subplot
    fig.set_figwidth(15)

    # Automatic limits
    if include_apparent:
        ylim = (min(min(conf_set[:,1])-0.005, min(box_cp[1])-0.005, min(box_normal[1])-0.005),
                 max(max(conf_set[:,1])+0.003,max(box_cp[1])+0.003,max(box_normal[1])+0.003))
        xlim = (min(min(conf_set[:,0])-0.01,min(box_cp[0])-0.01,min(box_normal[0])-0.01),
                 max(max(conf_set[:,0])+0.001, max(box_cp[0])+0.001, max(box_normal[0])+0.001))
    else:
        ylim = (min(conf_set[:,1])-0.005,max(conf_set[:,1])+0.003)
        xlim = (min(conf_set[:,0])-0.01,max(conf_set[:,0])+0.001)

    ax.set_ylim(*ylim)  # Set graph range for y
    ax.set_xlim(*xlim)  # Set graph range for x


    ax.grid(b=None, which='major', axis='both', linestyle='--', linewidth=0.5)  # Make a grid
    ax.set_axisbelow(True)  # Make grid be drawn behind the plot
    ax.fill(conf_set[hull.vertices, 0], conf_set[hull.vertices, 1], fc=(0, 1, 0, 0.2), ec=(0, 0, 0, 1),
            linewidth=2, label='95% CS for $(\\theta_1,\\theta_0)$')
    if include_apparent:
        ax.fill(box_cp[0], box_cp[1], alpha=0.2, facecolor='red', edgecolor='red',
                 linewidth=2, hatch= '\\', label='95% CS for Apparent $(\\theta_1,\\theta_0)$')
        ax.plot(point[0], point[1], 'ro', label='Estimated Apparent $(\\theta_1,\\theta_0)$')
        # ax.fill(box_normal[0], box_normal[1], alpha=0.2, facecolor='purple', edgecolor='purple',
        #         linewidth=2, hatch='/', label='90.25% Naive Normal Confidence Set')
    if unknown == False:
        ax.plot(estimated_set[:,0],estimated_set[:,1], 'red', linewidth=2, label='Estimated Identified Set for $(\\theta_1,\\theta_0)$')
    elif unknown == True:
        hull_point = ConvexHull(estimated_set)
        ax.fill(estimated_set[hull_point.vertices, 0], estimated_set[hull_point.vertices, 1],
                linewidth=2, alpha=1, facecolor='red', edgecolor='red', label='Estimated Identified Set for $(\\theta_1,\\theta_0)$')
    ax.set_xlabel('$\\theta_1$', labelpad=10, fontsize=font_size, family='Times New Roman')
    ax.set_ylabel('$\\theta_0$', labelpad=10, fontsize=font_size, family='Times New Roman', rotation=0)
    ax.xaxis.set_label_coords(1, -0.02) # move x labels to the right
    ax.yaxis.set_label_coords(-0.02, 1) # move y labels to the top
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    # Order legend
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels,prop=font, loc=4)
    # ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=5,prop=font)
    # ax.legend()

    plt.savefig(filename, transparent=transparent,bbox_inches='tight')
    plt.close()

def calculate(s1, s0, t1r1,t1r0,t0r1,t0r0, wrongly_agree_0=False, wrongly_agree_1=True, grid_steps_estimate = 1000,
            grid_steps_conf = 316, gridsteps_s = 10, method = '2', boot_samples = 500, alpha=0.05,
            parallel = True, num_threads = -1, filename='graph', include_apparent = True, seed = False):
    """
    Omnibus function for calculation. Plots and saves graphs so the Graphs folder.

    :param s1: Sensitivity of the reference test. Float scalar
    :param s0: Specificity of the reference test. Float scalar
    :param t1r1: Number of positives on both tests. Integer scalar
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
    :param t0r0: Number of negatives on both tests. Integer scalar
    :param wrongly_agree_0: Tests have a tendency to wrongly agree for y=0. Default False. Boolean scalar
    :param wrongly_agree_1: Tests have a tendency to wrongly agree for y=1. Default False. Boolean scalar
    :param grid_steps_estimate: Grid size for finding estimated identified set. Default 1000. Boolean scalar
    :param grid_steps_conf: Number of points in the grid in EACH dimension for theta.
                            Total number grid_steps_conf^2. Default 1000. Integer scalar
    :param grid_steps_s: Number of points in the grid in EACH dimension for S.
                         Total number grid_steps^2 if both s1 and s0 not known exactly, gridsteps_s otherwise.
                         Default 1000. Integer scalar
    :param method: Test statistic type. Default "2". Possible values: "1", "2"
    :param boot_samples: Number of bootstrap samples for critical value calculations. Default 500. Integer scala
    :param alpha: Significance level. Default 0.05. Float scalar.
    :param wrongly_agree_1: Tests have a tendency to wrongly agree for y=1. Default False. Boolean scalar
    :param parallel: Parallelization indicator. Boolean scalar
    :param num_threads: Number of threads used in calculation. Default -1 - all available. Integer scalar
    :param filename: Name of the output graph. Default "conf_set". String
    :param include_apparent: If True, draws apparent estimates and corresponding projection confidence intervals.
                             Default True. Boolean scalar
    :param seed: Seed number. Default False. Boolean or integer scalar
    :return: List of points in the estimated identified and confidence sets
    """

    ## Entry data validation

    if type(s1) == list or type(s0) == list: # If s1 or s0 entered as list, taken as bounds on it

        if type(s1) != list:
            s1 = [s1,s1] # Reform into list if s1 given directly
        if type(s0) != list:
            s0 = [s0,s0]

        estimated_set = bounds_estimate_unknown(s1, s0, t1r1, t1r0, t0r1, t0r0, wrongly_agree_0 = wrongly_agree_0,
                                                  wrongly_agree_1=wrongly_agree_1, grid_steps=grid_steps_estimate,
                                                  gridsteps_s=gridsteps_s)

        conf_set = form_conf_sets_rsw_unknown(s1, s0, t1r1, t1r0, t0r1, t0r0, grid_steps=grid_steps_conf,
                                              gridsteps_s=gridsteps_s, method=method, alpha=alpha,
                                              boot_samples=boot_samples, parallel=parallel, num_threads=num_threads,
                                              wrongly_agree_0=wrongly_agree_0, wrongly_agree_1=wrongly_agree_1,
                                              seed=seed)

        print("Estimates of apparent measures are :" + str([round(t1r1 / (t1r1 + t0r1), 3),
                                                            round(t0r0 / (t1r0 + t0r0), 3)]))
        print("Estimated projection bounds for (theta1, theta0) are: " + str([[min(estimated_set[:, 0]),
                                                                               max(estimated_set[:, 0])],
                                                                              [min(estimated_set[:, 1]),
                                                                               max(estimated_set[:, 1])]]))
        print("Projected confidence sets for (theta1, theta0) are: " + str([[min(conf_set[:, 0]), max(conf_set[:, 0])],
                                                                      [min(conf_set[:, 1]), max(conf_set[:, 1])]]))
        graphing(conf_set[:, 0:2], estimated_set, t1r1,t1r0,t0r1,t0r0, alpha=0.05,
                 filename="Graphs/" + filename + ".png", unknown=True, include_apparent = include_apparent)

        print("Graphing done.")

    else: # If both s1 and s0 taken as floats, assumed to be known
        estimated_set = bounds_estimate(s1, s0, t1r1, t1r0, t0r1, t0r0, wrongly_agree_0 = wrongly_agree_0,
                                          wrongly_agree_1=wrongly_agree_1, grid_steps=grid_steps_estimate)

        conf_set = form_conf_sets(s1, s0, t1r1, t1r0, t0r1, t0r0, grid_steps=grid_steps_conf, method=method, alpha=alpha,
                                  boot_samples=boot_samples, parallel=parallel, num_threads=num_threads,
                                   wrongly_agree_0=wrongly_agree_0, wrongly_agree_1=wrongly_agree_1, seed=seed)
        print("Estimates of apparent measures are :" + str([round(t1r1 / (t1r1 + t0r1), 3),
                                                            round(t0r0 / (t1r0 + t0r0), 3)]))
        print("Estimated projection bounds for (theta1, theta0) are: " + str([[min(estimated_set[:, 0]),
                                                                               max(estimated_set[:, 0])],
                                                                              [min(estimated_set[:, 1]),
                                                                               max(estimated_set[:, 1])]]))

        print("Projected confidence sets for (theta1, theta0) are: " + str([[min(conf_set[:, 0]), max(conf_set[:, 0])],
                                                                      [min(conf_set[:, 1]), max(conf_set[:, 1])]]))

        graphing(conf_set, estimated_set, t1r1,t1r0,t0r1,t0r0, alpha=0.05, filename="Graphs/" + filename + ".png",
                 include_apparent = include_apparent)
        print("Graphing done.")

        return estimated_set, conf_set


def compare(s1, s0, t1r1,t1r0,t0r1,t0r0, wrongly_agree_0=False, wrongly_agree_1=True, grid_steps_estimate = 1000,
            filename='graph', include_apparent = True, font_size=18, transparent=True):
    """
    Omnibus function for comparison of estimates with different methods. Plots and saves graphs so the Graphs folder.

    :param s1: Sensitivity of the reference test. Float scalar
    :param s0: Specificity of the reference test. Float scalar
    :param t1r1: Number of positives on both tests. Integer scalar
    :param t1r0: Number of subjects who were positive on the index, but negative on the reference test. Integer scalar
    :param t0r1: Number of subjects who were negative on the index, but positive on the reference test. Integer scalar
    :param t0r0: Number of negatives on both tests. Integer scalar
    :param wrongly_agree_0: Tests have a tendency to wrongly agree for y=0. Default False. Boolean scalar
    :param wrongly_agree_1: Tests have a tendency to wrongly agree for y=1. Default False. Boolean scalar
    :param grid_steps_estimate: Grid size for finding estimated identified set. Default 1000. Boolean scalar
    :param filename: Name of the output graph. Default "conf_set". String
    :param include_apparent: If True, draws apparent estimates and corresponding projection confidence intervals.
                             Default True. Boolean scalar
    :param font_size: Font size for label ticks and labels. Default 18. Integer scalar
    :param transparent: If True, makes a transparent graph for slide embedding. Default False. Boolean scalar

    :return: List of points in the estimated identified sets
    """

    # Joint identified set estimate
    estimated_set = bounds_estimate(s1, s0, t1r1, t1r0, t0r1, t0r0, wrongly_agree_0 = wrongly_agree_0,
                                      wrongly_agree_1=wrongly_agree_1, grid_steps=grid_steps_estimate)

    # Emerson
    emerson_bounds = emerson(s1, s0, t1r1,t1r0,t0r1,t0r0)

    # Thibodeau
    thibodeau_bounds = thibodeau(s1,s0, t1r1, t1r0, t0r1, t0r0)

    print("Estimates of apparent measures are :" + str([round(t1r1 / (t1r1 + t0r1), 3),
                                                        round(t0r0 / (t1r0 + t0r0), 3)]))
    print("Estimated projection bounds for (theta1, theta0) are: " + str([[min(estimated_set[:, 0]),
                                                                           max(estimated_set[:, 0])],
                                                                          [min(estimated_set[:, 1]),
                                                                           max(estimated_set[:, 1])]]))
    print("Estimated Emerson bounds for (theta1, theta0) are: " + str(emerson_bounds))

    print("Estimated Thibodeau projection bounds for (theta1, theta0) are: " + str(thibodeau_bounds))


    ### Plot

    # Define convex hull
    thib = np.array([list(x) for x in product(thibodeau_bounds[0],thibodeau_bounds[1])])
    em = np.array([list(x) for x in product(emerson_bounds[0], emerson_bounds[1])])
    hull_thib = ConvexHull(thib)
    hull_em = ConvexHull(em)

    # Apparent estmates
    point = [t1r1/(t1r1+t0r1),t0r0/(t0r0+t1r0)]

    fig, ax = plt.subplots();
    fig.set_figheight(10)  # Set figure size when using a single subplot
    fig.set_figwidth(15)

    # Set limits
    ylim = (min(min(estimated_set[:, 1]),min(thib[:, 1]),min(em[:, 1]),point[1]) - 0.005,
            max(max(estimated_set[:, 1]),max(thib[:, 1]),max(em[:, 1]),point[1])+0.003)
    xlim = (min(min(estimated_set[:, 0]),min(thib[:, 0]),min(em[:, 0]),point[0]) - 0.001,
            max(max(estimated_set[:, 0]),max(thib[:, 0]),max(em[:, 0]),point[0])+0.001)

    ax.set_ylim(*ylim)  # Set graph range for y
    ax.set_xlim(*xlim)  # Set graph range for x

    ax.grid(b=None, which='major', axis='both', linestyle='--', linewidth=0.5)  # Make a grid
    ax.set_axisbelow(True)  # Make grid be drawn behind the plot

    # ax.fill(thib[hull_thib.vertices, 0], thib[hull_thib.vertices, 1], alpha=0.2, facecolor='red', edgecolor='red',
    #         linewidth=2, hatch='//', label='Estimated Thibodeau (1981) Bounds')
    # ax.fill(em[hull_em.vertices, 0], em[hull_em.vertices, 1], alpha=0.2, facecolor='green', edgecolor='green',
    #         linewidth=2, hatch='\\', label='Estimated Emerson et al. (2018) Bounds')

    ax.fill(thib[hull_thib.vertices, 0], thib[hull_thib.vertices, 1],alpha=0.5, facecolor='white', ec=(0, 0, 0, 1),
            linewidth=2, hatch='//', label='Estimated Thibodeau (1981) Bounds')
    ax.fill(em[hull_em.vertices, 0], em[hull_em.vertices, 1],alpha=0.3, facecolor='grey', ec=(0, 0, 0, 1),
            linewidth=2, hatch='..', label='Estimated Emerson et al. (2018) Bounds')

    if include_apparent:
        ax.plot(point[0], point[1], 'ro', label='Estimated Apparent $(\\theta_1,\\theta_0)$')

    ax.plot(estimated_set[:, 0], estimated_set[:, 1], 'red', linewidth=2,
            label='Estimated Identified Set for $(\\theta_1,\\theta_0)$')

    ax.set_xlabel('$\\theta_1$', labelpad=10, fontsize=font_size, family='Times New Roman')
    ax.set_ylabel('$\\theta_0$', labelpad=10, fontsize=font_size, family='Times New Roman', rotation=0)
    ax.xaxis.set_label_coords(1, -0.02)  # move x labels to the right
    ax.yaxis.set_label_coords(-0.02, 1)  # move y labels to the top
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    # Order legend
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, prop=font, loc=4)

    plt.savefig("Graphs/comparison_"+filename+".png", transparent=transparent, bbox_inches='tight')
    plt.close()

    ### Prevalence Analysis

    pt_grid = np.array([0.05, 0.1, 0.2])

    thibodeau_prevalence = np.array([(pt_grid+thibodeau_bounds[1][0]-1)/(thibodeau_bounds[0][1]+thibodeau_bounds[1][0]-1),
                            (pt_grid+thibodeau_bounds[1][1]-1)/(thibodeau_bounds[0][0]+thibodeau_bounds[1][1]-1)])
    emerson_prevalence = np.array([(pt_grid+emerson_bounds[1][0]-1)/(emerson_bounds[0][1]+emerson_bounds[1][0]-1),
                            (pt_grid+emerson_bounds[1][1]-1)/(emerson_bounds[0][0]+emerson_bounds[1][1]-1)])
    prevalence = np.array([
        np.minimum((pt_grid+min(estimated_set[:, 1])-1)/(min(estimated_set[:, 0])+min(estimated_set[:, 1])-1),
                   (pt_grid+max(estimated_set[:, 1])-1)/(max(estimated_set[:, 0])+max(estimated_set[:, 1])-1)),
        np.maximum((pt_grid + min(estimated_set[:, 1]) - 1) / (min(estimated_set[:, 0]) + min(estimated_set[:, 1]) - 1),
                   (pt_grid + max(estimated_set[:, 1]) - 1) / (
                               max(estimated_set[:, 0]) + max(estimated_set[:, 1]) - 1))])

    d = [list(map(str, np.round(l,3))) for l in [thibodeau_prevalence.T,emerson_prevalence.T,prevalence.T]]
    pd.DataFrame(d)


    ## Prevalence graphing

    min_pt = 1 - min(min(estimated_set[:, 1]),min(thib[:, 1]),min(em[:, 1])) # P(t=1) cannot be smaller than 1-\theta_0
    max_pt = min(min(estimated_set[:, 0]),min(thib[:, 0]),min(em[:, 0])) # P(t=1) cannot be larger than \theta_1

    pt_grid = np.linspace(min_pt, 0.3, 1000) # Grid of points for P(t=1)


    thibodeau_prevalence = np.array([(pt_grid+thibodeau_bounds[1][0]-1)/(thibodeau_bounds[0][1]+thibodeau_bounds[1][0]-1),
                            (pt_grid+thibodeau_bounds[1][1]-1)/(thibodeau_bounds[0][0]+thibodeau_bounds[1][1]-1)])
    emerson_prevalence = np.array([(pt_grid+emerson_bounds[1][0]-1)/(emerson_bounds[0][1]+emerson_bounds[1][0]-1),
                            (pt_grid+emerson_bounds[1][1]-1)/(emerson_bounds[0][0]+emerson_bounds[1][1]-1)])
    prevalence = np.array([
        np.minimum((pt_grid+min(estimated_set[:, 1])-1)/(min(estimated_set[:, 0])+min(estimated_set[:, 1])-1),
                   (pt_grid+max(estimated_set[:, 1])-1)/(max(estimated_set[:, 0])+max(estimated_set[:, 1])-1)),
        np.maximum((pt_grid + min(estimated_set[:, 1]) - 1) / (min(estimated_set[:, 0]) + min(estimated_set[:, 1]) - 1),
                   (pt_grid + max(estimated_set[:, 1]) - 1) / (
                               max(estimated_set[:, 0]) + max(estimated_set[:, 1]) - 1))])

    # Prevalence table

    # # Prevalence graphing
    fig, ax = plt.subplots();
    fig.set_figheight(10)  # Set figure size when using a single subplot
    fig.set_figwidth(15)

    # # Set limits
    # ylim = (min(prevalence[0][0],thibodeau_prevalence[0][0],emerson_prevalence[0][0]) - 0.005,
    #         min(prevalence[1][-1],thibodeau_prevalence[1][-1],emerson_prevalence[1][-1])+ 0.003)
    # xlim = (pt_grid[0] - 0.01,pt_grid[-1] + 0.001)
    #
    # ax.set_ylim(*ylim)  # Set graph range for y
    # ax.set_xlim(*xlim)  # Set graph range for x
    #
    #
    # ax.grid(b=None, which='major', axis='both', linestyle='--', linewidth=0.5)  # Make a grid
    # ax.set_axisbelow(True)  # Make grid be drawn behind the plot
    #
    # # Plot the data
    # # Prevalence bounds
    # ax.plot(pt_grid, prevalence[0], label='Prevalence Bounds', linestyle='-', color="black")
    # ax.plot(pt_grid, prevalence[1], linestyle='-', color="black")
    # ax.plot(pt_grid, emerson_prevalence[0], linestyle='--', color="black",
    # label='Emerson et al. (2018) Prevalence Bounds')
    # ax.plot(pt_grid, emerson_prevalence[1], linestyle='--', color="black")
    # ax.plot(pt_grid, thibodeau_prevalence[0], linestyle='-.', color="black",
    # label='Thibodeau (1981) Prevalence Bounds')
    # ax.plot(pt_grid, thibodeau_prevalence[1], linestyle='-.', color="black")

    # ax.set_xlabel('$P(t=1)$', labelpad=10, fontsize=font_size, family='Times New Roman')
    # ax.set_ylabel('$P(y=1)$', labelpad=10, fontsize=font_size, family='Times New Roman', rotation=0)
    # ax.xaxis.set_label_coords(1, -0.02)  # move x labels to the right
    # ax.yaxis.set_label_coords(-0.02, 1)  # move y labels to the top
    # ax.tick_params(axis="x", labelsize=font_size)
    # ax.tick_params(axis="y", labelsize=font_size)


    # Prevalence bounds width

    # Set limits
    ylim = (min(min(prevalence[1]-prevalence[0]),min(thibodeau_prevalence[1]-thibodeau_prevalence[0]),
                min(emerson_prevalence[1]-emerson_prevalence[0])),
            max(max(prevalence[1] - prevalence[0]), max(thibodeau_prevalence[1] - thibodeau_prevalence[0]),
                max(emerson_prevalence[1] - emerson_prevalence[0])))
    xlim = (pt_grid[0],pt_grid[-1])

    ax.set_ylim(*ylim)  # Set graph range for y
    ax.set_xlim(*xlim)  # Set graph range for x

    ax.plot(pt_grid, prevalence[1]-prevalence[0], label='Prevalence Bounds', linestyle='-', color="black")
    ax.plot(pt_grid, emerson_prevalence[1]-emerson_prevalence[0], linestyle='--', color="black",
            label='Emerson et al. (2018) Prevalence Bounds')
    ax.plot(pt_grid, thibodeau_prevalence[1]-thibodeau_prevalence[0], linestyle='-.', color="black",
            label='Thibodeau (1981) Prevalence Bounds')

    # # Length ratios
    # ax.plot(pt_grid, (prevalence[1]-prevalence[0])/(prevalence[1]-prevalence[0]), label='Prevalence Bounds', linestyle='-', color="black")
    # ax.plot(pt_grid, (emerson_prevalence[1]-emerson_prevalence[0])/(prevalence[1]-prevalence[0]), linestyle='--', color="black",
    #         label='Emerson et al. (2018) Prevalence Bounds')
    # ax.plot(pt_grid, (thibodeau_prevalence[1]-thibodeau_prevalence[0])/(prevalence[1]-prevalence[0]), linestyle='-.', color="black",
    #         label='Thibodeau (1981) Prevalence Bounds')


    ax.set_xlabel('$P(t=1)$', labelpad=10, fontsize=font_size, family='Times New Roman')
    ax.set_ylabel('Prevalence Bound Width', labelpad=10, fontsize=font_size, family='Times New Roman', rotation=0)
    ax.xaxis.set_label_coords(0.970, -0.04)  # move x labels to the right
    ax.yaxis.set_label_coords(0.05, 1)  # move y labels to the top
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    # Order legend
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels, prop=font, loc=0)
    # ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=5,prop=font)
    # ax.legend()

    plt.savefig("Graphs/comparison_"+filename+"_prevalence"+".png", transparent=transparent, bbox_inches='tight')
    plt.close()

    return estimated_set


