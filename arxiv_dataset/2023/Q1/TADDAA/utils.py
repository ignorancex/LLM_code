#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yu Wang
"""
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import t
from scipy.stats import chi2

import numdifftools as nd
import autograd.scipy.stats.norm as norm
import scipy.optimize as opt
from scipy.special import gamma
import scipy


def sample_size(delta, alpha = 0.05):
    """
    compute required sample size based on tolerance level delta
    
    parameters
    ----------
    delta: margin of error for the confidence interval
    alpha: significance level for the test
    """
    
    f = lambda x: t.ppf(1-alpha/2, df = x-1)/np.sqrt(x)-delta
    
    return fsolve(f, x0=10)

def sample_size_variance(delta, alpha = 0.05):
    """
    compute required sample size based on tolerance level delta
    
    parameters
    ----------
    delta: margin of error for the confidence interval
    alpha: significance level for the test
    """
    
    f = lambda x: np.log(chi2.ppf(1-alpha/2, df = x-1))-np.log(chi2.ppf(alpha/2, df = x-1))-2*delta
    
    return fsolve(f, x0=10)


def mean_diff_ci(x, mu, alpha=0.05, k=1):
    """
    One sample confidence interval for mean difference
    parameters
    ----------
    x: samples from the distribution to be checked
    mu: fixed mean in the hypothesis
    alpha: (1-alpha)*100% confidence interval is calculated 
    k: confidence interval is computed based on k-th coordinate
    
    Returns
    -------
    ci_lb: lower bound of confidence interval
    ci_ub: upper bound of confidence interval
    """
    x_bar = np.mean(x, axis = 0)
    s_x = np.cov(x.T)
    n = len(x)
    p = len(x.T)
    if p==1:
        ci_lb=(x_bar-mu)+np.sqrt(s_x/n)*t.ppf(alpha/2, df=n-1)
        ci_ub=(x_bar-mu)-np.sqrt(s_x/n)*t.ppf(alpha/2, df=n-1)
    else:
        ci_lb=(x_bar-mu)[k-1]+np.sqrt(s_x[k-1, k-1]/n)*t.ppf(alpha/2, df=n-1)
        ci_ub=(x_bar-mu)[k-1]-np.sqrt(s_x[k-1, k-1]/n)*t.ppf(alpha/2, df=n-1)   

    return ci_lb,ci_ub

def variance_ratio_ci(x,diag_sigma, alpha=0.05, k=1):
    """
    One sample confidence interval for variance ratio
    parameters
    ----------
    x: samples from the distribution to be checked
    diag_sigma: diagonal of fixed covariance matrix in the hypothesis
    alpha: (1-alpha)*100% confidence interval is calculated 
    k: confidence interval is computed based on k-th coordinate
    
    Returns
    -------
    ci_lb: lower bound of confidence interval
    ci_ub: upper bound of confidence interval
    """
    x_bar = np.mean(x, axis = 0)
    
    s_x = np.cov(x.T)
    n = len(x)
    if len(x.T)==1:
        ds_x = s_x
    else:
        ds_x = np.diag(s_x)
    
    lb = np.zeros(len(x.T))
    up = np.zeros(len(x.T))
    for i in np.arange(len(lb)):
        if len(x.T)==1:
            ratio_0 = ds_x/diag_sigma
        else:
            ratio_0 = ds_x[i]/diag_sigma[i]
        lb[i] = (n-1)*ratio_0/chi2.ppf(1-alpha/2, df=n-1)
        up[i] = (n-1)*ratio_0/chi2.ppf(alpha/2, df=n-1)
    return lb[k-1], up[k-1]

def quantile_diff_ci(x, mu, diag_sigma, p = 0.5, alpha = 0.05, k=1):
    x_bar = np.mean(x, axis = 0)
    s_x = np.cov(x.T)
    n = len(x)
    if len(x.T)==1:
        s = s_x**(0.5)
    else:
        s = np.diag(s_x)**(0.5)
#    c = np.sqrt((n-1)/2)*gamma((n-1)/2)/gamma(n/2)
    c = 1
    zp = scipy.stats.norm.ppf(p)
    z_alpha = scipy.stats.norm.ppf(1-alpha/2)
    if len(x.T)==1:
        lb = x_bar+zp*c*s-z_alpha*s/np.sqrt(n)*np.sqrt(1+n*zp**2*(c**2-1))
        ub = x_bar+zp*c*s+z_alpha*s/np.sqrt(n)*np.sqrt(1+n*zp**2*(c**2-1))
    else:
        lb = x_bar[k-1] + zp*c*s[k-1]-z_alpha*s[k-1]/np.sqrt(n)*np.sqrt(1+n*zp**2*(c**2-1))
        ub = x_bar[k-1] + zp*c*s[k-1]+z_alpha*s[k-1]/np.sqrt(n)*np.sqrt(1+n*zp**2*(c**2-1))
    return lb-zp*(diag_sigma[k-1]**0.5)-mu[k-1], ub-zp*(diag_sigma[k-1]**0.5)-mu[k-1]


    
def ci_diff_mean_two_sample(x, y, alpha=0.05, k=1):
    """
    Two sample confidence interval for mean difference
    parameters
    ----------
    x: samples from the first distribution
    y: samples from the second distribution
    alpha: (1-alpha)*100% confidence interval is calculated 
    k: confidence interval is computed based on k-th coordinate
    
    Returns
    -------
    ci_lb: lower bound of confidence interval
    ci_ub: upper bound of confidence interval
    """
    x_bar = np.mean(x, axis = 0)
    y_bar = np.mean(y, axis = 0)
    s_x = np.cov(x.T)
    s_y = np.cov(y.T)
    n = len(x)
    p = len(x.T)
    s_p =  0.5*(s_x+s_y)
    if p==1:
        ci_lb=(x_bar-y_bar)-np.sqrt(2/n*s_p)*np.sqrt(stats.f.ppf(alpha, 1, 2*n-2))
        ci_ub=(x_bar-y_bar)+np.sqrt(2/n*s_p)*np.sqrt(stats.f.ppf(alpha, 1, 2*n-2))
    else:
        ci_lb=(x_bar-y_bar)[k-1]-np.sqrt(2/n*s_p[k-1, k-1])\
        *np.sqrt((2*(n-1)*p/(2*n-p-1))*stats.f.ppf(1-alpha, p, 2*n-p-1))
        ci_ub=(x_bar-y_bar)[k-1]+np.sqrt(2/n*s_p[k-1, k-1])\
        *np.sqrt((2*(n-1)*p/(2*n-p-1))*stats.f.ppf(1-alpha, p, 2*n-p-1))     

    return ci_lb,ci_ub



def ci_f_variance(x,y, alpha=0.05, k=1):
    """
    Two sample confidence interval for variance ratio
    parameters
    ----------
    x: samples from the first distribution
    y: samples from the second distribution
    alpha: (1-alpha)*100% confidence interval is calculated 
    k: confidence interval is computed based on k-th coordinate
    
    Returns
    -------
    ci_lb: lower bound of confidence interval
    ci_ub: upper bound of confidence interval
    """
    
    x_bar = np.mean(x, axis = 0)
    
    y_bar = np.mean(y, axis = 0)
    
    s_x = np.cov(x.T)
    
    s_y = np.cov(y.T)
    if len(x.T)==1:
        ds_x = s_x
        ds_y = s_y
    else:
        ds_x = np.diag(s_x)
    
        ds_y = np.diag(s_y)
    
    lb = np.zeros(len(x.T))
    up = np.zeros(len(x.T))
    for i in np.arange(len(lb)):
        if len(x.T)==1:
            F = ds_x/ds_y
        else:
            F = ds_x[i]/ds_y[i]
        lb[i] = stats.f.ppf(alpha/2, len(x)-1, len(y)-1)/(F)
        up[i] = stats.f.ppf(1-alpha/2, len(x)-1, len(y)-1)/(F)
    return (lb[k-1]-1), (up[k-1]-1)


def Laplace_approx(weights, means, stds):
    """Approximate mixture of one-dimensional Normal using Laplace approximation
    
    parameters
    ----------
    witghts : `numpy.ndarray`, shape (n,)
      weights of mixture of normals 
    means : `numpy.ndarray`, shape (n,)
      means of normal distributions
    stds: `numpy.ndarray`, shape (n,)
      standarad deviations of normal distributions
    Returns
    -------
    samples: `numpy.ndarray`, shape (n,d)
      approximate samples generated using BBVI
    """
    def log_density_mixture(x, weights=weights, means=means, stds=stds):
        l=len(weights)
        density=0
        for i in range(l):
            density += weights[i]*norm.pdf(x, means[i], stds[i])
        return np.log(density)  
    
    Hessian_mixture = nd.Hessian(log_density_mixture)
    laplace_mean = opt.fmin(lambda x: -log_density_mixture(x), 0.01)
    laplace_std = np.sqrt(-Hessian_mixture(laplace_mean))
    
    return dict(laplace_mean=laplace_mean,
                laplace_std=laplace_std)
    
###### R^2 iteration plot
def maximum_r_square(x, y):
    result = 0
    for i in range(x.shape[1]):
        r_2 = np.corrcoef(x[:, i], y[:, i])[0, 1]**2
        result = max(result, r_2)
    return result

def every_r_square(x, y):
    result = []
    for i in range(x.shape[1]):
        result.append(np.corrcoef(x[:, i], y[:, i])[0, 1]**2)
    return result

def maximum_r_square_path(resamples_path, samples):
    result = []
    for i in range(resamples_path.shape[2]):
        result.append(maximum_r_square(resamples_path[:,:, i], samples))
    return result 

def coordinate_r_square_path(resamples_path, samples, coordinate):
    resamples_path_coordinate = resamples_path[:, coordinate-1, :]
    samples_coordinate = samples[:, coordinate-1]
    result = []
    for i in range(resamples_path_coordinate.shape[1]):
        result.append(np.corrcoef(resamples_path_coordinate[:, i], 
                                  samples_coordinate)[0, 1]**2)
    return result 


from scipy import stats
###### R^2 iteration plot
def maximum_rank_coor(x, y):
    result = 0
    for i in range(x.shape[1]):
        r_2 = stats.spearmanr(x[:, i], y[:, i])[0]
        result = max(result, r_2)
    return result

def every_rank_coor(x, y):
    result = []
    for i in range(x.shape[1]):
        result.append(stats.spearmanr(x[:, i], y[:, i])[0])
    return result

def maximum_rank_coor_path(resamples_path, samples):
    result = []
    for i in range(resamples_path.shape[2]):
        result.append(maximum_rank_coor(resamples_path[:,:, i], samples))
    return result 

def coordinate_rank_coor_path(resamples_path, samples, coordinate):
    resamples_path_coordinate = resamples_path[:, coordinate-1, :]
    samples_coordinate = samples[:, coordinate-1]
    result = []
    for i in range(resamples_path_coordinate.shape[1]):
        result.append(stats.spearmanr(resamples_path_coordinate[:, i], samples_coordinate)[0])
    return result 







