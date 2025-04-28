#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yu Wang
"""
import numpy as np
from scipy.stats import t
from scipy.stats import chi2
from scipy.special import gamma
from scipy.stats import binom

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
        ci_lb=(x_bar-mu)-np.sqrt(s_x/n)*t.ppf(alpha/2, df=n-1)
        ci_ub=(x_bar-mu)+np.sqrt(s_x/n)*t.ppf(alpha/2, df=n-1)
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




def quantile_diff_ci(x, x_0, p = 0.5, alpha = 0.05, k=1):
    n = len(x)
    l = int(binom.ppf(alpha/2, n, p))
    u = int(binom.ppf(1-alpha/2, n, p)+1)
    x_k = sorted(x[:, k-1])
    return x_k[l-1]-np.quantile(x_0[:, k-1], q=p), x_k[u-1]-np.quantile(x_0[:, k-1], q=p)

#def quantile_diff_ci(x, mu, diag_sigma, p = 0.5, alpha = 0.05, k=1):
#    x_bar = np.mean(x, axis = 0)
#    s_x = np.cov(x.T)
#    n = len(x)
#    if len(x.T)==1:
#        s = s_x**(0.5)
#    else:
#        s = np.diag(s_x)**(0.5)
#    c = np.sqrt((n-1)/2)*gamma((n-1)/2)/gamma(n/2)
#    zp = scipy.stats.norm.ppf(p)
#    z_alpha = scipy.stats.norm.ppf(1-alpha/2)
#    if len(x.T)==1:
#        lb = x_bar+zp*c*s-z_alpha*s/np.sqrt(n)*np.sqrt(1+n*zp**2*(c**2-1))
#        ub = x_bar+zp*c*s+z_alpha*s/np.sqrt(n)*np.sqrt(1+n*zp**2*(c**2-1))
#    else:
#        lb = x_bar[k-1] + zp*c*s[k-1]-z_alpha*s[k-1]/np.sqrt(n)*np.sqrt(1+n*zp**2*(c**2-1))
#        ub = x_bar[k-1] + zp*c*s[k-1]+z_alpha*s[k-1]/np.sqrt(n)*np.sqrt(1+n*zp**2*(c**2-1))
#        
#    return lb-zp*(diag_sigma[k-1]**0.5)-mu[k-1], ub-zp*(diag_sigma[k-1]**0.5)-mu[k-1]
#    
    
    

    
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