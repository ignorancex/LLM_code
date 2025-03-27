"""
Created by Emille Ishida in 10 June 2016.

Example of functions to be used as input to CosmoABC. 
You are free to customize this functions to your own problem
as long as you respect the input/ouput requirements and 
***
    update the function names into the keywords 

    distance_func
    simulation_func
    prior_func
  
    in the user input file
***. 

Updated by Martin Kilbinger 2019 - 2020.

"""


import numpy as np
import os
from scipy.stats import norm,  multivariate_normal
from scipy.stats import uniform
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from covest import *


def model_cov(p):
    """Linear model.

    input: p - dict: keywords 
                a, scalar - angular coefficient
                b, scalar - linear coefficient
                xmin, xmax, int - bounderies for explanatory variable
                nobs, int - number of observations in a catalog
                cov, matrix - covariance matrix between observations
                

    output: y, array - draw from normal distribution with mean
                        a*x + b and scatter sqrt(cov)
    """
    if bool(p['xfix']):
        try:                
            x = p['dataset1'][:,0]
        except KeyError:
            raise ValueError('Observed data not defined!\nif you are doing distance tests, set xfix=0')

    else:
        x = uniform.rvs(loc=p['xmin'], scale=p['xmax'] - p['xmin'], size=int(p['nobs']))

    
    x.sort()
    ytrue = np.array(p['a']*x + p['b'])

    if isinstance(p['cov_est'], float):
        cov_est = np.loadtxt('cov_est.txt')
        #raise ValueError('Covariance is not a matrix!')
    else:
        cov_est = p['cov']

    y = multivariate_normal.rvs(mean=ytrue, cov=cov_est)

    return np.array([[x[i], y[i], p['a'], p['b']] for i in range(int(p['nobs']))])
    #return np.array([[x[i], y[i]] for i in range(int(p['nobs']))])



def gaussian_prior(par, func=False):
    """
    Gaussian prior.
  
    input: par -> dictionary of parameter values
                  keywords: mean, standard_devitation, 
                            min and max
                  values: all scalars 
           func -> boolean (optional)
                   if True returns the pdf random variable. 
                   Default is False.
    output: scalar (if func=False)
            gaussian probability distribution function (if func=True)
    """

    np.random.seed()    
    dist = norm(loc=par['pmean'], scale=par['pstd'])
    flag = False  
    while flag == False:   
        draw = dist.rvs() 
        if par['min'] < draw and draw < par['max']:
            flag = True
     
    if func == False:
        return draw
    else:
        return dist


def linear_dist(d2, p, abs_b=True):
    """
    Distance between observed and simulated catalogues using
    least squares between observed fitted and simulated parameters a, b.

    input: d2 -> array of simulated catalogue
           p -> dictonary of input parameters

    output: list of 1 scalar (distance)
    """

    data_sim = {}
    data_sim['x'] = d2[:,0]
    data_sim['y'] = d2[:,1]

    data_obs = {}
    data_obs['x'] = p['dataset1'][:,0]
    data_obs['y'] = p['dataset1'][:,1]

    data1_sim = np.array([[data_sim['x'][k], 1] for k in range(data_sim['x'].shape[0])])
    data1_obs = np.array([[data_obs['x'][k], 1] for k in range(data_obs['x'].shape[0])])

    mod_sim0 = sm.OLS(data_sim['y'], data1_sim)
    mod_obs0 = sm.OLS(data_obs['y'], data1_obs)

    mod_sim = mod_sim0.fit()
    mod_obs = mod_obs0.fit()
      
    delta_a = mod_sim.params[0] - mod_obs.params[0]
    if abs_b:
        delta_b = abs(mod_sim.params[1]) - abs(mod_obs.params[1])
    else:
        delta_b = mod_sim.params[1] - mod_obs.params[1]
   
    res = np.sqrt( pow(delta_a, 2) + pow(delta_b, 2) )

    return np.atleast_1d(res)


def linear_dist_noabsb(d2, p):
    """
    Distance between observed and simulated catalogues using
    least squares between observed fitted and simulated parameters a, b.

    input: d2 -> array of simulated catalogue
           p -> dictonary of input parameters

    output: list of 1 scalar (distance)
    """

    return linear_dist(d2, p, abs_b=False)
