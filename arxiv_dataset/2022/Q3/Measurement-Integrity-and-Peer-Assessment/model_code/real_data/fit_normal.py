"""
Functions that fit a normal distribution to real grading data.

@author: Noah Burrell <burrelln@umich.edu>
"""

from scipy.optimize import fmin
from scipy.stats import norm, entropy
from statistics import mean, stdev

from load import load_all

def discretize_normal(mu, sigma, lower, upper):
    """
    Transforms a given normal distrubtion into a discrete distribution over the integer range [lower, upper].
    Works by placing the probability density for every real number that rounds to a given int into mass on that int.
    All density below lower (resp. above upper) gets placed on that value.

    Parameters
    ----------
    mu : int or float
        The mean of the normal distribution.
    sigma : int or float
        The standard deviation of the normal distribution.
    lower : int
        The lower bound of the discrete integer interval.
    upper : int
        The upper bound of the discrete integer interval.

    Returns
    -------
    dist : list of float
        Discrete probability over the integer range [lower, upper].

    """
    dist = []
    
    i = lower
    while i < upper:
        total = sum(dist)
        x = i + 0.5
        new = norm.cdf(x, mu, sigma) - total
        dist.append(new)
        i += 1
        
    total = sum(dist)
    last = 1 - total
    dist.append(last)
        
    return dist

def kl_divergence(params):
    """
    Discretizes a given normal distribution and computes the KL divergence between that and a global (discrete) empirical distribution.
    Empirical distribution is global because of how scipy optimization function works.
    
    Parameters
    ----------
    params : double (or list) of 2 int/floats 
        First float is mu, the mean of the normal distribution.
        Second float is sigma, the standard deviation of the normal distribution.

    Returns
    -------
    kl_div : float
        KL divergence between global empirical distribution and the discretized normal distribution characterized by "params".

    """
    
    mu, sigma = params
    global empirical_dist, lower, upper
    
    discrete_normal = discretize_normal(mu, sigma, lower, upper)
    
    kl_div = entropy(empirical_dist, discrete_normal)
    
    return kl_div

def minimize_kl(mu, sigma):
    """
    Optimization function to fit a normal distribution to (discrete) data by minimizing the KL divergence between a discretized normal distribution and the empirical distribution from the data. 

    Parameters
    ----------
    mu : int or float
        The mean of the normal distribution.
    sigma : int or float
        The standard deviation of the normal distribution.
        
    Both parameters are initial values from which to start the optimiziation.

    Returns
    -------
    m_opt : float
        The mean of the optimal normal distribution.
    s_opt : float
        The standard deviation of the optimal normal distribution.
    fopt : float
        The optimal KL divergence (achieved by the discretized distribution characterized by m_opt and s_opt).
        Gives a measure of the fit of the optimal distribution.

    """
    opt = fmin(func=kl_divergence, x0=[mu,sigma], full_output=True)
    m_opt, s_opt, fopt = opt[0][0], opt[0][1], opt[1]
    return m_opt, s_opt, fopt

def fit_data(kl_div=False, coarsened=False):
    """
    Fits normal distributions to the real data for each of the 4 available semesters. 

    Parameters
    ----------
    kl_div : bool, optional
        If True, data is fit using the minimize_kl() function. 
        If False, data is fit using scipy.stats.norm.fit().
        The default is False.
    coarsened : bool, optional
        Set to True if the grades were coarsened when it was read in (meaning mapped into the standard [0, 10] integer range). 
        Set to False if grades were not coarsened.
        The default is False.

    Returns
    -------
    opt_params : list of triples.
        [ (m_opt, s_opt, f_opt) ]
        m_opt : float
            The mean of the optimal normal distribution.
        s_opt : float
            The standard deviation of the optimal normal distribution.
        f_opt : float or None
            The optimal KL divergence (achieved by the discretized distribution characterized by m_opt and s_opt).
            Gives a measure of the fit of the optimal distribution.
            "None" if fitting is done with a method that doesn't give an evaluation score.

    """
    
    all_semesters = load_all(coarsened)
    
    opt_params = []
    
    for i, semester in enumerate(all_semesters):
        #students = semester[0]
        submissions = semester[1]
        
        data = [sub.true_grade for sub in submissions]
        
        if kl_div:
            sample_mean = mean(data)
            sample_sd = stdev(data)
            
            l = len(data)
            
            global empirical_dist, lower, upper
            lower = 0
            if coarsened: 
                upper = 10
            else:
                if i < 2:
                    upper = 100
                else:
                    upper = 30
                    
            empirical_dist = [data.count(i)/l for i in range(upper + 1)]
            m_opt, s_opt, f_opt = minimize_kl(sample_mean, sample_sd)
        
        else:
            f_opt = None
            m_opt, s_opt = norm.fit(data)
            
        opt_params.append((m_opt, s_opt, f_opt))
        
    return opt_params