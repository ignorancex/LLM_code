import numpy as np
from scipy.stats import norm, chi2, beta


def success_counts(t_obs, t_ref):
    extreme_counts = np.sum(t_ref>t_obs)
    if extreme_counts: 
        return extreme_counts
    else: 
        return 1
    
    
def t_to_pvalue_empirical(t_obs, t_ref):
    extreme_counts = success_counts(t_obs, t_ref)
    return extreme_counts/len(t_ref)
    
    
def t_to_pvalue(t, df):
    return chi2.sf(t, df)


def pvalue_to_Zscore(p):
    return norm.ppf(1-p) # equivalent to -norm.ppf(p)


def Zscore_to_pvalue(Z):
    return 1-norm.cdf(Z)


def Jefferys_interval(n_counts, n_trials, z = 1):
    alpha = Zscore_to_pvalue(z)
    lower = beta.ppf(alpha, n_counts+0.5, n_trials-n_counts+0.5)
    upper = beta.ppf(1-alpha, n_counts+0.5, n_trials-n_counts+0.5)
    return np.array([np.where(n_counts>0, lower, 0.), np.where(n_counts < n_trials, upper, 1.)]).T