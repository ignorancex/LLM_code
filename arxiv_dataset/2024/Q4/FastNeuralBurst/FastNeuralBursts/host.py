# author: Koustav Konar <n.koustav.konar@gmail.com>

# Host model for dispersion contribution from FRB host galaxy
# Two models available
#   1. Lognormal
#   2. Gaussian (truncated at zero)

import numpy as np
 
def lognormal_host_dm(z, exp_mu, sigma):
    '''
    Samples DM_host from a lognormal distribution
    ------------
    PARAMETERS: 
        exp_mu (torch element): median of lognormal
        sigma (torch element): scale of lognormal
    ------------
    RETURNS:
        dm_host, del_mu, del_sigma (arrays): host_dm, partial derivatives wrt free parameters (median, scale)
    -------------
    '''
    # print(exp_mu.to('cpu'), type(exp_mu))
    mu = np.log(exp_mu.to('cpu')) - 0.5*sigma**2
    sigma = sigma.item()
    num = len(z)

    dm_host = np.random.lognormal(mu, sigma, num)/(1+z)
    dm_host = dm_host.ravel()
    
    # del_mu = np.exp(mu + 0.5 * sigma**2) / (1+z)
    del_mu = np.exp(0.5 * sigma**2) / (1+z)
    del_mu = np.array(del_mu)
    # del_mu = del_mu.ravel()
    del_sigma = sigma * del_mu
    # del_sigma = del_sigma.ravel()
    
    return dm_host, del_mu, del_sigma


def gaussian_host_dm(z, mean_gauss, std_dev_gauss):
    '''
    Samples DM_host from a truncated (at zero) Gaussian distribution
    ------------
    PARAMETERS:
        z (array): redshift of FRBs
        mean_gauss (torch element): mean of Gaussian
        std_dev_gauss (torch element): standard deviation of Gaussian
    ------------
    RETURNS:
        dm_host, del_mu, del_sigma (arrays): host_dm, 1, 2 (yet to be added)
    ------------
    '''
    mean = mean_gauss.item()  
    std_dev = std_dev_gauss.item()  
    num = len(z)
    dm_host = np.zeros(num)  

    for i in range(num):
        sample = -1
        while sample <= 0:
            sample = np.random.normal(mean, std_dev)
        dm_host[i] = sample
    
    dm_host = dm_host/ (1 + z)
    return dm_host, 1, 2
