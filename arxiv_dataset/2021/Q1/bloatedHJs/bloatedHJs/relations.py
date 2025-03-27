import numpy as np


def ln_normal(xmodel, mu, sigma):
    '''
    computes probability assuming xmodel has a normal distribution
    xmodel ~ N(mu, sigma)
    '''
    lnC = np.log(2 * np.pi * sigma**2)
    significance = (xmodel -  mu) / sigma
    lnp = significance**2 

    return -0.5 * ( lnC + lnp )

def ln_lognormal(xmodel, mu, sigma):
    ''' 
    computes probability assuming xmodel has a lognormal distribution
    xmodel ~ N(ln(mu), sigma) 
    '''

    lnC = np.log(2 * np.pi * sigma ** 2 * xmodel ** 2)
    significance = (np.log(xmodel) - mu) / sigma
    lnp = significance**2 
    
    return -0.5 * ( lnC + lnp )

def gaussian(x, alpha):
    K = ( (x - alpha[1]) / alpha[2] ) **2
    
    return alpha[0] * np.exp( -K * 0.5 )

def poly(x, alpha):
    return alpha[0]*x**4 + alpha[1]*x**3 + alpha[2]*x**2 + alpha[3]*x + alpha[4]
