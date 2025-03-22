import numpy as np
import pickle
from scipy import optimize

def inference(x, sigma=1):
    """ 
    Computes the median + uncertainties 
    (1sigma or 2sigma) 

    Args: 
        x: array. usually samples from MCMC
    Returns:
        array: median, lower uncertainty, upper uncertainty
    """

    if sigma == 1:
        percentile = [16, 50, 84] # 1sigma
    elif sigma ==2:
        percentile = [2.5, 50, 97.5] # 2sigma

    y = np.percentile(x, percentile)
    
    med  = y[1]
    errl = y[1] - y[0]
    erru = y[2] - y[1]

    return med, errl, erru

def read_interp_func(filename):
    ''' read interpolation function '''
    f = open(filename, 'rb')
    interp_func = pickle.load(f)
    f.close()
    return interp_func

def lumi_given_str(Mp, zhomo, Mstar, a, radius, interp_func_radius):
    '''
    function to solve for x in f(x) = y 
    technically, given Mp, Zhomo, Mstar, a, and radius 
    what is the luminosity of the planet.
    '''
    interp_fn2 = lambda x: interp_func_radius((Mp, zhomo, Mstar, x, a)) - radius # find x for f(x) = y
    try:
        lumi = optimize.newton(interp_fn2, 1) # 1 is just a starting value 
    except (ValueError, RuntimeError):
        print('Invalid value!')
        return 10

    return lumi



