import os
import pandas as pd
import numpy as np

from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy import special as sp

import warnings

from diffsph.profiles.templates import *
from diffsph.utils.dictionaries import *
from diffsph.utils.consts import *

#  #######################################################################

def check_cache():
    '''
    Function checks whether the /.diffsph_cache/ folder exists. If it does not exists, it creates it
    
    :return: folder directory name
    :rtype: str
    '''
    DEFAULT_CACHE_DIR = os.path.expanduser("~/.diffsph_cache")
    if not os.path.exists(DEFAULT_CACHE_DIR):
        os.makedirs(DEFAULT_CACHE_DIR)
    return DEFAULT_CACHE_DIR

def load_data(folder):
    '''
    Function loads data from folder
    
    :return: data organized in form of a python dictionary 
    :rtype: dict
    '''
    data = {}
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith('.csv'):
                filepath = os.path.join(dirpath, filename)
                arr = pd.read_csv(filepath, header = None).values.flatten()
                key = os.path.splitext(filename)[0] # remove extension from filename
                data[key] = arr
    return data

def delta_float(inp):
    """
    Float number for variable ``'delta'``

    :param inp: variable ``'delta'`` as *str* (``'kol'``, ``'kra'``, etc.) or *float*   

    :return: float number associated with ``'inp'``
    :rtype: float
    """
    if isinstance(inp, float):
        return inp
    if var_to_str(inp) not in delta_to_float.keys():
        raise KeyError('Turbulence model not found. Please use another input for variable delta')
    return delta_to_float[var_to_str(inp)]

def var_to_str(inp):
    """
    Dictionary for variables ``'delta'``, ``'hyp'``, ``'galaxy'``, ``'ref'`` and ``'rad_temp'``
    
    :param inp: input string or number

    :return: default variable name
    :rtype: str
    """
    for idx in all_names.keys():
        if str(inp) in all_names[idx]:
            return idx
        
def sort_kwargs(**kwargs):
    """
    Function sorts keyword arguments alphabetically
    
    :return: sorted keywords with corresponding entries
    :rtype: dict
    """
    kwkeys = list(kwargs.keys())
    kwkeys.sort()
    return {kwk: kwargs[kwk] for kwk in kwkeys} 
        
def hypothesis_index(hyp):
    """
    Index of the hypothesis (1 for decaying DM or generic scenario, 2 for WIMP self-annihilation).

    :param str hyp: hypothesis: ``'wimp'``, ``'decay'`` or ``'generic'``)

    :return: hypothesis index
    :rtype: int
    """
    if hyp.lower() in all_names['generic']:
        return 1
    elif hyp.lower() in all_names['decay']:
        return 1
    elif hyp.lower() in all_names['wimp']:
        return 2
    raise KeyError('Hypothesis not found. Please use \'generic\' for the generic hypothesis, \'wimp\' for WIMP self-annihilation or \'decay\' for DM decay.')
    

def evaluate(f, x, **kwargs):
    """
    Function converts string into a python function's name and evaluates it
    
    :param f: function to be evaluated
    :param x: first argument of :math:`f`
    
    :return: :math:`f(x)`
    """
    if type(f) == str:
        return eval(f.lower())( x, **kwargs)
    return f( x, **kwargs)

#  TB, FWHM and HFD calculators

def TB(brightness, theta, nu, *args, **kwargs):
    """
    Brightness temperature conversion
    
    .. math::
        T_B = \\frac{c^2}{2\\,k\\,\\nu^2}I_\\nu

    :param brightness: generic brightness function in Jy/sr
    :param theta: angular radius (as the first argument of the generic brighness function)
    :param nu: frequency (as the second argument of the generic brighness function)

    :return: brightness temperature in mK
    """
    return 1e3 * 1e-23 * c0 ** 2 / 2 / kB / (nu * 1e9)**2 * brightness(theta, nu, *args, **kwargs)

def df(func , **kwargs): 
    return 2 * func(**kwargs) - 1

def fwhm(brightness, thmax, *args, **kwargs):
    """
    Full width at half maximum 

    :param brightness: generic brightness function
    :param thmax: signal's angular radius

    :return: Full width at half maximum in arcmin
    """
    
    f0 = lambda th: 2 * brightness(th, *args, **kwargs) / brightness(theta = 1e-128 * thmax, * args, **kwargs) - 1
    
    return 2 * fsolve(lambda th : f0(th), 0.1 * thmax )[0]

def hfd(fluxdens, thmax, *args, **kwargs):
    """
    Half-flux diameter 

    :param brightness: generic brightness function
    :param thmax: signal's angular radius

    :return: Half-flux diameter in arcmin
    """
    f0 = lambda th: 2 * fluxdens(th, *args, **kwargs) / fluxdens(thmax, *args, **kwargs) - 1
    
    return 2 * fsolve(lambda th : f0(th), 0.1 * thmax )[0]

# %%%%%%%%%%%%%%%%%%%%%
# % f_n(x) and g_n(x) %
# %%%%%%%%%%%%%%%%%%%%%

def f(n, x):
    '''
    Basis function in Fourier-expanded brightness formula
    
    .. math::
        f_n(x)=2\\int_x^1\\frac{\\sin(n\\pi y) dy}{\\sqrt{y^2-x^2}}
    
    :return: :math:`f_n` as a function of :math:`x`
    '''
    return 2 * quad(
        lambda y: np.sin(n * np.pi * np.sqrt( y ** 2 + x ** 2) ) / np.sqrt( y ** 2 + x ** 2), 0, np.sqrt(1 - x ** 2)
    )[0]

def g(n, x):
    '''
    Basis function in Fourier-expanded flux density formula
    
    .. math::
        g_n(x)=2\\int_x^1\\sqrt{y^2-x^2}\\sin(n\\pi y) dy
    
    :return: :math:`g_n` as a function of :math:`x`
    '''
    return 2 * quad(lambda y: np.sin(n * np.pi * y) * np.sqrt(y**2 - x**2), x, 1)[0]

# %%%%%%%%%%%%%%%%%%%%%%%%
# % Flux-density kernels %
# %%%%%%%%%%%%%%%%%%%%%%%%

def ker_0(r, dist):
    '''
    .. math::
        \\kappa_0(r,R) = \\frac1{R}\\log\\sqrt{\\frac{R+r}{R-r}}
    '''
    return 1 / 2 / dist * np.log( (dist + r) / (dist - r) )

def ker_1(r, theta, dist):
    '''
    .. math::
        \\kappa_1(\\theta,r,R) = \\frac1{R}\\log\\frac{R\\cos\\theta+\\sqrt{r^2-R^2\\sin^2\\theta}}{\\sqrt{R^2-r^2}}
    '''
    return 1 / dist * np.log( 
        (dist * np.cos(theta) + np.sqrt( r**2 - dist**2 * np.sin(theta)**2 ) ) / np.sqrt(dist**2 - r**2) 
    )

# Flux density factors (full)

def halo_fd_tot(n, dist, rh):
    '''
    Total flux-density halo/bulge factor:
    
    .. math::
        \\mathcal H_n(r_h,R) = 2\\,\\int_0^{r_h}dr\\, r\\, \\kappa_0(r,R) \\frac{\\sin\\left(\\frac{n\\pi r}{r_h}\\right)}r\\ , 
    
    where :math:`R`, :math:`rh` and :math:`n` are, respectively the distance, halo radius and Fourier index 
    
    :return: Halo flux-density factor 
    '''
    return 2 * rh / n / dist * (
        np.cos( n * np.pi * dist / rh ) * ( 
            sp.sici( n * np.pi * (dist / rh + 1) )[1] - sp.sici( n * np.pi * (dist / rh - 1) )[1] 
        ) + np.sin( n * np.pi * dist / rh ) * ( 
            sp.sici( n * np.pi * (dist / rh + 1) )[0] - sp.sici( n * np.pi * (dist / rh - 1) )[0] 
        ) - ( -1 ) ** n * np.log( ( dist + rh ) / ( dist - rh )  )
    )

def halo_fd(n, theta, dist, rh):
    '''
    Partial (:math:`\\theta`\ -dependent) flux-density halo/bulge factor:
    
    .. math::
        \\mathcal H_n(\\theta) = \\mathcal H_n(r_h,R) - 2\\,\\int_{R\\sin(\\theta)}^{r_h}dr\\, r\\, \\kappa_1(r,R,\\theta) \\frac{\\sin\\left(\\frac{n\\pi r}{r_h}\\right)}r \\ ,
    
    where :math:`R`, :math:`rh` and :math:`n` are, respectively the distance, halo radius and Fourier index
    '''
    
    if 0. < theta <= np.arcsin(rh / dist):
        return halo_fd_tot(n, dist, rh) - 2 * np.pi * quad(
            lambda r: 2 * r * np.sin( n * np.pi * r / rh ) / r * ker_1(r, theta, dist), dist * np.sin(theta), rh
        )[0]
    if theta > np.arcsin(rh / dist):
        return halo_fd_tot(n, dist, rh)
    return 0.

# Flux density factors (approximate)

def approxhalo_fd_tot(n, dist, rh):
    '''
    Total flux-density halo/bulge factor (approximate formula):
    
    .. math::
        \\mathcal H_n(r_h,R) \\simeq 4\\pi\\int_0^{r_h}dr\\, r^2 \\frac{\\sin\\left(\\frac{n\\pi r}{r_h}\\right)}r \\ ,
    
    where :math:`R`, :math:`rh` and :math:`n` are, respectively the distance, halo radius and Fourier index 
   
    '''
    return 4 * ( -1 )**( n - 1 ) * rh**2 / n / dist**2

def approxhalo_fd(n, theta, dist, rh):
    '''
    Partial (:math:`\\theta`\ -dependent) flux-density halo/bulge factor (approximate formula):
    
    .. math::
        \\mathcal H_n(\\theta) = \\mathcal H_n(r_h,R) - 2\\,\\int_{R\\sin(\\theta)}^{r_h}dr\\, r\\, \\kappa_1(r,R,\\theta) \\frac{\\sin\\left(\\frac{n\\pi r}{r_h}\\right)}r
    
    where :math:`R`, :math:`rh` and :math:`n` are, respectively the distance, halo radius and Fourier index 
    '''
    return approxhalo_fd_tot(n, dist, rh) - 2 * np.pi *  rh**2 / dist**2 * g( n, x = dist * np.sin(theta) / rh ) 

