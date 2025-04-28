import pandas as pd
import numpy as np

import csv
import hashlib

from diffsph.profiles.massmodels import *
from diffsph.spectra.synchrotron import *

from diffsph.utils.dictionaries import *
from diffsph.utils.tools import *
from diffsph.utils.consts import *

cache_path = check_cache()
cache = load_data(cache_path)

# ######################### ################# ######################### 
# ######################### Exact Predictions ######################### 
# ######################### ################# ######################### 

############################## Emissivity

def synch_emissivity(r, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, manual = False, high_res = False, accuracy = 1, **kwargs):
    '''
    Model-specific emissivity from synchrotron radiation
    
    :param r: galactocentric distance in kpc
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0` (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)


    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Emissivity in erg/cm :math:`{}^3` /Hz/s/sr
    :rtype: float
    '''
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')

    rkeys = kwargs.keys() - ( kwargs.keys() & {'self_conjugate', 'sv', 'width', 'rate'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}   
    
    arg_str = str([nu, var_to_str(galaxy), var_to_str(rad_temp), var_to_str(hyp), ratio, D0, var_to_str(delta), B, manual, high_res, accuracy, sort_kwargs(**rkwargs)]) 
    filehash = hashlib.sha256(arg_str.encode()).hexdigest()

    sc = kwargs.get('self_conjugate', True)
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    bare = 0.    
    
    if 0 <= r < rh: 
        if filehash + '.csv' not in os.listdir(cache_path):
            m = which_N(nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, high_res, accuracy, **rkwargs)
            coeffs = pd.read_csv(
                os.path.join(cache_path, filehash + '.csv' ), header = None
            ).values.flatten()
            
        elif filehash not in cache.keys():
            coeffs = pd.read_csv(
                os.path.join(cache_path, filehash + '.csv' ), header = None
            ).values.flatten()

        else:
            coeffs = cache[filehash]

        bare = cm_to_kpc * sum(
            coeffs[n - 1] * np.sin( n * np.pi * r / rh ) / r for n in range(1, len(coeffs)) 
        )                    

        if hyp in all_names['decay']:
            return 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * bare
        elif hyp in all_names['wimp']:
            return 1e-9 * kwargs['sv']  / 2 ** (not sc) / 2 / kwargs['mchi']  ** 2 / 4 / np.pi * bare
        else:
            return cm_to_kpc**2 * kwargs['rate'] * bare

    return 0.

############################## Brightness

def synch_brightness(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, manual = False, high_res = False, accuracy = 1, **kwargs):
    '''
    Model-specific brightness from synchrotron radiation
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0`. (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)


    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile

    :return: Brightness in Jy/sr
    :rtype: float
    '''    
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')

    rkeys = kwargs.keys() - ( kwargs.keys() & {'self_conjugate', 'sv', 'width', 'rate'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}   
    
    arg_str = str([nu, var_to_str(galaxy), var_to_str(rad_temp), var_to_str(hyp), ratio, D0, var_to_str(delta), B, manual, high_res, accuracy, sort_kwargs(**rkwargs)]) 
    filehash = hashlib.sha256(arg_str.encode()).hexdigest()
    
    sc = kwargs.get('self_conjugate', True)
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    thmax = np.arcsin(rh / dist)
    bare = 0.

    if 0 <= theta < thmax * rad_to_arcmin:
        if filehash + '.csv' not in os.listdir(cache_path):
            m = which_N(nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, high_res, accuracy, **rkwargs)
            coeffs = pd.read_csv(
                os.path.join(cache_path, filehash + '.csv' ), header = None
            ).values.flatten()
            
        elif filehash not in cache.keys():
            coeffs = pd.read_csv(
                os.path.join(cache_path, filehash + '.csv' ), header = None
            ).values.flatten()

        else:
            coeffs = cache[filehash]
            
            
        bare = sum(
            coeffs[n - 1] * f( n, theta / rad_to_arcmin / thmax) for n in range(1, len(coeffs)) 
        )
        
        if hyp in all_names['decay']:
            return 1e23 * 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * bare
        elif hyp in all_names['wimp']:
            return 1e23 * 1e-9 * kwargs['sv'] / 2 ** (not sc) / 2 / kwargs['mchi'] ** 2 / 4 / np.pi * bare
        else:
            return 1e23 * cm_to_kpc**2 * kwargs['rate'] * bare

    return 0.

def synch_TB(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, manual = False, high_res = False, accuracy = 1, **kwargs):
    '''
    Model-specific brightness temperature from synchrotron radiation
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0`. (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)


    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Brightness temperature in mK
    '''
    return TB(synch_brightness, theta, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, high_res, accuracy, **kwargs)


############################## Flux density

def synch_flux_density(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, manual = False, high_res = False, accuracy = 1, **kwargs):
    '''
    Model-specific flux density from synchrotron radiation
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0`. (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)


    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Flux density in µJy
    '''
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    
    rkeys = kwargs.keys() - ( kwargs.keys() & {'self_conjugate', 'sv', 'width', 'rate'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}   
    
    arg_str = str([nu, var_to_str(galaxy), var_to_str(rad_temp), var_to_str(hyp), ratio, D0, var_to_str(delta), B, manual, high_res, accuracy, sort_kwargs(**rkwargs)]) 
    filehash = hashlib.sha256(arg_str.encode()).hexdigest()
    
    sc = kwargs.get('self_conjugate', True)
    
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    thmax = np.arcsin(rh / dist)
    
    bare = 0.
    
    if filehash + '.csv' not in os.listdir(cache_path):
        m = which_N(nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, high_res, accuracy, **rkwargs)
        coeffs = pd.read_csv(
            os.path.join(cache_path, filehash + '.csv' ), header = None
        ).values.flatten()
            
    elif filehash not in cache.keys():
        coeffs = pd.read_csv(
            os.path.join(cache_path, filehash + '.csv' ), header = None
        ).values.flatten()

    else:
        coeffs = cache[filehash]
        
    bare = sum(
        coeffs[n - 1] * halo_fd( n , theta / rad_to_arcmin, dist, rh) for n in range(1, len(coeffs)) 
    )
                
    if hyp in all_names['decay']:
        return 1e23 * 1e6 * 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * bare
    elif hyp in all_names['wimp']:
        return 1e23 *  1e6 * 1e-9 * kwargs['sv'] / 2 ** (not sc) / 2 / kwargs['mchi'] ** 2 / 4 / np.pi * bare
    else:
        return 1e23 * 1e6 * cm_to_kpc**2 * kwargs['rate'] * bare


# ######################### ####################### ######################### 
# ######################### Approximate Predictions ######################### 
# ######################### ####################### ######################### 

############################## Emissivity


def synch_emissivity_approx(r, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, regime = 'B', manual = False, **kwargs):
    '''
    Model-specific emissivity from synchrotron radiation in the Regime "A", "B" or "C" approximations
    
    :param r: galactocentric distance in kpc
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    - ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    - ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Emissivity in erg/cm :math:`{}^3` /Hz/s/sr
    '''
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
        
    sc = kwargs.get('self_conjugate', True)
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    
    tau0 = rh ** 2 / D0 / cm_to_kpc **2 
    
    rkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'self_conjugate', 'sv', 'width', 'rate', 'Gamma'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}
    
    if 0 <= r < rh: 
        
        if hyp == 'wimp': 
            return 1e-9 * kwargs['sv'] / 2 ** (not sc) / 2 / kwargs['mchi'] ** 2 / 4 / np.pi * Hem(
                r, galaxy, rad_temp, 'wimp', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'wimp', mchi = kwargs['mchi'], channel = kwargs['channel'])
        elif hyp == 'decay': 
            return 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * Hem(
                r, galaxy, rad_temp, 'decay', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'decay', mchi = kwargs['mchi'], channel = kwargs['channel'])
        else:
            return 1e-9 * cm_to_kpc**3 * kwargs['rate'] *  Hem(
                r, galaxy, rad_temp, 'generic', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'generic', Gamma = kwargs['Gamma'])
        
    return 0.

############################## Brightness

def synch_brightness_approx(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, regime = 'B', manual = False, **kwargs):
    '''
    Model-specific brightness from synchrotron radiation in the Regime "A", "B" or "C" approximations
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    - ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    - ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Brightness in Jy/sr
    '''
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    
    sc = kwargs.get('self_conjugate', True)
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    
    tau0 = rh ** 2 / D0 / cm_to_kpc **2 

    rkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'self_conjugate', 'sv', 'width', 'rate', 'Gamma'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}
    
    if 0 <= theta < np.arcsin(rh / dist) * rad_to_arcmin: 
        
        if hyp == 'wimp': 
            return 1e23 * 1e-9 * kwargs['sv'] / 2 ** (not sc) / 2 / kwargs['mchi'] ** 2 / 4 / np.pi * Hbr(
                theta, galaxy, rad_temp, 'wimp', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'wimp', mchi = kwargs['mchi'], channel = kwargs['channel'])
        elif hyp == 'decay': 
            return 1e23 * 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * Hbr(
                theta, galaxy, rad_temp, 'decay', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'decay', mchi = kwargs['mchi'], channel = kwargs['channel'])
        else:
            return 1e23 * 1e-9 * cm_to_kpc**2 * kwargs['rate'] *  Hbr(
                theta, galaxy, rad_temp, 'generic', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'generic', Gamma = kwargs['Gamma'])

    return 0.
                            
def synch_TB_approx(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, regime = 'B', manual = False, **kwargs):
    '''
    Model-specific brightness temperature in the Regime "A", "B" or "C" approximations
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    - ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    - ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Brightness temperature in mK
    '''
    return TB(synch_brightness_approx, theta, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, regime, manual, **kwargs)
    
    
def synch_flux_density_approx(theta, nu, galaxy, rad_temp, hyp = 'wimp', ratio = 1, D0 = 3e28, delta = 'kol', B = 2, regime = 'B', manual = False, **kwargs):
    '''
    Model-specific flux density from synchrotron radiation in the Regime "A", "B" or "C" approximations
    
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    - ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    - ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: Flux density in µJy
    '''
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    
    sc = kwargs.get('self_conjugate', True)
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    
    tau0 = rh ** 2 / D0 / cm_to_kpc **2 

    rkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'self_conjugate', 'sv', 'width', 'rate', 'Gamma'} )
    rkwargs = {rk: kwargs[rk] for rk in rkeys}
    
    if 0 <= theta < np.arcsin(rh / dist) * rad_to_arcmin: 
        
        if hyp == 'wimp': 
            return 1e23 * 1e6 * 1e-9 * kwargs['sv'] / 2 ** (not sc) / 2 / kwargs['mchi'] ** 2 / 4 / np.pi * Hfd(
                theta, galaxy, rad_temp, 'wimp', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'wimp', mchi = kwargs['mchi'], channel = kwargs['channel'])
        elif hyp == 'decay': 
            return 1e23 * 1e6 * 1e-9 * kwargs['width'] / kwargs['mchi'] / 4 / np.pi * Hfd(
                theta, galaxy, rad_temp, 'decay', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'wimp', mchi = kwargs['mchi'], channel = kwargs['channel'])
        else:
            return 1e23 * 1e6 * 1e-9 * cm_to_kpc**2 * kwargs['rate'] *  Hfd(
                theta, galaxy, rad_temp, 'generic', ratio, regime, manual, **rkwargs
            ) * X(nu, tau0, delta, B, 'generic', Gamma = kwargs['Gamma'])

    return 0.


# ####################################################### Auxiliary functions

def coeff(n, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual = False, **kwargs):
    '''
    n-th coefficient participating in the Fourier-expanded Green's function solution of the CRE transport equation 
        
    .. math:: 
        s_n = h_n\\times X_n
    
    :param n: order of the halo/bulge factor
    :param theta: angular radius in arcmin
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    - ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    - ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile

    :return: `n`-th coefficient in the `which_N` function  
    '''
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    tau0 = rh ** 2 / D0 / cm_to_kpc **2 
    if not manual:
        ref = kwargs['ref']
        
        if hyp in ('wimp', 'decay'): 
            return h(n, galaxy, rad_temp, hyp, ratio, ref = ref) * X(nu, tau0 / n ** 2 , delta, B, hyp, 
                                                                     mchi = kwargs['mchi'], 
                                                                     channel = kwargs['channel'])
        
        return h(n, galaxy, rad_temp, hyp, ratio, ref = ref) * X(nu, tau0 / n ** 2 , delta, B, 'generic', 
                                                                     Gamma = kwargs['Gamma'])
    
    hkeys = kwargs.keys() - ( kwargs.keys() & {'mchi', 'channel', 'Gamma'} )
    hargs = {hk: kwargs[hk] for hk in hkeys}                 
    
    if hyp in ('wimp', 'decay'): 
        return h(n, galaxy, rad_temp, hyp, ratio, manual = True, **hargs) * X(nu, tau0 / n ** 2 , delta, B, hyp, 
                                                                     mchi = kwargs['mchi'], 
                                                                     channel = kwargs['channel'])
        
    return h(n, galaxy, rad_temp, hyp, ratio, manual = True, **hargs) * X(nu, tau0 / n ** 2 , delta, B, 'generic', 
                                                                     Gamma = kwargs['Gamma'])


def which_N(nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual = False, high_res = False, accuracy = 1, **kwargs):
    '''
    Determines at which order should the Fourier-expanded Green's function solution be truncated and stores the associated :math:`s_n = h_n\\times X_n` coefficients as an array in the ``/cache`` folder
        
    :param nu: frequency in GHz
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and half-light radii (default value = 1) 
    :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm :math:`{}^2`/s (default value = :math:`3\\times 10^{28}` cm :math:`{}^2` /s) 
    :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    :type delta: float, str
    :param B: magnitude of the magnetic field's smooth component in :math:`\\mu` G (default value :math:`= 2 \\mu` G) 
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``) 
    :param bool high_res: spatial resolution. If ``'True'``, :func:`synch_emissivity` computes as many terms as needed in order to converge at :math:`r=0`. (default value = ``'False'``) 
    :param accuracy: theoretical accuracy in % (default value = 1%)

    Keyword arguments
        
    * ``hyp = 'wimp'``    (default)                    
    
    :param sv: annihilation rate (annihilation cross section times relative velocity) :math:`\\sigma v` in cm :math:`{}^3`/s (default value = :math:`3 \\times 10^{-26}` cm :math:`{}^3` /s)
    :param self_conjugate: if set ``'True'`` (default value) the DM particle is its own antiparticle
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: annihilation channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
    
    * ``hyp = 'decay'``                          
    
    :param width: decay width of the DM particle in 1/s
    :param mchi: mass of the DM particle in GeV/c :math:`{}^2`
    :param str channel: decay channel: :math:`b\\bar b` (``'bb'``), :math:`\\mu^+ \\mu^-` (``'mumu'``), :math:`W^+ W^-` (``'WW'``), etc.
                        
    * ``hyp = 'generic'``
    
    :param Gamma: power-law exponent of the generic CRE source (:math:`1.1 < \\Gamma < 3`)
    :param rate: CRE production rate in 1/s
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``
        
    :param rs: scale radius in kpc
    :param rhos: characteristic density in GeV/cm :math:`{}^3`
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
            
    :return: series truncation order *N*
    '''
    arg_str = str([nu, var_to_str(galaxy), var_to_str(rad_temp), var_to_str(hyp), ratio, D0, var_to_str(delta), B, manual, high_res, accuracy, sort_kwargs(**kwargs)])
    
    filename = hashlib.sha256(arg_str.encode()).hexdigest() + ".csv"
    
    # Initiate loop          
    m = 1

    # Inititate total sums
    
    S = coeff(1, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, **kwargs)
    Slow = S
    
    # Inititate series 
    slist = [S, coeff(2, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, **kwargs)]
    
    if high_res:

        # Loop
        while np.abs(slist[-1] / S) > 0.01 * accuracy:
            m += 1
            S += slist[-1]

            # Add a new entry to list if the accuracy is < 1% 
            slist.append(coeff(m + 1, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, **kwargs) )

    # Loop
    while np.abs(slist[-1] / Slow / m) > 0.01 * accuracy:    
        m += 1
        S += slist[-1]
        Slow += slist[-1]  * (-1) ** (m - 1) / m 

        # Add a new entry to list if the accuracy is < 1% 
        slist.append(coeff(m + 1, nu, galaxy, rad_temp, hyp, ratio, D0, delta, B, manual, **kwargs) )

    # Save file once the desired accuracy is achieved     
    with open(os.path.join(cache_path, filename), mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        for row in slist:
            writer.writerow([row])
    return m


# ########################### Classes (for v2)


def RA_rad(galaxy):
    return 2 * np.pi * (
        float(gen_data['R.A. (J2000)'][galaxy][:2]) + (
            float(gen_data['R.A. (J2000)'][galaxy][3:5]) + float(gen_data['R.A. (J2000)'][galaxy][6:]) / 60 
        ) / 60 
    ) / 24

def Dec_rad(galaxy):
    return np.pi * float(gen_data['Decl. (J2000)'][galaxy][0] + '1') * (
        float(gen_data['Decl. (J2000)'][galaxy][1:3]) + (
            float(gen_data['Decl. (J2000)'][galaxy][4:6]) + float(gen_data['Decl. (J2000)'][galaxy][7:]) / 60 
        ) / 60 
    ) / 180

class transport:
    def __init__(self, rh = None, B = None, D0 = None, tau0 = None, delta = None):
        if rh:
            self._rh = rh
            if (D0 is not None) and (tau0 is not None):
                raise Exception('ERROR: rh, D0 and tau0 are interdependent. Cannot set them all at the same time.')
            elif D0:
                D0_conv = D0 * cm_to_kpc **2 * Gyr_to_sec
                self._D0 = D0
                self._tau0 = self.rh ** 2 / D0_conv
            elif tau0:
                tau0sec = tau0 * Gyr_to_sec
                rhcm = rh / cm_to_kpc
                self._tau0 = tau0
                self._D0 = rhcm ** 2 / tau0sec
        elif rh is None:
            if (D0 is not None) and (tau0 is not None):
                tau0sec = tau0
                D0_conv = D0 * cm_to_kpc **2 * Gyr_to_sec
        
                self._D0 = D0
                self._tau0 = tau0
                self._rh = np.sqrt(tau0sec * D0_conv)
        self.B = B
        self.delta = delta
            
        
    @property
    def D0(self):
        return self._D0
    
    @property
    def tau0(self):
        return self._tau0
    
    @property
    def rh(self):
        return self._rh

    @D0.setter
    def D0(self, value):
        self._D0 = value
        
    @tau0.setter
    def tau0(self, value):
        self._tau0 = value
    
    @rh.setter
    def rh(self, value):
        self._rh = value

# Functions

    def Elosses(self,E):
        """
        Total energy loss function in GeV/s
    
        :param E: cosmic-ray energy in GeV
        :param B: magnitude of the magnetic field’s smooth component in :math:`\\mu`\G
    
        :return: energy-loss rate in GeV/s
        """
        return b0 * (1 + (self.B / Bc)**2) * E**2
    
    def Dcoeff(self, E):
        """
        Diffusion coefficient in cm :math:`{}^2` /s
    
        :param E: cosmic-ray energy in GeV
        :param delta: power-law exponent of the diffusion coefficient as a function of the CRE's energy (default value = 1/3 or ``'kol'``)
    
        :return: Diffusion coefficient for CRE with energy :math:`E` (GeV) in cm :math:`{}^2` /s
        """
        return self.D0 * E ** delta_float(self.delta)
    
    def Syrovatskii_var(self, E):
        """
        Syrovatskii variable in kpc\ :sup:`2`
    
        :param E: cosmic-ray energy in GeV
        :param B: magnitude of the magnetic field’s smooth component in :math:`\\mu`\G
        :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm\ :sup:`2`/s
        :param delta: power-law exponent of the diffusion coefficient as a function of the CRE’s energy (default value = 1/3) 
    
        :return: Syrovatskii variable in kpc\ :sup:`2`
        """
        _delta = delta_float(self.delta)
        return self.D0 / b0 / (1 + (self.B / Bc)**2) / (1 - _delta) / E**(1 - _delta) * cm_to_kpc**2
    
    def eta_var(self, E):
        """
        :math:`\\eta` variable as a function of the CRE's energy, magnetic field, tau and delta parameters 
    
        :param E: CRE energy in GeV
        :param B: magnetic field strength in µG
        :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm\ :sup:`2`/s
        :param delta: power-law exponent of the diffusion coefficient as a function of the CRE’s energy
    
        :return: :math:`\\eta` variable
        """
        return np.pi ** 2 * self.Syrovatskii_var(E) / self.rh ** 2
    
    
    def hatXne(self, E, E0):
        """
        CRE number-density function kernel in s/GeV :math:`\\hat X_n`

        :param E: CRE energy in GeV
        :param E0: injected CRE's energy in GeV
        :param B: magnetic field strength in µG
        :param D0: magnitude of the diffusion coefficient for a 1 GeV CRE in cm\ :sup:`2`/s
        :param delta: power-law exponent of the diffusion coefficient as a function of the CRE’s energy
            
        :return: Electron number density kernel in s/GeV 
        """
        if E > E0:
            return 0
        return 2 * np.exp(-(self.eta_var(E) - self.eta_var(E0))) / self.Elosses(E)
    
    def _Elosses_ics(self,E):
        return b0 * E**2
    def _Elosses_synchrotron(self,E):
        return b0 * (self.B / Bc)**2 * E**2

    def _timescale_losses(self,E):
        return E / self.Elosses(E) / Gyr_to_sec
    def _timescale_diffusion(self,E):
        return self.rh ** 2 / self.Dcoeff(E) / cm_to_kpc**2 / Gyr_to_sec

    def _timescale_ics(self,E):
        return E / self._Elosses_ics(E) / Gyr_to_sec
    def _timescale_synchrotron(self,E):
        return E / self._Elosses_synchrotron(E) / Gyr_to_sec