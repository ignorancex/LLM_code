from diffsph.profiles.templates import *
from diffsph.profiles.hfactors import *

import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# foo = pd.read_csv('./galaxies/foo.csv', index_col='Name') # Other Ref. / Halo model 

tab0 = pd.read_csv(os.path.join(dir_path, 'data/sat_dsphs.csv')).drop(columns = 'arXiv')
tab1 = pd.read_csv(os.path.join(dir_path, 'data/DMhalos_dsphs.csv'))

gen_data = tab0.set_index('Name')
dm_data = pd.merge(tab0, tab1, on = ['Name','Abbr']).set_index(['Name','Halo Model', 'arXiv'])

# DM mass density using specific mass models 

def rho(r, rad_temp, manual = False, **kwargs):
    """
    Dark matter density
        
    :param r: galactocentric distance
    :param rad_temp: template ('NFW', 'Einasto', etc.)
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``)         
        
    Keyword arguments
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``

    :param rs: scale radius 
    :param rhos: characteristic density
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile       
    
    :return: dark matter density
    """
    func = rad_temp_dict[var_to_str(rad_temp)]['func']
    
    if not manual:
        halo_mod = var_to_str(rad_temp)
        gal = var_to_str(kwargs['galaxy'])
        arxiv = var_to_str(kwargs['ref'])
        
        if (gal, halo_mod, arxiv) not in dm_data.index:
            raise KeyError('Galaxy/halo model/reference combination not found. Please check your spelling.')
        
        rs = dm_data['rs [kpc]'][gal, halo_mod, arxiv]
        rhos = dm_data['rhos [GeV/cm3]'][gal, halo_mod, arxiv]
 
        args = {arg: dm_data[arg][gal, halo_mod, arxiv] for arg in rad_temp_dict[halo_mod]['args']}
        return func(r, rs, rhos, **args)
    args = {arg: kwargs[arg] for arg in rad_temp_dict[var_to_str(rad_temp)]['args'] + ('rs', 'rhos')}
    return func(r, **args)

# ###########################################################################

# Halo factors using specific mass models 

def h(n, galaxy, rad_temp, hyp, ratio, manual = False, **kwargs):
    """
    Model-specific n-th halo factor
        
    :param n: order of the halo/bulge factor
    :param rh: diffusion halo/bulge radius
    :param rad_temp: radial template 
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and the half-light radius
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``)         
        
    Keyword arguments
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``

    :param rs: scale radius 
    :param rhos: characteristic density
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile  
    
    :return: halo factor
    """
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    if not manual:
        ref = kwargs['ref']
        if (var_to_str(galaxy), var_to_str(rad_temp), var_to_str(ref)) not in dm_data.index:
            raise KeyError('Galaxy/halo model/reference combination not found. Please check your spelling.')
        def rad_func(r): 
            return rho(r, var_to_str(rad_temp), galaxy = var_to_str(galaxy), ref = var_to_str(ref))
        return halo_factor(n, rh, hyp, rad_func)
    return halo_factor(n, rh, hyp, var_to_str(rad_temp), **kwargs)

# H functions using specific mass models 

def Hem(r, galaxy, rad_temp, hyp, ratio, regime = 'B', manual = False, **kwargs):
    """
    Model-specific emissivity halo/bulge function in the Regime "A", "B" or "C" approximations
        
    :param r: galactocentric distance in kpc
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and the half-light radius (default value = 1) 
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``)         
        
    Keyword arguments
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``

    :param rs: scale radius 
    :param rhos: characteristic density
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile  
    
    :return: emissivity halo/bulge function
    """
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
        
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    
    if not manual:
        ref = kwargs['ref']
        if (var_to_str(galaxy), var_to_str(rad_temp), var_to_str(ref)) not in dm_data.index:
            raise KeyError('Galaxy/halo model/reference combination not found. Please check your spelling.')
    
        def rad_func(r): 
            return rho(r, var_to_str(rad_temp), galaxy = var_to_str(galaxy), ref = var_to_str(ref))

        return  H_emissivity(r, rh, hyp, rad_func, regime)
    
    return H_emissivity(r, rh, hyp, var_to_str(rad_temp), regime, **kwargs)

# ###########################################################################


#  Brightness H functions using specific mass models

def Hbr(tharcmin, galaxy, rad_temp, hyp, ratio, regime = 'B', manual = False, **kwargs):
    """
    Model-specific brightness halo/bulge function in the Regime "A", "B" or "C" approximations
        
    :param tharcmin: angular radius in arcmin
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and the half-light radius (default value = 1) 
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``)         
        
    Keyword arguments
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``

    :param rs: scale radius 
    :param rhos: characteristic density
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile  
    
    :return: brightness halo/bulge function
    """
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    
    if not manual:
        ref = kwargs['ref']
        if (var_to_str(galaxy), var_to_str(rad_temp), var_to_str(ref)) not in dm_data.index:
            raise KeyError('Galaxy/halo model/reference combination not found. Please check your spelling.')
        
        def rad_func(r): 
            return rho(r, var_to_str(rad_temp), galaxy = var_to_str(galaxy), ref = var_to_str(ref))

        return  H_brightness(tharcmin / rad_to_arcmin, dist, rh, hyp, rad_func, regime)
    
    return H_brightness(tharcmin / rad_to_arcmin, dist, rh, hyp, var_to_str(rad_temp), regime, **kwargs)


# ###########################################################################
          
# # # Flux density H functions using specific mass models

def Hfd(tharcmin, galaxy, rad_temp, hyp, ratio, regime = 'B', manual = False, **kwargs):
    """
    Model-specific flux-density halo/bulge function in the Regime "A", "B" or "C" approximations
       
    :param tharcmin: angular radius in arcmin
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``
    :param ratio: ratio between the diffusion halo/bulge and the half-light radius (default value = 1) 
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``)                 
        
    Keyword arguments
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``

    :param rs: scale radius 
    :param rhos: characteristic density
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile  
    
    :return: brightness halo/bulge function
    """
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    
    rh = ratio * gen_data['rhalf [kpc]'][var_to_str(galaxy)]
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    
    if not manual:
        ref = kwargs['ref']
        if (var_to_str(galaxy), var_to_str(rad_temp), var_to_str(ref)) not in dm_data.index:
            raise KeyError('Galaxy/halo model/reference combination not found. Please check your spelling.')
        
        def rad_func(r): 
            return rho(r, var_to_str(rad_temp), galaxy = var_to_str(galaxy), ref = var_to_str(ref))

        return  H_fluxdens(tharcmin / rad_to_arcmin, dist, rh, hyp, rad_func, regime)
    
    return H_fluxdens(tharcmin / rad_to_arcmin, dist, rh, hyp, var_to_str(rad_temp), regime, **kwargs)


# # # J- and D- factors


def J(tharcmin, galaxy, rad_temp, manual = False, **kwargs):
    """
    Model-specific J factor in Gev :math:`{}^2`/cm :math:`{}^5`
        
    :param tharcmin: angular radius in arcmin
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``)         
                
    Keyword arguments
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``

    :param rs: scale radius 
    :param rhos: characteristic density
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile  

    :return: J factor
    """
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    
    if not manual:
        ref = kwargs['ref']
        if (var_to_str(galaxy), var_to_str(rad_temp), var_to_str(ref)) not in dm_data.index:
            raise KeyError('Galaxy/halo model/reference combination not found. Please check your spelling.')
        
        def rad_func(r): 
            return rho(r, var_to_str(rad_temp), galaxy = var_to_str(galaxy), ref = var_to_str(ref))
        return  J_factor(tharcmin / rad_to_arcmin , dist, rad_func)
    
    return  J_factor(tharcmin / rad_to_arcmin , dist, var_to_str(rad_temp), **kwargs)


def D(tharcmin, galaxy, rad_temp, manual = False, **kwargs):
    """
    Model-specific D factor in GeV/cm :math:`{}^2`
        
    :param tharcmin: angular radius in arcmin
    :param str galaxy: name of the galaxy 
    :param str rad_temp: radial template (``'NFW'``, ``'Einasto'``, etc.)
    :param bool manual: manual input of parameter values in rad_temp (default value = ``'False'``)         
                
    Keyword arguments
        
    -  ``manual = 'False'``
    
    :param ref: reference used (``'Martinez'`` or ``'1309.2641'``, ``'Geringer-Sameth'`` or ``'1408.0002'``, etc.)
    
    -  ``manual = 'True'``

    :param rs: scale radius 
    :param rhos: characteristic density
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile  

    :return: D factor
    """
    if var_to_str(galaxy) not in gen_data.index:
        raise KeyError('Galaxy not found. Please check your spelling or use the function add_dwarf to add new data.')
    
    dist = gen_data['Dist [kpc]'][var_to_str(galaxy)]
    
    if not manual:
        ref = kwargs['ref']
        if (var_to_str(galaxy), var_to_str(rad_temp), var_to_str(ref)) not in dm_data.index:
            raise KeyError('Galaxy/halo model/reference combination not found. Please check your spelling.')
        
        def rad_func(r): 
            return rho(r, var_to_str(rad_temp), galaxy = var_to_str(galaxy), ref = var_to_str(ref))
        return  D_factor(tharcmin / rad_to_arcmin , dist, rad_func)
    
    return  D_factor(tharcmin / rad_to_arcmin , dist, var_to_str(rad_temp), **kwargs)
