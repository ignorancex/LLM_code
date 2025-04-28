from diffsph.utils.consts import *
from diffsph.utils.tools import *
from diffsph.profiles.templates import *
from diffsph.profiles.analytics import *

import pandas as pd
import numpy as np
import scipy.integrate as integrate
import scipy.special as sp

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Halo coefficients h_n() %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%

# Generic case

def halo_factor(n, rh, hyp, rad_temp, **kwargs):
    """
    n-th order halo/bulge factor h_n for a given model (e.g. NFW, Einasto, Plummer, ...) 
    Arguments ``'n'``, ``'rh'``, ``'hyp'`` and ``'rad_temp'`` are necessary. Remaining arguments depend on the 
    adopted halo model. 
        
    :param n: order of the halo/bulge factor
    :param rh: diffusion halo/bulge radius
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``) 
    :param rad_temp: radial template 
    
    Keyword arguments

    :param rs: scale radius 
    :param rhos: characteristic density
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
    
    :return: halo/bulge factor
    """
    k = hypothesis_index(hyp)
    
    # Point Source (analytic result)
    if type(rad_temp) == str and rad_temp.lower() in ['ps', 'point', 'point source', 'point-source']:
        return n / 2 / rh**2
    # Constant density (analytic result)
    elif type(rad_temp) == str and rad_temp.lower() in ['c', 'const', 'constant density']:
        if rh <= kwargs['rs']:
            return 3 * rh / 2 / np.pi ** 2 / kwargs['rs']**3 * (-1) ** (n - 1) / n
        raise KeyError('Parameter rs is not larger than rh')
    # singular isothermal sphere (analytic result)
    elif type(rad_temp) == str and rad_temp.lower() in ['sis', 'singular isothermal sphere', 'isothermal sphere']:
        if k == 1:
            return cm_to_kpc**(-1) * kwargs['sigmav'] ** 2  / rh / np.pi / GNw * sp.sici( n * np.pi)[0]
        raise KeyError('Halo template only valid for DM decay')
    else:
        return  2 / rh * integrate.quad(lambda r: evaluate(rad_temp, r, **kwargs)**k  * r * np.sin(n * np.pi * r / rh), 0, rh)[0] / cm_to_kpc ** (hyp.lower() in all_names['wimp'] + all_names['decay'])
    
# %%%%%%%%%%%%%%%%%%%%%%
# % Regimes A, B and C %
# %%%%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%%
# Emissivity H functions%
# %%%%%%%%%%%%%%%%%%%%%%%

def Hem_A(r, rh, hyp, rad_temp, **kwargs):
    """
    Generic emissivity halo/bulge function for Regime A

    :param r: galactocentric distance
    :param rh: diffusion halo/bulge radius
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``)
    :param rad_temp: radial template
    
    
    Keyword arguments

    :param rs: scale radius
    :param rhos: characteristic density 
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile

    :return: emissivity halo/bulge function using the Regime-A approximation
    """
    k = hypothesis_index(hyp)
    
    if 0. < r < rh:
        
        # Point Source (integral can be solved analytically)
        if type(rad_temp) == str and rad_temp.lower() in ['ps', 'point', 'point source', 'point-source']:
            return ps(r, rs=kwargs['rs'])     
        elif type(rad_temp) == str and rad_temp.lower() in ['c', 'const', 'constant density']:
            if kwargs['rs'] >= rh:
                return 3 / 4 / np.pi / kwargs['rs']**3
            raise KeyError('Parameter rs is not larger than rh')            
        elif type(rad_temp) == str and rad_temp.lower() == 'sis':
            if k != 1:
                raise KeyError('Halo template only valid for DM decay')
            return sis(r, kwargs['sigmav'])
        else: 
            return evaluate(rad_temp, r, **kwargs)**k 
    else:
        return 0.


def Hem_B(r, rh, hyp, rad_temp, **kwargs):
    """
    Generic emissivity halo/bulge function for Regime B
           
    :param r: galactocentric distance
    :param rh: diffusion halo/bulge radius
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``)
    :param rad_temp: radial template
        
    Keyword arguments

    :param rs: scale radius
    :param rhos: characteristic density 
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile

    :return: emissivity halo/bulge function using the Regime-B approximation
    """
    if 0. < r < rh:
        if type(rad_temp) == str and rad_temp.lower() in ['c', 'const', 'constant density']:
            if kwargs['rs'] <= rh:
                raise KeyError('Parameter rs is not larger than rh')
        return cm_to_kpc ** (hyp.lower() in all_names['wimp'] + all_names['decay']) * halo_factor(n=1, rh=rh, hyp=hyp, rad_temp=rad_temp, **kwargs) * np.sin(np.pi * r / rh) / r
    else:
        return 0.


def Hem_C(r, rh, hyp, rad_temp, **kwargs):
    """
    Generic emissivity halo/bulge function for Regime C
           
    :param r: galactocentric distance
    :param rh: diffusion halo/bulge radius
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``)
    :param rad_temp: radial template
    
    Keyword arguments

    :param rs: scale radius
    :param rhos: characteristic density 
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile

    :return: emissivity halo/bulge function using the Regime-C approximation
    """
    k = hypothesis_index(hyp)
    
    if 0. < r < rh:     
        if type(rad_temp) == str and rad_temp.lower() in ['ps', 'point', 'point source', 'point-source']:
            return np.pi / 4 / rh**3 * (rh / r - 1)
        elif type(rad_temp) == str and rad_temp.lower() in ['c', 'const', 'constant density']:
            if kwargs['rs'] >= rh:
                return np.pi / 8 / kwargs['rs']**3 * (1 - r**2 / rh**2)
            raise KeyError('Parameter rs is not larger than rh')            
        elif type(rad_temp) == str and rad_temp.lower() == 'sis':
            if k != 1:
                raise KeyError('Halo template only valid for DM decay')
            return np.pi * kwargs['sigmav']**2 / 2 / GNw / rh**2 * np.log( rh / r)
        else: 
            return np.pi**2 / rh**2 * ( integrate.quad(
                lambda rp: (rp / r) * evaluate(rad_temp, rp, **kwargs)**k * (1/2 * (r + rp) - 1/2 * (r - rp) - r * rp / rh), 0, r
            )[0] + integrate.quad(
                lambda rp: (rp / r) * evaluate(rad_temp, rp, **kwargs)**k * (1/2 * (r + rp) - 1/2 * (rp - r) - r * rp/rh), r, rh
            )[0])
    return 0.


def H_emissivity(r, rh, hyp, rad_temp, regime, **kwargs):
    """
    Generic emissivity halo/bulge function in the Regime "A", "B" or "C" approximations

    :param r: galactocentric distance
    :param rh: diffusion halo/bulge radius
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``)
    :param rad_temp: radial template
    :param regime: regime of the approximation (upper/lower case a, b, c or I/II/III).
               
    Keyword arguments

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
    if (regime not in ['A', 'B', 'C']) and (regime not in ['a', 'b', 'c']) and (regime not in ['I', 'II', 'III']):
        raise KeyError('Regime not found. Please use upper/lower letters or I/II/III to define regimes')

    
    if regime in ['A', 'a', 'I']:
        return Hem_A(r=r, rh=rh, hyp=hyp, rad_temp=rad_temp, **kwargs)
    if regime in ['B', 'b', 'II']:
        return Hem_B(r=r, rh=rh, hyp=hyp, rad_temp=rad_temp, **kwargs)
    if regime in ['C', 'c', 'III']:
        return Hem_C(r=r, rh=rh, hyp=hyp, rad_temp=rad_temp, **kwargs)


# %%%%%%%%%%%%%%%%%%%%%%%%
#  Brightness H functions%
# %%%%%%%%%%%%%%%%%%%%%%%%

def H_brightness(theta, dist, rh, hyp, rad_temp, regime, **kwargs):
    """
    Generic emissivity halo/bulge function in the Regime "A", "B" or "C" approximations

    :param theta: angular distance in rad units
    :param dist: distance to earth
    :param rh: diffusion halo/bulge radius
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``)        
    :param halo_model: DM halo model
    :param rad_temp: radial template
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    
    Keyword arguments

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
    if (regime not in ['A', 'B', 'C']) and (regime not in ['a', 'b', 'c']) and (regime not in ['I', 'II', 'III']):
        raise KeyError('Regime not found. Please use upper/lower letters or I/II/III to define regimes')

    if 0. <= theta < np.arcsin(rh / dist):
        t = dist * np.sin(theta) / rh
        temp = 0
        if type(rad_temp) == str and rad_temp.lower() in ['ps', 'point', 'point source', 'point-source']:
            if regime in ['A', 'a', 'I']:
                ra = kwargs['rs']
                return psbrA(theta, kwargs['rs'], rh, dist)
                
            elif regime in ['C', 'c', 'III']:
                return psbrC(t, rh)
                
        elif type(rad_temp) == str and rad_temp.lower() in ['c', 'const', 'constant density']:
            if kwargs['rs'] <= rh:
                raise KeyError('Parameter rs is not larger than rh')
            
            if regime in ['A', 'a', 'I']:
                return cobrA(t, kwargs['rs'], rh)
                
            elif regime in ['C', 'c', 'III']:
                return cobrC(t, kwargs['rs'], rh)
                
        elif type(rad_temp) == str and rad_temp.lower() == 'sis':
            if regime in ['A', 'a', 'I']:
                return cm_to_kpc ** (-1) * sisbrA(t, kwargs['sigmav'], rh)
                
            elif regime in ['C', 'c', 'III']:
                return cm_to_kpc ** (-1) * sisbrC(t, kwargs['sigmav'], rh)

        if regime in ['B', 'b', 'II']:
            return halo_factor(n=1, rh=rh, hyp=hyp, rad_temp=rad_temp, **kwargs) * f(n=1, x=t)
        
        else:
            return 2 * integrate.quad(
                    lambda l: H_emissivity(
                        r=np.sqrt(l**2 + dist**2 * np.sin(theta)**2), rh=rh, hyp=hyp, rad_temp=rad_temp,
                        regime=regime, **kwargs
                    ), 0, np.sqrt( rh**2 - dist**2 * np.sin(theta)**2 )
                )[0] / cm_to_kpc ** (hyp.lower() in all_names['wimp'] + all_names['decay'])
            
    else:
        return 0.

# %%%%%%%%%%%%%%%%%%%%%%%%%
# Flux density H functions%
# %%%%%%%%%%%%%%%%%%%%%%%%%

def H_fluxdens(theta, dist, rh, hyp, rad_temp, regime, **kwargs):
    """
    Generic flux-density halo/bulge function in the Regime "A", "B" or "C" approximations

    :param theta: angular distance in rad units
    :param dist: distance to earth
    :param rh: diffusion halo/bulge radius
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``)
    :param halo_model: DM halo model
    :param rad_temp: radial template
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.

    
    Keyword arguments

    :param rs: scale radius 
    :param rhos: characteristic density 
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile

    :return: flux density halo/bulge function
    """
    if (regime not in ['A', 'B', 'C']) and (regime not in ['a', 'b', 'c']) and (regime not in ['I', 'II', 'III']):
        raise KeyError('Regime not found. Please use upper/lower letters or I/II/III to define regimes')

    if theta <= 0.:
        return 0.
    
    if type(rad_temp) == str and rad_temp.lower() in ['ps', 'point', 'point source', 'point-source']:
        if regime in ['A', 'a', 'I']:
            return 1 / dist ** 2
            
        elif regime in ['C', 'c', 'III']:
            if 0. < theta <= np.arcsin(rh / dist):
                return psfdC(theta, rh, dist)
                
            elif theta > np.arcsin(rh / dist):
                return psfdCmax(rh, dist) 
                
    elif type(rad_temp) == str and rad_temp.lower() in ['c', 'const', 'constant density']:
        if kwargs['rs'] <= rh:
            raise KeyError('Parameter rs is not larger than rh')
                
        if regime in ['A', 'a', 'I']:
            if 0. < theta <= np.arcsin(rh / dist):
                return cofdA(theta, kwargs['rs'], rh, dist)
                  
            elif theta > np.arcsin(rh / dist):
                return cofdAmax(kwargs['rs'], rh, dist)
                    
        elif regime in ['C', 'c', 'III']:
            if 0. < theta <= np.arcsin(rh / dist):
                return cofdC(theta, kwargs['rs'], rh, dist)
                    
            elif theta > np.arcsin(rh / dist):
                return cofdCmax(kwargs['rs'], rh, dist)
            
    if regime in ['B', 'b', 'II']:
        return halo_factor(n=1, rh=rh, hyp=hyp, rad_temp=rad_temp, **kwargs) * halo_fd(n=1, theta=theta, dist=dist, rh=rh)
                                                          
    else:
        if 0. < theta <= np.arcsin(rh / dist):
            return 2 * np.pi * ( 
                integrate.quad(
                    lambda r: 2 * r * H_emissivity(
                        r=r, rh=rh, hyp=hyp, rad_temp=rad_temp, regime=regime, **kwargs
                    ) * ker_0(r, dist), 0, rh
                )[0] - integrate.quad(
                    lambda r: 2 * r * H_emissivity(
                        r=r, rh=rh, hyp=hyp, rad_temp=rad_temp, regime=regime, **kwargs
                    ) * ker_1(r, theta, dist), dist * np.sin(theta), rh
                )[0]
            ) / cm_to_kpc ** (hyp.lower() in all_names['wimp'] + all_names['decay'])
        
        elif theta > np.arcsin(rh / dist):
            return 2 * np.pi * integrate.quad(
                lambda r: 2 * r * H_emissivity(
                    r=r, rh=rh, hyp=hyp, rad_temp=rad_temp, regime=regime, **kwargs
                ) * ker_0(r, dist), 0, rh
            )[0] / cm_to_kpc ** (hyp.lower() in all_names['wimp'] + all_names['decay'])
        

def J_factor(theta, dist, rad_temp, **kwargs):
    '''
    Generic "J" factor
    
    :param theta: angular distance in rad units
    :param dist: distance to earth
    :param rad_temp: radial template
    
    Keyword arguments

    :param rs: scale radius
    :param rhos: characteristic density
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile

    :return: J factor
    '''
    return H_fluxdens(theta=theta, dist=dist, rh=dist, rad_temp=rad_temp, hyp = 'dm ann', regime = 'a', **kwargs)

def D_factor(theta, dist, rad_temp, **kwargs):
    '''
    Generic "D" factor

    :param theta: angular distance in rad units
    :param dist: distance to earth
    :param rad_temp: radial template
    
    Keyword arguments

    :param rs: scale radius
    :param rhos: characteristic density
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile
        
    :return: D factor
    '''
    return H_fluxdens(theta=theta, dist=dist, rh=dist, rad_temp=rad_temp, hyp = 'dm dec', regime = 'a', **kwargs)


# %%%%%%%%%%%%%%%%%%%%%%%%%
    # Approximate formula %
# %%%%%%%%%%%%%%%%%%%%%%%%%

# Useful for checking 

def H_fluxdens_approx(theta, dist, rh, hyp, rad_temp, regime, **kwargs):
    """
    Generic flux-density halo/bulge function in the Regime "A", "B" or "C" approximations (alternative formula)

    :param theta: angular distance in rad units
    :param dist: distance to earth
    :param rh: diffusion halo/bulge radius
    :param str hyp: hypothesis: ``'wimp'`` (**default**), ``'decay'`` or ``'generic'``)
    :param halo_model: DM halo model
    :param rad_temp: radial template
    :param regime: regime of the approximation. Must be either upper or lower case a, b, c or I/II/III.
    
    Keyword arguments

    :param rs: scale radius
    :param rhos: characteristic density
    :param alpha: exponent :math:`\\alpha` in the :py:func:`diffsph.profiles.templates.hdz` profile  
    :param beta: exponent :math:`\\beta` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param gamma: exponent :math:`\\gamma` in the :py:func:`diffsph.profiles.templates.hdz` profile 
    :param alphaE: parameter :math:`\\alpha_E` in the :py:func:`diffsph.profiles.templates.enst` profile
    :param rc: core radius parameter :math:`r_c` in the :py:func:`diffsph.profiles.templates.cnfw` profile
    :param sigmav: velocity dispersion parameter :math:`\\sigma_v` in the :py:func:`diffsph.profiles.templates.sis` profile

    :return: flux density halo/bulge function
    """
    if (regime not in ['A', 'B', 'C']) and (regime not in ['a', 'b', 'c']) and (regime not in ['I', 'II', 'III']):
        raise KeyError('Regime not found. Please use upper/lower letters or I/II/III to define regimes')
    if type(rad_temp) == str and rad_temp.lower() in ['c', 'const', 'constant density']:   
        if kwargs['rs'] <= rh:
            raise KeyError('Parameter rs is not larger than rh')

    if theta <= 0.:
        return 0.

    if regime in ['B', 'b', 'II']:
        if 0. < theta <= np.arcsin(rh / dist):
            return halo_factor(n=1, rh=rh, hyp=hyp, rad_temp=rad_temp, **kwargs) *  approxhalo_fd(n=1, theta=theta, dist=dist, rh=rh)
        
        elif theta > np.arcsin(rh / dist):
            return halo_factor(n=1, rh=rh, hyp=hyp, rad_temp=rad_temp, **kwargs) * approxhalo_fd_tot(n=1, dist=dist, rh=rh)
    
    else:
        temp = 0
        if 0. < theta <= np.arcsin(rh / dist):
            temp = 4 * np.pi * ( 
                integrate.quad(
                    lambda r: r**2 * H_emissivity(
                        r=r, rh=rh, hyp=hyp, rad_temp=rad_temp, regime=regime, **kwargs
                    ) , 0, rh
                )[0] - integrate.quad(
                    lambda r: r * np.sqrt( r**2 - dist **2 * np.sin(theta)**2 ) * H_emissivity(
                        r=r, rh=rh, hyp=hyp, rad_temp=rad_temp, regime=regime, **kwargs
                    ) , dist * np.sin(theta), rh
                )[0]
            ) / dist ** 2 
        
        elif theta > np.arcsin(rh / dist):
            temp = 4 * np.pi * integrate.quad(
                lambda r: r **2 * H_emissivity(
                    r=r, rh=rh, hyp=hyp, rad_temp=rad_temp, regime=regime, **kwargs
                ), 0, rh
            )[0] / dist ** 2
        if hyp.lower() in ann_names + dec_names:
            return cm_to_kpc ** (-1) * temp
        return temp