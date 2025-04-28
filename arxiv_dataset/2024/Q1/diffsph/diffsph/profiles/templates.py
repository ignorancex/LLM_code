from diffsph.utils.consts import *

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Catalogue of R(r) functions %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Dark Matter halos

def hdz(r, rs, rhos, alpha, beta, gamma):
    """
    Hernquist/Diemand/Zhao dark-matter halo template.
    
    .. math::
        \\rho(r) = \\frac{\\rho_s}{(r/r_s)^\\gamma(1+(r/r_s)^\\alpha)^{\\frac{\\beta-\\gamma}\\alpha}}
        
    Using default values alpha = 1, beta = 3 and gamma = 1 results in the default NFW halo profile.

    :param r: main variable (galactocentric distance)
    :param rs: scale radius 
    :param rhos: chraracteristic density
    :param alpha: inner exponent
    :param beta: large-r exponent
    :param gamma: small-r exponent
        
    :return: density at galactocentric distance r
    """
    return rhos / (r/rs)**gamma / (1 + (r/rs)**alpha)**((beta-gamma)/alpha)


def nfw(r, rs, rhos):
    """
    Navarro/Frenk/White dark-matter halo template.
    
    .. math::
        \\rho(r) = \\frac{\\rho_s}{(r/r_s)(1+r/r_s)^2}
    
    :param r: main variable (galactocentric distance)
    :param rs: scale radius 
    :param rhos: chraracteristic density

    :return: density at galactocentric distance :math:`r`
    """
    return hdz(r, rs, rhos, 1, 3, 1)

def cnfw(r, rs, rhos, rc):
    """
    Cored Navarro/Frenk/White dark-matter halo template.        

    .. math::
        \\rho(r) = \\frac{\\rho_s}{(r/r_s+r_c/r_s)(1+r/r_s)^2}
        
    :param r: main variable (galactocentric distance)
    :param rs: scale radius 
    :param rhos: chraracteristic density
    :param rc: core radius

    :return: density at galactocentric distance :math:`r`
    """
    return rhos / ( rc/rs + r/rs * (1 + r/rs)**2)


def sis(r, sigmav):
    """
    Singular isothermal sphere
    
    .. math::
        \\rho(r) = \\frac{\\sigma_v^2}{2\\pi G r^2}    
        
    :param r: main variable (galactocentric distance)
    :param sigmav: velocity dispersion

    :return: density at galactocentric distance :math:`r`
    """
    return sigmav**2 / 2 / np.pi / GNw / r**2


def enst(r, rs, rhos, alphaE=.17):
    """
    Einasto dark-matter halo profile.
    
    .. math::
        \\rho(r) = \\rho_s\\exp\\left[-\\frac{2}{\\alpha_E}\\left(\\frac{r^{\\alpha_E}}{r_s^{\\alpha_E}}-1\\right)\\right]

    :param r: main variable (galactocentric distance)
    :param rs: scale radius
    :param rhos: charactieristic density
    :param alphaE: power-law slope of the Einasto profile, (default value = 0.17)

    :return: density at galactocentric distance :math:`r`
    """
    return rhos * np.exp(-2 * ((r/rs)**alphaE - 1) / alphaE)


def bkrt(r, rs, rhos):
    """
    Burkert dark-matter halo profile.
    
    .. math::
        \\rho(r) = \\frac{\\rho_s}{(1+r/r_s)(1+r^2/r_s^2)}

    :param r: main variable (galactocentric distance)
    :param rs: scale radius
    :param rhos: charactieristic density

    :return: density at galactocentric distance :math:`r`
    """
    return rhos / (1 + r/rs) / (1 + (r/rs)**2)


def ps_iso(r, rs, rhos):
    """
    Pseudo-isothermal sphere dark-matter halo profile.
     
    .. math::
        \\rho(r) = \\frac{\\rho_s}{1+r^2/r_s^2}
        
    :param r: main variable (galactocentric distance)
    :param rs: scale radius
    :param rhos: characteristic density

    :return: density at galactocentric distance :math:`r`

    """
    return hdz(r, rs, rhos, 2, 2, 0)


# Generic radial shapes

def ps(r, rs):
    '''
    Point source template
    
    .. math::
        \\rho(r) = \\frac1{4\\pi r^2}\\delta(r)
    
    :param r: main variable (galactocentric distance)
    :param rs: characteristic radius

    :return: density at galactocentric distance :math:`r`

    '''
    return np.exp( - r**2 / 2 / rs**2) / (2 * np.pi * rs**2)**(3 / 2)


def const(r, rs):
    """
    Constant (top-hat) template
    
    .. math::
        \\rho(r) = \\frac3{4\\pi r_s^3}\\Theta(r_s-r)
        
    :param rs: characteristic radius

    :return: constant density
    """
    return 3 / 4 / np.pi / rs**3


def plmm(r, rs):
    """
    Plummer template
    
    .. math::
        \\rho(r) = \\frac3{4\\pi r_s^3}\\frac1{(1+r^2/r_s^2)^{5/2}}

    :param r: main variable (distance to the center)
    :param rs: Plummer radius
    :param rhoa: central density

    :return: density of the Plummer sphere at distance :math:`r`
    """
    return hdz(r, rs, 3 / 4 / np.pi / rs**3, 2, 5, 0)

# Dictionary


rad_temp_dict = {
    'HDZ': {'func': hdz, 'args': ('alpha', 'beta', 'gamma'), 
            'names': ('HDZ', 'Hernquist', 'Diemand', 'Zhao', 'gen NFW', 'gNFW', 'genNFW')},
    'NFW': {'func': nfw, 'args': (), 
            'names': ('NFW', 'Navarro', 'Frenk', 'White', 'Navarro-Frenk-White', 'nfw')},
    'cNFW': {'func': cnfw, 'args': ('rc',), 
             'names': ('cored NFW', 'cNFW', 'CNFW')},
    'sis': {'func': sis, 'args':'sigmav', 'names': ('sis', 'SIS', 'singular isothermal sphere', 'isothermal sphere')},
    'Enst': {'func': enst, 'args': ('alphaE',), 'names': ('Enst', 'Einasto')},
    'Bkrt': {'func': bkrt, 'args': (), 'names': ('Bkrt', 'Burkert')},
    'ps_iso': {'func': ps_iso, 'args': (), 'names':('ps_iso', 'pseudo isothermal sphere')},
    'ps': {'func': ps, 'args':'rs', 'names': ('ps', 'point', 'point source', 'point-source')},
    'const': {'func':const, 'args':'rs', 'names': ('c', 'const', 'constant density', 'constant')},
    'plmm': {'func': plmm, 'args': 'rs', 'names': ('Plummer', 'plmm')}
    # add more halo_mod mappings as needed
}
