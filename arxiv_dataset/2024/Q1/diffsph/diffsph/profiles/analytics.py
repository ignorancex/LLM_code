import numpy as np
from scipy import special as sp
from scipy.integrate import quad

from diffsph.utils.consts import GNw

#  Point sources

def psbrA(theta, rs, rh, dist):
    '''
    Brigthness H-function for point sources in the regime-A approximation
    
    :param theta: angular radius in rad
    :param rs: scale radius
    :param rh: diffusion radius parameter
    :param dist: distance to the source
    '''
    return np.exp( - dist**2 * np.sin(theta)**2 / 2 / rs**2 ) / (2 * np.pi * rs**2) * (1 - sp.gammaincc(
        1/2, (rh**2 - dist**2 * np.sin(theta)**2) / 2 / rs**2 
    ) ) 

def psbrC(t, rh):
    '''
    Brigthness H-function for point sources in the regime-C approximation
    Variable t is defined as 

    :param t: :math:`D\\sin(\\theta)/r_h`, where :math:`\\theta` (``theta``), :math:`r_h` (``rh``) and :math:`D` (``dist``) are defined below
    :param theta: angular radius in rad  
    :param rh: diffusion radius parameter
    :param dist: distance to the source
    '''
    return np.pi / 2 / rh ** 2 * (np.log((1 + np.sqrt(1 - t**2)) / t) - np.sqrt(1 - t**2))

def psfdC(theta, rh, dist):
    '''
    Flux-density H-function for point sources in the regime-C approximation
    
    :param theta: angular radius in rad
    :param rh: diffusion radius parameter
    :param dist: distance to the source
    '''
    return np.pi**2 / 2 / dist / rh**3 * (
                        - dist * rh - (dist**2 + rh**2) * np.log(
                            ( dist * np.cos(theta) + np.sqrt( rh**2 - dist**2 * np.sin(theta)**2 ) ) / ( dist + rh )
                        ) + dist * rh * np.log(
                            ( 1 - rh**2 / dist**2 ) *
                                ( rh * np.cos(theta) + np.sqrt( rh**2 - dist**2 * np.sin(theta)**2 ) ) / 
                                 ( rh * np.cos(theta) - np.sqrt( rh**2 - dist**2 * np.sin(theta)**2 ) ) 
                            ) + dist * np.cos(theta) * (
                             np.sqrt( rh**2 - dist**2 * np.sin(theta)**2 ) - 2 * rh * np.log(
                                  ( rh + np.sqrt(rh**2 - dist**2 * np.sin(theta)**2 ) ) / dist / np.sin(theta) 
                             )
                        ) 
                    )
def psfdCmax(rh, dist):
    '''
    Maximum value for the flux-density H-function for point sources in the regime-C approximation
    
    :param rh: diffusion radius parameter
    :param dist: distance to the source
    '''
    return np.pi**2 / 2 / rh**2 * ( np.log( 1 - rh**2 / dist **2 ) - ( rh / dist + dist / rh ) / 2 * np.log(( dist - rh ) / (dist + rh )) - 1
                                  )
#  Constant (Top hat)

def cobrA(t, rs, rh):
    '''
    Brightness H-function for the 'constant' top-hat source in the regime-A approximation

    :param t: :math:`D\\sin(\\theta)/r_h`, where :math:`\\theta` (``theta``), :math:`r_h` (``rh``) and :math:`D` (``dist``) are defined below
    :param theta: angular radius in rad
    :param rs: scale radius
    :param rh: diffusion radius parameter
    :param dist: distance to the source
    '''
    return 3 * rh / 2 / np.pi / rs**3 * np.sqrt(1 - t**2)

def cobrC(t, rs, rh):
    '''
    Brightness H-function for the 'constant' source in the regime-C approximation

    :param t: :math:`D\\sin(\\theta)/r_h`, where :math:`\\theta` (``theta``), :math:`r_h` (``rh``) and :math:`D` (``dist``) are defined below
    :param theta: angular radius in rad
    :param rs: scale radius
    :param rh: diffusion radius parameter
    '''
    return rh * np.pi / 6 / rs**3  * np.sqrt(1 - t**2) ** 3

def cofdA(theta, rs, rh, dist):
    '''
    Flux-density H-function for the 'constant' top-hat source in the regime-A approximation
    
    :param theta: angular radius in rad
    :param rs: scale radius
    :param rh: diffusion radius parameter
    :param dist: distance to the source
    '''
    return 3 / 2 / rs**3 * (rh - np.sqrt( rh**2 - dist**2 * np.sin(theta)**2 ) * np.cos(theta) + dist * ( 1 - rh**2 / dist**2 ) * np.log(( dist * np.cos(theta) + np.sqrt( rh**2 - dist**2 * np.sin(theta)**2 ) ) / ( dist + rh )))

def cofdAmax(rs, rh, dist):
    '''
    Maximum value for the flux-density H-function for the 'constant' top-hat source in the regime-A approximation
    
    :param rs: scale radius
    :param rh: diffusion radius parameter
    :param dist: distance to the source
    '''
    return 3 / 2/ rs**3 * ( rh + dist * ( 1 - rh**2 / dist **2 ) / 2 * np.log( ( dist - rh ) / (dist + rh )))

def cofdC(theta, rs, rh, dist):
    '''
    Flux-density H-function for the 'constant' top-hat source in the regime-C approximation
    
    :param theta: angular radius in rad
    :param rs: scale radius        
    :param rh: diffusion radius parameter
    :param dist: distance to the source
    '''
    return np.pi**2 / 3 / rs ** 3 *  rh * ( 
                        ( 5 / 8 - 3 * dist**2 / 8 / rh**2 ) * 
                        ( 1 - np.cos(theta) * np.sqrt( rh**2 - dist**2 * np.sin(theta)**2 ) )
                        + 3 / 8 * ( dist**2 - rh**2 )**2 / dist / rh**3 * np.log(
                                  ( dist * np.cos(theta) - np.sqrt( rh**2 - dist**2 * np.sin(theta)**2 ) ) / ( dist - rh )
                             ) 
                        + dist**2 / 4 / rh**3 * np.sin(theta)**2 * np.cos(theta) * np.sqrt( rh**2 - dist**2 * np.sin(theta)**2 )
                    )

def cofdCmax(rs, rh, dist):
    '''
    Maximum value for the flux-density H-function for the 'constant' top-hat source in the regime-C approximation
    
    :param rs: scale radius
    :param rh: diffusion radius parameter
    :param dist: distance to the source
    '''
    return np.pi**2 * rh / 24 / rs**3 * ( 
                        5 - 3 * (dist / rh)**2 + 3 * ( dist**2 - rh**2 )**2 / 2 / dist / rh**3 * np.log( 
                            ( dist + rh ) / (dist - rh ) 
                        )    
                    )

# Singular isothermal

def sisbrA(t, sigmav, rh):
    '''
    Brightness H-function for the singular isothermal source in the regime-A approximation

    :param t: :math:`D\\sin(\\theta)/r_h`, where :math:`\\theta` (``theta``), :math:`r_h` (``rh``) and :math:`D` (``dist``) are defined below
    :param theta: angular radius in rad 
    :param sigmav: velocity dispersion parameter
    :param rh: diffusion radius parameter
    :param dist: distance to the source    
    '''
    return sigmav ** 2 / np.pi / GNw / rh * np.arctan( np.sqrt(1 - t**2 ) / t) / t

def sisbrC(t, sigmav, rh):
    '''
    Brightness H-function for the singular isothermal  source in the regime-C approximation

    :param t: :math:`D\\sin(\\theta)/r_h`, where :math:`\\theta` (``theta``), :math:`r_h` (``rh``) and :math:`D` (``dist``) are defined below
    :param theta: angular radius in rad
    :param sigmav: velocity dispersion parameter
    :param rh: diffusion radius parameter
    :param dist: distance to the source
    '''
    return np.pi * sigmav ** 2 / GNw / rh * (np.sqrt(1 - t**2 ) + t * ( np.arctan( t / np.sqrt(1 - t**2 ) ) - np.pi/2 ) )