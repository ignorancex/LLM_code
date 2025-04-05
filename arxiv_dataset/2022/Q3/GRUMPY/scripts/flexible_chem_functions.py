import numpy as np

def ZIGM_floor(time):
    '''
    Function that defines the IGM floor (in linear units, not [M/H] units) as a function of time (Gyr)
    '''
    
    #convert time to redshift here
    zred = time
    A = -99
    B = -99
    Z_IGM = A + B*zred
    return Z_IGM


def nsm_flexible_yields():
    '''
    Function where use can flexibly define NSM yields. Yields are given as total ejecta mass per event.
    In stochastic case, 
    '''

    #this would be the number of the 




def ccsne_flexible_yields():
    '''
    Function where use can flexibly define CC-SNE yields. Yields can be either defined as a function of 
    stellar mass and metallicity (which is then later integrated with IMF to get SSP yield) or can be 
    given as total amount ejected by SSP per Msun. In the later scenario, the assumption is that the yield will
    not depend on the stellar mass and will just contribute to overall normalization of SSP yield.
    '''
    return



def agb_flexible_yields():
    '''
    Function where use can flexibly define AGB yields. Yields can be either defined as a function of 
    stellar mass and metallicity (which is then later integrated with IMF to get SSP yield) or can be 
    given as total amount ejected by SSP per Msun. In the later scenario, the assumption is that the yield will
    not depend on the stellar mass and will just contribute to overall normalization of SSP yield.
    '''
    return


def sn1a_flexible_yields():
    '''
    Function where use can flexibly define SNe Ia yields. Yields can be either defined as a function of 
    stellar mass and metallicity (which is then later integrated with IMF to get SSP yield) or can be 
    given as total amount ejected by SSP per Msun. In the later scenario, the assumption is that the yield will
    not depend on the stellar mass and will just contribute to overall normalization of SSP yield.

    One can extend this flexibility even to normalization, but that is degenrate with yields so that flexibility
    is implicit in here.
    '''
    #approximation for Sanders et al. SNe Ia yields
    # m = 0.31431361784399675
    # b = 0.24288017091128827
    # feh*m + b 
    return

