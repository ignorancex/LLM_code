#### FUNCTIONS FOR MINIMUM BH MASS CALCULATION ###

######################################
## Imports and definitions
import numpy as np
from scipy.optimize import fsolve

from astropy import constants as const
from matplotlib import ticker, cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches


##########################################################################################
# 0. Qcrit values
##########################################################################################
def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)


def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)



##############################
# Part 1 of zeta_RL
def part1(q, beta):
    """    
    dln(a)/dln m_d
    note q == m_donor/m_accretor
    """
    return -2 * (1 - beta*q - (1 - beta) * (q + 1./2.) * (q / (q + 1) )  )
##############################
# Part 2 of zeta_RL
def part2(q):
    """
    dln(Rl/a)/dlnq
    note q == m_donor/m_accretor
    """
    A = (q**(1./3.)) / 3.
    B = 2./(q**(1./3.))
    C = (  1.2 * q**(1./3.) +  1./(1. + q**(1./3)) ) / (0.6 * q**(2./3.) + np.log(1 + q**(1./3.) ) )
    return A * (B - C)
##############################
# Part 3 of zeta_RL
def part3(q, beta):
    """
    dln(q)/dln m
    note q == m_donor/m_accretor
    """
    return 1 + beta*q
##############################
# Zeta RL
def zeta_rl(q, beta):
    return part1(q, beta) + (part2(q) * part3(q, beta))
    


##############################
# Find Qcrit from zeta
##############################
def get_qcrit_from_zeta(Beta, Zeta_eff):
    """
    solve for the critical mass ratio, given an effective zeta and zeta_RL
    !!   note q == m_donor/m_accretor  !!
    q     ----------> [(list of) floats] m_donor/m_accretor
    qcrit2     ----------> [float] stability mass ratio Q_{<mt2} = M_2/MBH < qcrit2
    fcore1     ----------> [float] the core mass fraction of star 1 (Mcore1/Mstar1)
    beta       ----------> [float] the mas accretion fraction M_acc = beta M_don
    asn,bsn    ----------> [float] fit parameters for 'supernova' mass loss dMsn = a*Mcore + b (asusming Mcore = fcore*Mtams)
    """
    
    def zeta_HG_is_zeta_rl(q, *ARG):
        """ ARG will contain beta and zeta 
        this function solves zeta_rl - zeta_eff = 0"""
        beta,  zeta_eff = ARG 
        return zeta_rl(q, beta) - zeta_eff

    # fsolve does not like to receive arrays (4 is just my guess everywhere)
    qcrit1 = []
    qcrit2 = []
    
    # Check if zeta is an array
    if not isinstance(Zeta_eff, float):
        print('Zeta_eff is not a float')
        for zeta in Zeta_eff:
            # Mass transfer 1
            Qcrit1        = 1./fsolve(zeta_HG_is_zeta_rl, 4., args = (Beta, zeta))[0]
            # Mass transfer 2 (Edd. lim)
            Qcrit2        = fsolve(zeta_HG_is_zeta_rl, 4., args = (0, zeta))[0]
                      
            #Add to array
            qcrit1.append(Qcrit1)
            qcrit2.append(Qcrit2)
            
    # Check if Beta is an array
    elif not isinstance(Beta, float):
        print('beta is not a float')
        for b in Beta:
            # Mass transfer 1
            Qcrit1        = 1./fsolve(zeta_HG_is_zeta_rl, 4., args = (b, Zeta_eff))[0]
            # Mass transfer 2 (Edd. lim)
            Qcrit2        = fsolve(zeta_HG_is_zeta_rl, 4., args = (0, Zeta_eff))[0]
            
            #Add to array
            qcrit1.append(Qcrit1)
            qcrit2.append(Qcrit2)
            
    # Assume both zeta and zeta are a float
    else:
        print('both beta and zeta are floats')
        # Mass transfer 1
        qcrit1            = 1./fsolve(zeta_HG_is_zeta_rl, 4., args = (Beta, Zeta_eff))[0]
        # Mass transfer 2 (Edd. lim)
        qcrit2            = fsolve(zeta_HG_is_zeta_rl, 4., args = (0, Zeta_eff))[0]

    return np.c_[(qcrit1, qcrit2)]


# q_crits = get_qcrit_from_zeta(Beta = 1.0 , Zeta_eff = np.arange(5,6,0.1)) 
# print('first set of qcrits: ', q_crits[0], 'first mass transfer: ', q_crits[:,0], 'reverse mass transfer: ', q_crits[:,1])


##########################################################################################
# 1. dMsn and fcore
##########################################################################################


##############################
# 'supernova' mass loss 
# diff between Mcore and M_BH
##############################
def dMsn(Mcore, asn = -0.9, bsn = 13.9, csn =0, mthresh = 14.8):
    """
    Mcore      ----------> [list of floats] core mass, typically assumed as fcore*Mtams
    asn,bsn    ----------> [float] fit parameters for 'supernova' mass loss dMsn = a*Mcore + b (asusming Mcore = fcore*Mtams)
    mthresh    ----------> [float] Threshold mass for full fallback (i.e. no SN mass loss)
    """    
    # basic Msn loss
    dMsn    = csn*Mcore**2 + asn*Mcore + bsn
    
    # correct for where the core is higher than the threshold for full fallback
    full_fb       = Mcore > mthresh
    
    dMsn[full_fb] = 0 # full fb = no mass loss
    
    return dMsn


##############################
# Core mass fraction function
##############################
def fcore(Mtams,  a_f = 0.0026, b_f= 0.247):
    """
    Mtams      ----------> [list of floats] mass at TAMS = MZAMS1 or approx \tilde{M}_2
    a_f,b_f    ----------> [float] fit parameters for f_core = a_f*M_tams + b_f 
    """
    # core mass fraction
    f_core    = a_f*Mtams + b_f
    
    return f_core



##########################################################################################
# 2. Min M_zams
##########################################################################################

##############################
# minimum M_ZAMS(1) through method A
##############################
def minMzams1_dMsn(Q_ZAMS, qcrit2 = 4.6, beta = 1.0, fc1 = 0.4, asn = -0.9, bsn= 13.9 ):
    """
    Method A: assuming dMsn is a function of M_zams, while keeping f_core constant
    Q_ZAMS     ----------> [(list of) floats] zams mass ratios M_2/M_1
    qcrit2     ----------> [float] stability mass ratio Q_{<mt2} = M_2/MBH < qcrit2
    fcore1     ----------> [float] the core mass fraction of star 1 (Mcore1/Mstar1)
    beta       ----------> [float] the mas accretion fraction M_acc = beta M_don
    asn,bsn    ----------> [float] fit parameters for 'supernova' mass loss dMsn = a*Mcore + b (asusming Mcore = fcore*Mtams)
    """
    
    minMzams = (bsn* qcrit2)/( qcrit2*fc1*(1 - asn) - beta*(1 - fc1)  - Q_ZAMS)
    
    return minMzams

##############################
# minimum M_ZAMS(1) through method A
##############################
def minMzams1_dMsn_quadratic(Q_ZAMS, qcrit2 = 4.6, beta = 1.0, fc1 = 0.4, asn = -0.1, bsn= 1.5, csn = -2.14 ):
    """
    Method A: assuming dMsn is a quadratic function of M_zams, while keeping f_core constant
    Q_ZAMS     ----------> [(list of) floats] zams mass ratios M_2/M_1
    qcrit2     ----------> [float] stability mass ratio Q_{<mt2} = M_2/MBH < qcrit2
    fcore1     ----------> [float] the core mass fraction of star 1 (Mcore1/Mstar1)
    beta       ----------> [float] the mas accretion fraction M_acc = beta M_don
    asn,bsn    ----------> [float] fit parameters for 'supernova' mass loss dMsn = a*Mcore + b (asusming Mcore = fcore*Mtams)
    """
    
    A     = asn * fc1**2
    B     = csn
    C     = (bsn - 1)*fc1 + ((Q_ZAMS + (1 - fc1)*beta )/qcrit2 )
    
    
    minMzams = (np.sqrt(C**2 - 4*A*B) - C)/(2 * A)
    
    return minMzams



##############################
# minimum M_ZAMS(1) through method A
##############################
def minMzams1_fcore(Q_ZAMS, qcrit2 = 4.6, beta = 1.0, a_f = 0.0026, b_f= 0.247 ):
    """
    Method B: assuming f_core is a function of M_zams
    Q_ZAMS     ----------> [(list of) floats] zams mass ratios M_2/M_1
    qcrit2     ----------> [float] stability mass ratio Q_{<mt2} = M_2/MBH < qcrit2
    beta       ----------> [float] the mas accretion fraction M_acc = beta M_don
    a_f,b_f    ----------> [float] fit parameters for f_core = a_f*M_tams + b_f 
    """
    minMzams = 1./a_f * ( (Q_ZAMS + beta - b_f*(qcrit2 + beta) ) /(qcrit2 + beta) )

#     minMzams = 1./a_f * ( (Q_ZAMS + beta)/(qcrit2 + beta) -b_f )
    
    return minMzams




##########################################################################################
# 3. Min MBHs
##########################################################################################

##############################
# minimum MBH(1)
##############################
def get_analyticalMBH1(use_zeta = False, zeta = 6.5, qcrit1 = 0.25, qcrit2 = 4.575, Beta = 1., 
                       Fc1 = 0.33, M_threshold = 14.8,
                       use_dMsn = True, A_sn = -0.9, B_sn = 13.9, C_sn = 0.,  A_f = 0.0026, B_f = 0.247, verbose = False): 
    """
    This 
    use_zeta    ----------> [Bool] If you want to use zeta to determine qcrit1 and qcrit2
    zeta        ----------> [(list of) float] stability mass ratio Q_{<mt2} = M_2/MBH < qcrit2
    qcrit1      ----------> [(list of) float] stability mass ratio Q_{<mt1} is also zams mass ratios M_2/M_1
    qcrit2      ----------> [(list of) float] stability mass ratio Q_{<mt2} = M_2/MBH < qcrit2
    Beta        ----------> [(list of) float] the mas accretion fraction M_acc = beta M_don (for MT1)
    Fc1         ----------> [float] the core mass fraction of star 1 (Mcore1/Mstar1)
    M_threshold ----------> [float] Threshold mass for full fallback (i.e. no SN mass loss)
    use_dMsn    ----------> [Bool] If you want to use method (A), with dMsn(Mzams), ot method (B) where fcore(Mzams)
    A_sn,B_sn   ----------> [float,float] fit parameters for 'supernova' mass loss dMsn = a*Mcore + b (asusming Mcore = fcore*Mtams)
    A_f,B_f     ----------> [float,float] fit parameters for f_core = a_f*M_tams + b_f 
    
    """
    ########################
    # Calculate qcrit1 and 2 based on zeta
    if use_zeta:
        if verbose: print('We will ignore the supplied qcrit1 =%s, qcrit2 =%s, and recaclulate them based on zeta =%s '%(qcrit1, qcrit2,zeta ))
        q_crits = get_qcrit_from_zeta(Beta = 1.0 , Zeta_eff = zeta) 
    else:
        if np.logical_and(not isinstance(qcrit1, float), isinstance(qcrit2, float) ):
            if verbose: print('len(qcrit1)=%s, while qcrit2=%s, I assume you want to repeat qcrit2 for every qcrit1'%(len(qcrit1),qcrit2) )
            qcrit2 = np.repeat(qcrit2, len(qcrit1))
        q_crits = np.c_[(qcrit1, qcrit2)]
    if verbose: print('q_crits', q_crits)
    
    
    ########################
    # Using method (A): fc = const, and dMsn(fc*Mzams)
    if use_dMsn: 
        print('using method A, fc = const, dMsn(M)')
        Minzams1 = minMzams1_dMsn(q_crits[:,0], qcrit2 = q_crits[:,1], beta = Beta, fc1 = Fc1,  asn = A_sn, bsn= B_sn)

    ########################
    # Using method (B): fc(Mzams1), and dMsn=0
    else:
        print('using method B, fc(M), dMsn = 0')
        Minzams1 = minMzams1_fcore(q_crits[:,0], qcrit2 = q_crits[:,1], beta = Beta, a_f = A_f, b_f = B_f)
        Fc1      = fcore(Minzams1,  a_f = A_f, b_f= B_f)
        
    if verbose: print('\n Minzams1', Minzams1, 'Fc1', Fc1)
    
    
    ########################
    # The assumed core mass of M1
    M_core1 =  Fc1 * Minzams1  
    
    # dMsn hase some treshold of full fallback, after which it is 0
    #dMsn1   = dMsn(M_core1, asn = A_sn, bsn = B_sn, mthresh = M_threshold)
    dMsn1   = dMsn(M_core1, asn = A_sn, bsn = B_sn, csn = C_sn, mthresh = M_threshold)
    
    # final (minimmum) BH mass1
    M_BH1   = M_core1 - dMsn1
    
    return M_BH1, q_crits



##############################
# minimum MBH(2)
##############################
def get_analyticalMBH2(minM_BH1 = None, qcrits = None, use_zeta = False, Zeta = 6.5, qcrit1 = 0.25, qcrit2 = 4.575, Beta = 1.,
                       Fc1 = 0.33, Fc2 = 0.33, M_thresh = 14.8,
                       use_dMsn = True, A_sn = -0.9, B_sn = 13.9, A_f = 0.0026, B_f = 0.247):
    #C1 = -0.364, C2 = 16.104):
    """
    adopting minMzams1_dSN
    Q_ZAMS ----------> zams mass ratios M_2/M_1
    qcrit2   ----------> stability mass ratio Q_{<mt2} = M_2/MBH < qcrit2
    beta   ----------> beta is the mas accretion M_acc = beta M_don
    a,b    ----------> fit parameters for f_core = a * M_ZAMS1 + b (see fit below)
    
    """
    if minM_BH1 is None:
        # Get min value for MBH1, and calc qcrits if zeta is used
        minM_BH1, qcrits  = get_analyticalMBH1(use_zeta = use_zeta, zeta = Zeta, qcrit1 = qcrit1, qcrit2 = qcrit2, Beta = Beta, 
                           Fc1 = Fc1, M_threshold = M_thresh, use_dMsn = use_dMsn, A_sn = A_sn, B_sn = B_sn, A_f = A_f, B_f = B_f)
    else:
        print('using predefined MBH1 and qcrits')

    # The minimum value of tilde{M}_2, from eq. 1 (q_crits[:,1] = updated version of qcrit2)
    min_Mtilde_2     = qcrits[:,1] * minM_BH1
    
    # Recalculate the core mass fractions if you use method B
    if not use_dMsn:
        Fc2      = fcore(min_Mtilde_2,  a_f = A_f, b_f= B_f)
        print('\n making fcore Fc2', Fc2)

    # Minimum core mass2 that comes out of min_Mtilde_2
    minM_core2       = Fc2 * min_Mtilde_2
    
    # dMsn hase some treshold of full fallback, after which it is 0
    dMsn2   = dMsn(minM_core2, asn = A_sn, bsn = B_sn, mthresh = M_thresh)
    
    #################
    # final minimum, including SN mass loss
    minM_BH2         = minM_core2 - dMsn2
    
    return minM_BH2

