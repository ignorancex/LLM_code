#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    bisp_logll:
    -----------
        Definition of the log-likelihood functions to perform the line-of-sight
        decomposition and starlight polarization signal along distance.
        
        Also includes convenient functions to wrap the log-L function for
        smooth usage and functions to handle and manipulate
        prior_transform functions to be used by the ``dynesty``.

        The log-likelihood functions relate to the layer models developed
        in Pelgrims+ 2022, A&A, ... ...
        
        Models implement 1, 2, 3, 4, or 5 dusty and magnetized layers along
        distance.
        6 parameters describe each layers:
            - plx: the parallax of the cloud (1./distance) [mas]
            - q: the q=Q/I Stokes parameter of the cloud   [fraction]
            - u: the u=U/I Stokes parameter of the cloud   [fraction]
            - Cqq: the qq element of the covariance matrix describing
                    the intrinsic scatter (as due to turbulence)
                    [fraction^2]
            - Cuu: the uu element of the covariance matrix describing
                    the intrinsic scatter (as due to turbulence)
                    [fraction^2]
            - Cqu: the qu element of the covariance matrix describing
                    the intrinsic scatter (as due to turbulence)
                    [fraction^2]
        #

        The likelihood accounts for observational uncertainties in both
        stellar parallax (assumed to be Gaussian errors) and stellar
        polarization and explicitely account for the intrinsic scatter
        in polarization properties expected from turbulence in the ISM.
        
        Measurements of parallaxes and polarization are uncorrelated as
        they are obtained from different experiments.
        
@author: V.Pelgrims

"""

# # # Imports:
import numpy as np
from scipy import special
from scipy import stats


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Wrapping the log-likelihood function
def def_model(ModelParam,ModelName,StarData,*args):
    '''
        Wraps the log-likelihood function of the chosen model,
        given its name, and pass it the star data.
    '''
    
    if ModelName == '1Layer':
        def model_log_likelihood(ModelParam):
            return logLL_OneLayer(ModelParam,StarData[2],StarData[0],StarData[1],\
                                    StarData[6],StarData[3],StarData[4],StarData[5])
        #
    elif ModelName == '2Layers':
        def model_log_likelihood(ModelParam):
            return logLL_TwoLayers(ModelParam,StarData[2],StarData[0],StarData[1],\
                                    StarData[6],StarData[3],StarData[4],StarData[5])
        #
    elif ModelName == '3Layers':
        def model_log_likelihood(ModelParam):
            return logLL_ThreeLayers(ModelParam,StarData[2],StarData[0],StarData[1],\
                                    StarData[6],StarData[3],StarData[4],StarData[5])
        #
    elif ModelName == '4Layers':
        def model_log_likelihood(ModelParam):
            return logLL_FourLayers(ModelParam,StarData[2],StarData[0],StarData[1],\
                                    StarData[6],StarData[3],StarData[4],StarData[5])
        #
    elif ModelName == '5Layers':
        def model_log_likelihood(ModelParam):
            return logLL_FiveLayers(ModelParam,StarData[2],StarData[0],StarData[1],\
                                    StarData[6],StarData[3],StarData[4],StarData[5])
        #
    else:
        message = 'The model name does not point to any implemented model.'
        raise ValueError(message)
        #
    #
    return model_log_likelihood(ModelParam)


# Definition of the prior_transform functions according to specified
# type and values for each parameters.
def def_prior_IntSc(u,PriorType,ParamList,StarData):
    '''
        Defines the appropriate prior_transform functions to be used by
        ``dynesty`` from lists of prior types ('Flat' or 'Gauss')
        and list of values to specify the forms of the prior functions.
        
        See the documentation if BISP_1.Priors for the format.
    '''

    Nclouds = len(PriorType)
    p = np.copy(u)

    for nn in range(Nclouds):
        for pp in range(6):
            if PriorType[nn][pp] == 'Gauss':
                # special care for the parallax
                if (pp == 0 and \
                    (ParamList[nn][pp] <= ParamList[nn][pp+6])):
                    message = ('the mean is <= than the stddev while it',\
                                'cannot be negative. You should not do that.',\
                                    'Aborded.')
                    raise ValueError(message)
                else:
                    p[6*nn+pp] = ParamList[nn][pp] +\
                                    (ParamList[nn][pp+6]*\
                                        stats.norm.ppf(u[6*nn+pp]))
                #
            elif PriorType[nn][pp] == 'Flat':
                p[6*nn+pp] = (ParamList[nn][pp+6]-ParamList[nn][pp])*\
                                u[6*nn+pp] + ParamList[nn][pp]
                #
                if (pp == 5):
                    # we actually want Cqu**2 < Cqq * Cuu
                    v_crit = np.sqrt(p[6*nn+pp-2] * p[6*nn+pp-1])
                    p[6*nn+pp] = 2 * v_crit * u[6*nn+pp] - v_crit
                #
                if ((nn >= 1) and (pp == 0)):
                    if PriorType[nn-1][pp] == 'Flat':
                        v_crit = np.minimum(np.sort(StarData[2][StarData[2]<p[6*(nn-1)+pp]])[-5],\
                                            ParamList[nn][pp])
                        p[6*nn+pp] = (ParamList[nn][pp+6]-v_crit)*\
                                    u[6*nn+pp] + v_crit
                    else:
                        v_crit = ParamList[nn-1][pp]-2*ParamList[nn-1][pp+6]
                        p[6*nn+pp] = (ParamList[nn][pp+6]-v_crit)*\
                                    u[6*nn+pp] + v_crit
                        #p[6*nn+pp] = (ParamList[nn][pp+6]-ParamList[nn][pp])*\
                        #            u[6*nn+pp] + ParamList[nn][pp]
                #
            else:
                raise ValueError('oups: Wrong PriorType')
            
        #
    return p


'''
###############################################################################
#
# Log-Likelihood functions: define the log-likelihood of a model with N clouds
#
###############################################################################
'''


'''
###############################################################################
                                1 layer model
###############################################################################
'''

def logLL_OneLayer(theta,star_plx,star_q,star_u,err_plx,err_q,err_u,c_qu):
    '''
        Implementation of the 1-layer model.
        Input:
        - theta: includes the 6 parameters for the layer
        - star_plx: parallaxes of the stars in [mas]
        - star_q: q=Q/I Stokes parameter of the star [fraction]
        - star_u: u=U/I Stokes parameter of the star [fraction]
        - err_...: corresponding errors
        - c_qu: uncertainty correlation between q and u from observation
                [fraction^2]
        Output:
        logll: the log-likelihood estimated for the model with parameter
                values in theta and for the input data
    '''

    plx1,q1,u1,Cint_qq,Cint_uu,Cint_qu = theta
    #
    if Cint_qu**2 >= Cint_qq * Cint_uu:
        return -np.inf
    #
    q_m_b = 0*star_q
    u_m_b = 0*star_q
    q_m_f = 0*star_q
    u_m_f = 0*star_q
    Cqq = err_q**2 #+ Cint_qq
    Cuu = err_u**2 #+ Cint_uu
    Cqu = c_qu #+ Cint_qu
    #
    # probability of each star to be behind the cloud at plx1
    W_b = (1./2 + 1./2*special.erf((plx1 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in front of the cloud at plx1
    W_f = 1-W_b
    #  in this case the polarization remain unchanged
    #   and the covariance matrix remain unchanged too
    # handling the covariance matrix
    detC_f = Cqq * Cuu - Cqu**2
    invCqq_f = Cuu / detC_f
    invCuu_f = Cqq/detC_f
    invCqu_f = - Cqu/detC_f
    #
    # behind cloud1
    #  in this case the polarization get added the pol of the cloud:
    q_m_b += q1
    u_m_b += u1
    #
    # adding the intrinsic scatter term for stars background to the cloud
    Cqq_1 = Cqq + Cint_qq
    Cuu_1 = Cuu + Cint_uu
    Cqu_1 = Cqu + Cint_qu
    #
    detC_b = Cqq_1 * Cuu_1 - Cqu_1**2
    invCqq_b = Cuu_1 / detC_b
    invCuu_b = Cqq_1/detC_b
    invCqu_b = - Cqu_1/detC_b
    #
    # per-star likelihood of measuring their q,u given their plx,sig_plx
    # and the cloud parameters:
    # l_star = W_f * l(star_in_fg) + W_b * l(star_in_gb)
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    eta_q_f = star_q - q_m_f
    eta_u_f = star_u - u_m_f
    #
    chi2_b = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)
    chi2_f = (eta_q_f**2 * invCqq_f + eta_u_f**2 * invCuu_f +\
                    2 * eta_q_f * eta_u_f * invCqu_f)
    #
    l_i = W_f*np.exp(-.5*chi2_f)/(2*np.pi*np.sqrt(detC_f)) +\
            W_b*np.exp(-.5*chi2_b)/(2*np.pi*np.sqrt(detC_b))
    #
    l_i = l_i[l_i!=0]
    logll = np.sum(np.log(l_i))
    #
    if np.isnan(logll):
        logll = - np.inf
    #
    return logll

#

'''
################################################################################
                                2 layers models
################################################################################
'''

#
def logLL_TwoLayers(theta,star_plx,star_q,star_u,err_plx,err_q,err_u,c_qu):
    '''
        Implementation of the 2-layer model.
        Input:
        - theta: includes the 2 x 6 parameters for the layers
        - star_plx: parallaxes of the stars in [mas]
        - star_q: q=Q/I Stokes parameter of the star [fraction]
        - star_u: u=U/I Stokes parameter of the star [fraction]
        - err_...: corresponding errors
        - c_qu: uncertainty correlation between q and u from observation
                [fraction^2]
        Output:
        logll: the log-likelihood estimated for the model with parameter
                values in theta and for the input data
    '''

    plx1,q1,u1,C1int_qq,C1int_uu,C1int_qu,plx2,q2,u2,C2int_qq,C2int_uu,C2int_qu, = theta
    #
    if ((C1int_qu**2 >= C1int_qq * C1int_uu) or
            (C2int_qu**2 >= C2int_qq * C2int_uu)):
                #or (np.sum((star_plx<plx1)*(star_plx>plx2))<5.)):
        return -np.inf
    if (plx1<=plx2):
        return -np.inf
    #
    #
    q_m_b = 0*star_q
    u_m_b = 0*star_q
    q_m_f = 0*star_q
    u_m_f = 0*star_q
    Cqq = err_q**2
    Cuu = err_u**2
    Cqu = c_qu
    #
    
    # probability of each star to be in front of all cloud
    P_0 = (1./2 - 1./2*special.erf((plx1 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in between cloud1 and cloud2
    P_1 = (1./2*special.erf((plx1 - star_plx)/(np.sqrt(2)*err_plx)) -\
            1./2*special.erf((plx2 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in the background of cloud2
    P_2 = (1./2 + 1./2*special.erf((plx2 - star_plx)/(np.sqrt(2)*err_plx)))
        
    # 0. star in front of every clouds
    #  in this case the polarization remain unchanged
    #   and the covariance matrix remain unchanged too
    # handling the covariance matrix
    detC_f = Cqq * Cuu - Cqu**2
    invCqq_f = Cuu / detC_f
    invCuu_f = Cqq/detC_f
    invCqu_f = - Cqu/detC_f
    #
    eta_q_f = star_q - q_m_f
    eta_u_f = star_u - u_m_f
    #
    chi2_f = (eta_q_f**2 * invCqq_f + eta_u_f**2 * invCuu_f +\
                    2 * eta_q_f * eta_u_f * invCqu_f)

    
    # 1.    #   #   #   #   #   #   #   #
    #  behind cloud1
    #   in this case the polarization of cloud1 is added to the pol.
    q_m_b += q1
    u_m_b += u1
    #
    #   and the intrinsic scatter contribution of cloud1 is also added
    Cqq_1 = Cqq + C1int_qq
    Cuu_1 = Cuu + C1int_uu
    Cqu_1 = Cqu + C1int_qu
    #
    detC_b1 = Cqq_1 * Cuu_1 - Cqu_1**2
    invCqq_b = Cuu_1 / detC_b1
    invCuu_b = Cqq_1/detC_b1
    invCqu_b = - Cqu_1/detC_b1
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_1 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)
    # 2.
    #  behind cloud2
    #   in this case the polarization of cloud1 is added to the pol.
    q_m_b += q2
    u_m_b += u2
    #
    #   and the intrinsic scatter contribution of cloud2 is also added
    Cqq_2 = Cqq_1 + C2int_qq
    Cuu_2 = Cuu_1 + C2int_uu
    Cqu_2 = Cqu_1 + C2int_qu
    #
    detC_b2 = Cqq_2 * Cuu_2 - Cqu_2**2
    invCqq_b = Cuu_2 / detC_b2
    invCuu_b = Cqq_2 / detC_b2
    invCqu_b = - Cqu_2 / detC_b2
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_2 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)
    
    # # #
    # per-star likelihood of measuring their q,u given their plx,sig_plx
    # and the cloud parameters:
    #
    l_i = P_0 * np.exp(-.5*chi2_f)/(2 * np.pi * np.sqrt(detC_f)) +\
            P_1 * np.exp(-.5*chi2_1)/(2 * np.pi * np.sqrt(detC_b1)) +\
            P_2 * np.exp(-.5*chi2_2)/(2 * np.pi * np.sqrt(detC_b2))
    #
    l_i = l_i[l_i!=0]
    logll = np.sum(np.log(l_i))
    #
    if np.isnan(logll):
        logll = - np.inf
    #
    return logll
#


'''
################################################################################
                                3 layers models
################################################################################
'''

#
def logLL_ThreeLayers(theta,star_plx,star_q,star_u,err_plx,err_q,err_u,c_qu):
    '''
        Implementation of the 3-layer model.
        Input:
        - theta: includes the 3 x 6 parameters for the layers
        - star_plx: parallaxes of the stars in [mas]
        - star_q: q=Q/I Stokes parameter of the star [fraction]
        - star_u: u=U/I Stokes parameter of the star [fraction]
        - err_...: corresponding errors
        - c_qu: uncertainty correlation between q and u from observation
                [fraction^2]
        Output:
        logll: the log-likelihood estimated for the model with parameter
                values in theta and for the input data
    '''

    plx1,q1,u1,C1int_qq,C1int_uu,C1int_qu,\
        plx2,q2,u2,C2int_qq,C2int_uu,C2int_qu,\
            plx3,q3,u3,C3int_qq,C3int_uu,C3int_qu = theta
    #
    if ((C1int_qu**2 >= C1int_qq * C1int_uu) or
            (C2int_qu**2 >= C2int_qq * C2int_uu) or
                (C3int_qu**2 >= C3int_qq * C3int_uu)): # or
                    #(np.sum((star_plx<plx1)*(star_plx>plx2))<5.) or
                    #    (np.sum((star_plx<plx2)*(star_plx>plx3))<5.)):
        return -np.inf
    if ((plx1<=plx2) or (plx2<=plx3)):
        return -np.inf
    #
    #
    q_m_b = 0*star_q
    u_m_b = 0*star_q
    q_m_f = 0*star_q
    u_m_f = 0*star_q
    Cqq = err_q**2
    Cuu = err_u**2
    Cqu = c_qu
    #

    # probability of each star to be in front of all cloud
    P_0 = (1./2 - 1./2*special.erf((plx1 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in between cloud1 and cloud2
    P_1 = (1./2*special.erf((plx1 - star_plx)/(np.sqrt(2)*err_plx)) -\
            1./2*special.erf((plx2 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in between cloud2 and cloud3
    P_2 = (1./2*special.erf((plx2 - star_plx)/(np.sqrt(2)*err_plx)) -\
            1./2*special.erf((plx3 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in the background of cloud3
    P_3 = (1./2 + 1./2*special.erf((plx3 - star_plx)/(np.sqrt(2)*err_plx)))
        
    
    # 0. stars in front of every clouds
    #  in this case the polarization remain unchanged (=0)
    #   and the covariance matrix remain unchanged too, only observational
    # handling the covariance matrix
    detC_f = Cqq * Cuu - Cqu**2
    invCqq_f = Cuu / detC_f
    invCuu_f = Cqq/detC_f
    invCqu_f = - Cqu/detC_f
    #
    eta_q_f = star_q - q_m_f
    eta_u_f = star_u - u_m_f
    #
    chi2_f = (eta_q_f**2 * invCqq_f + eta_u_f**2 * invCuu_f +\
                    2 * eta_q_f * eta_u_f * invCqu_f)

    
    # 1.    #   #   #   #   #   #   #   #
    #  behind cloud1
    #   in this case the polarization of cloud1 is added to the pol.
    q_m_b += q1
    u_m_b += u1
    #
    #   and the intrinsic scatter contribution of cloud1 is also added
    Cqq_1 = Cqq + C1int_qq
    Cuu_1 = Cuu + C1int_uu
    Cqu_1 = Cqu + C1int_qu
    #
    detC_b1 = Cqq_1 * Cuu_1 - Cqu_1**2
    invCqq_b = Cuu_1 / detC_b1
    invCuu_b = Cqq_1/detC_b1
    invCqu_b = - Cqu_1/detC_b1
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_1 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)
    # 2.
    #  behind cloud2
    #   in this case the polarization of cloud2 is added to the pol.
    q_m_b += q2
    u_m_b += u2
    #
    #   and the intrinsic scatter contribution of cloud2 is also added
    Cqq_2 = Cqq_1 + C2int_qq
    Cuu_2 = Cuu_1 + C2int_uu
    Cqu_2 = Cqu_1 + C2int_qu
    #
    detC_b2 = Cqq_2 * Cuu_2 - Cqu_2**2
    invCqq_b = Cuu_2 / detC_b2
    invCuu_b = Cqq_2 / detC_b2
    invCqu_b = - Cqu_2 / detC_b2
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_2 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)
    
    # 3.
    #  behind cloud3
    #   in this case the polarization of cloud3 is added to the pol.
    q_m_b += q3
    u_m_b += u3
    #
    #   and the intrinsic scatter contribution of cloud3 is also added
    Cqq_3 = Cqq_2 + C3int_qq
    Cuu_3 = Cuu_2 + C3int_uu
    Cqu_3 = Cqu_2 + C3int_qu
    #
    detC_b3 = Cqq_3 * Cuu_3 - Cqu_3**2
    invCqq_b = Cuu_3 / detC_b3
    invCuu_b = Cqq_3 / detC_b3
    invCqu_b = - Cqu_3 / detC_b3
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_3 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)

    # # #
    # per-star likelihood of measuring their q,u given their plx,sig_plx
    # and the cloud parameters:
    #
    l_i = P_0 * np.exp(-.5*chi2_f)/(2 * np.pi * np.sqrt(detC_f)) +\
            P_1 * np.exp(-.5*chi2_1)/(2 * np.pi * np.sqrt(detC_b1)) +\
            P_2 * np.exp(-.5*chi2_2)/(2 * np.pi * np.sqrt(detC_b2)) +\
            P_3 * np.exp(-.5*chi2_3)/(2 * np.pi * np.sqrt(detC_b3))
    #
    l_i = l_i[l_i!=0]
    logll = np.sum(np.log(l_i))
    #
    if np.isnan(logll):
        logll = - np.inf
    #
    return logll
#


'''
################################################################################
                                4 layers models
################################################################################
'''

# # # CHECK THE IMPLEMENTATION HERE IT IS LIKELY NOT CORRECT YET!
def logLL_FourLayers(theta,star_plx,star_q,star_u,err_plx,err_q,err_u,c_qu):
    '''
        Implementation of the 4-layer model.
        Input:
        - theta: includes the 4 x 6 parameters for the layers
        - star_plx: parallaxes of the stars in [mas]
        - star_q: q=Q/I Stokes parameter of the star [fraction]
        - star_u: u=U/I Stokes parameter of the star [fraction]
        - err_...: corresponding errors
        - c_qu: uncertainty correlation between q and u from observation
                [fraction^2]
        Output:
        logll: the log-likelihood estimated for the model with parameter
                values in theta and for the input data
    '''

    plx1,q1,u1,C1int_qq,C1int_uu,C1int_qu,\
        plx2,q2,u2,C2int_qq,C2int_uu,C2int_qu,\
            plx3,q3,u3,C3int_qq,C3int_uu,C3int_qu,\
                plx4,q4,u4,C4int_qq,C4int_uu,C4int_qu = theta
    #
    if ((C1int_qu**2 >= C1int_qq * C1int_uu) or
            (C2int_qu**2 >= C2int_qq * C2int_uu) or
                (C3int_qu**2 >= C3int_qq * C3int_uu) or
                    (C4int_qu**2 >= C4int_qq * C4int_uu)):
        return -np.inf
    if ((plx1<=plx2) or (plx2<=plx3) or (plx3<=plx4)):
        return -np.inf
    #
    #
    q_m_b = 0*star_q
    u_m_b = 0*star_q
    q_m_f = 0*star_q
    u_m_f = 0*star_q
    Cqq = err_q**2
    Cuu = err_u**2
    Cqu = c_qu
    #
    
    # probability of each star to be in front of all cloud
    P_0 = (1./2 - 1./2*special.erf((plx1 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in between cloud1 and cloud2
    P_1 = (1./2*special.erf((plx1 - star_plx)/(np.sqrt(2)*err_plx)) -\
            1./2*special.erf((plx2 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in between cloud2 and cloud3
    P_2 = (1./2*special.erf((plx2 - star_plx)/(np.sqrt(2)*err_plx)) -\
            1./2*special.erf((plx3 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in between cloud3 and cloud4
    P_3 = (1./2*special.erf((plx3 - star_plx)/(np.sqrt(2)*err_plx)) -\
            1./2*special.erf((plx4 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in the background of cloud4
    P_4 = (1./2 + 1./2*special.erf((plx4 - star_plx)/(np.sqrt(2)*err_plx)))


    # 0. stars in front of all clouds
    #  in this case the polarization remain unchanged (=0)
    #   and the covariance matrix remain unchanged too, only observational
    # handling the covariance matrix
    detC_f = Cqq * Cuu - Cqu**2
    invCqq_f = Cuu / detC_f
    invCuu_f = Cqq/detC_f
    invCqu_f = - Cqu/detC_f
    #
    eta_q_f = star_q - q_m_f
    eta_u_f = star_u - u_m_f
    #
    chi2_f = (eta_q_f**2 * invCqq_f + eta_u_f**2 * invCuu_f +\
                    2 * eta_q_f * eta_u_f * invCqu_f)

    
    # 1.    #   #   #   #   #   #   #   #
    #  behind cloud1
    #   in this case the polarization of cloud1 is added to the pol.
    q_m_b += q1
    u_m_b += u1
    #
    #   and the intrinsic scatter contribution from cloud1 is also added
    Cqq_1 = Cqq + C1int_qq
    Cuu_1 = Cuu + C1int_uu
    Cqu_1 = Cqu + C1int_qu
    #
    detC_b1 = Cqq_1 * Cuu_1 - Cqu_1**2
    invCqq_b = Cuu_1 / detC_b1
    invCuu_b = Cqq_1/detC_b1
    invCqu_b = - Cqu_1/detC_b1
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_1 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)
    # 2.
    #  behind cloud2
    #   in this case the polarization get added the pol of the cloud:
    q_m_b += q2
    u_m_b += u2
    # adding the intrinsic scatter term
    Cqq_2 = Cqq_1 + C2int_qq
    Cuu_2 = Cuu_1 + C2int_uu
    Cqu_2 = Cqu_1 + C2int_qu
    #
    detC_b2 = Cqq_2 * Cuu_2 - Cqu_2**2
    invCqq_b = Cuu_2 / detC_b2
    invCuu_b = Cqq_2 / detC_b2
    invCqu_b = - Cqu_2 / detC_b2
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_2 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)
    
    # 3.
    #  behind cloud3
    #   in this case the polarization of cloud3 is added to the pol.
    q_m_b += q3
    u_m_b += u3
    #   and the intrinsic scatter contribution of cloud3 is also added
    Cqq_3 = Cqq_2 + C3int_qq
    Cuu_3 = Cuu_2 + C3int_uu
    Cqu_3 = Cqu_2 + C3int_qu
    #
    detC_b3 = Cqq_3 * Cuu_3 - Cqu_3**2
    invCqq_b = Cuu_3 / detC_b3
    invCuu_b = Cqq_3 / detC_b3
    invCqu_b = - Cqu_3 / detC_b3
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_3 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)

    # 4.
    #  behind cloud4
    #   in this case the polarization of cloud4 is added to the pol.
    q_m_b += q4
    u_m_b += u4
    #   and the intrinsic scatter contribution of cloud4 is also added
    Cqq_4 = Cqq_3 + C4int_qq
    Cuu_4 = Cuu_3 + C4int_uu
    Cqu_4 = Cqu_3 + C4int_qu
    #
    detC_b4 = Cqq_4 * Cuu_4 - Cqu_4**2
    invCqq_b = Cuu_4 / detC_b4
    invCuu_b = Cqq_4 / detC_b4
    invCqu_b = - Cqu_4 / detC_b4
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_4 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)

    # # #
    # per-star likelihood of measuring their q,u given their plx,sig_plx
    # and the cloud parameters:
    #
    l_i = P_0 * np.exp(-.5*chi2_f)/(2 * np.pi * np.sqrt(detC_f)) +\
            P_1 * np.exp(-.5*chi2_1)/(2 * np.pi * np.sqrt(detC_b1)) +\
            P_2 * np.exp(-.5*chi2_2)/(2 * np.pi * np.sqrt(detC_b2)) +\
            P_3 * np.exp(-.5*chi2_3)/(2 * np.pi * np.sqrt(detC_b3)) +\
            P_4 * np.exp(-.5*chi2_4)/(2 * np.pi * np.sqrt(detC_b4))
    #
    l_i = l_i[l_i!=0]
    logll = np.sum(np.log(l_i))
    #
    if np.isnan(logll):
        logll = - np.inf
    #
    return logll
#


'''
################################################################################
                                5 layers models
################################################################################
'''

def logLL_FiveLayers(theta,star_plx,star_q,star_u,err_plx,err_q,err_u,c_qu):
    '''
        Implementation of the 5-layer model.
        Input:
        - theta: includes the 5 x 6 parameters for the layers
        - star_plx: parallaxes of the stars in [mas]
        - star_q: q=Q/I Stokes parameter of the star [fraction]
        - star_u: u=U/I Stokes parameter of the star [fraction]
        - err_...: corresponding errors
        - c_qu: uncertainty correlation between q and u from observation
                [fraction^2]
        Output:
        logll: the log-likelihood estimated for the model with parameter
                values in theta and for the input data
    '''

    plx1,q1,u1,C1int_qq,C1int_uu,C1int_qu,\
        plx2,q2,u2,C2int_qq,C2int_uu,C2int_qu,\
            plx3,q3,u3,C3int_qq,C3int_uu,C3int_qu,\
                plx4,q4,u4,C4int_qq,C4int_uu,C4int_qu,\
                    plx5,q5,u5,C5int_qq,C5int_uu,C5int_qu = theta
    #
    if ((C1int_qu**2 >= C1int_qq * C1int_uu) or
            (C2int_qu**2 >= C2int_qq * C2int_uu) or
                (C3int_qu**2 >= C3int_qq * C3int_uu) or
                    (C4int_qu**2 >= C4int_qq * C4int_uu) or
                        (C5int_qu**2 >= C5int_qq * C5int_uu)):
        return -np.inf
    if ((plx1<=plx2) or (plx2<=plx3) or (plx3<=plx4) or (plx4<=plx5)):
        return -np.inf
    #
    #
    q_m_b = 0*star_q
    u_m_b = 0*star_q
    q_m_f = 0*star_q
    u_m_f = 0*star_q
    Cqq = err_q**2
    Cuu = err_u**2
    Cqu = c_qu
    #
    
    # probability of each star to be in front of all cloud
    P_0 = (1./2 - 1./2*special.erf((plx1 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in between cloud1 and cloud2
    P_1 = (1./2*special.erf((plx1 - star_plx)/(np.sqrt(2)*err_plx)) -\
            1./2*special.erf((plx2 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in between cloud2 and cloud3
    P_2 = (1./2*special.erf((plx2 - star_plx)/(np.sqrt(2)*err_plx)) -\
            1./2*special.erf((plx3 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in between cloud3 and cloud4
    P_3 = (1./2*special.erf((plx3 - star_plx)/(np.sqrt(2)*err_plx)) -\
            1./2*special.erf((plx4 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in between cloud3 and cloud4
    P_4 = (1./2*special.erf((plx4 - star_plx)/(np.sqrt(2)*err_plx)) -\
            1./2*special.erf((plx5 - star_plx)/(np.sqrt(2)*err_plx)))
    # probability of each star to be in the background of cloud4
    P_5 = (1./2 + 1./2*special.erf((plx5 - star_plx)/(np.sqrt(2)*err_plx)))


    # 0. stars in front of all clouds
    #  in this case the polarization remain unchanged (=0)
    #   and the covariance matrix remain unchanged too, only observational
    # handling the covariance matrix
    detC_f = Cqq * Cuu - Cqu**2
    invCqq_f = Cuu / detC_f
    invCuu_f = Cqq/detC_f
    invCqu_f = - Cqu/detC_f
    #
    eta_q_f = star_q - q_m_f
    eta_u_f = star_u - u_m_f
    #
    chi2_f = (eta_q_f**2 * invCqq_f + eta_u_f**2 * invCuu_f +\
                    2 * eta_q_f * eta_u_f * invCqu_f)

    
    # 1.    #   #   #   #   #   #   #   #
    #  behind cloud1
    #   in this case the polarization of cloud1 is added to the pol.
    q_m_b += q1
    u_m_b += u1
    #
    #   and the intrinsic scatter contribution from cloud1 is also added
    Cqq_1 = Cqq + C1int_qq
    Cuu_1 = Cuu + C1int_uu
    Cqu_1 = Cqu + C1int_qu
    #
    detC_b1 = Cqq_1 * Cuu_1 - Cqu_1**2
    invCqq_b = Cuu_1 / detC_b1
    invCuu_b = Cqq_1/detC_b1
    invCqu_b = - Cqu_1/detC_b1
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_1 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)
    # 2.
    #  behind cloud2
    #   in this case the polarization get added the pol of the cloud:
    q_m_b += q2
    u_m_b += u2
    # adding the intrinsic scatter term
    Cqq_2 = Cqq_1 + C2int_qq
    Cuu_2 = Cuu_1 + C2int_uu
    Cqu_2 = Cqu_1 + C2int_qu
    #
    detC_b2 = Cqq_2 * Cuu_2 - Cqu_2**2
    invCqq_b = Cuu_2 / detC_b2
    invCuu_b = Cqq_2 / detC_b2
    invCqu_b = - Cqu_2 / detC_b2
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_2 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)
    
    # 3.
    #  behind cloud3
    #   in this case the polarization of cloud3 is added to the pol.
    q_m_b += q3
    u_m_b += u3
    #   and the intrinsic scatter contribution of cloud3 is also added
    Cqq_3 = Cqq_2 + C3int_qq
    Cuu_3 = Cuu_2 + C3int_uu
    Cqu_3 = Cqu_2 + C3int_qu
    #
    detC_b3 = Cqq_3 * Cuu_3 - Cqu_3**2
    invCqq_b = Cuu_3 / detC_b3
    invCuu_b = Cqq_3 / detC_b3
    invCqu_b = - Cqu_3 / detC_b3
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_3 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)

    # 4.
    #  behind cloud4
    #   in this case the polarization of cloud4 is added to the pol.
    q_m_b += q4
    u_m_b += u4
    #   and the intrinsic scatter contribution of cloud4 is also added
    Cqq_4 = Cqq_3 + C4int_qq
    Cuu_4 = Cuu_3 + C4int_uu
    Cqu_4 = Cqu_3 + C4int_qu
    #
    detC_b4 = Cqq_4 * Cuu_4 - Cqu_4**2
    invCqq_b = Cuu_4 / detC_b4
    invCuu_b = Cqq_4 / detC_b4
    invCqu_b = - Cqu_4 / detC_b4
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_4 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)

    # 5.
    #  behind cloud5
    #   in this case the polarization of cloud5 is added to the pol.
    q_m_b += q5
    u_m_b += u5
    #   and the intrinsic scatter contribution of cloud5 is also added
    Cqq_5 = Cqq_4 + C5int_qq
    Cuu_5 = Cuu_4 + C5int_uu
    Cqu_5 = Cqu_4 + C5int_qu
    #
    detC_b5 = Cqq_5 * Cuu_5 - Cqu_5**2
    invCqq_b = Cuu_5 / detC_b5
    invCuu_b = Cqq_5 / detC_b5
    invCqu_b = - Cqu_5 / detC_b5
    #
    eta_q_b = star_q - q_m_b
    eta_u_b = star_u - u_m_b
    #
    chi2_5 = (eta_q_b**2 * invCqq_b + eta_u_b**2 * invCuu_b +\
                    2 * eta_q_b * eta_u_b * invCqu_b)

    # # #
    # per-star likelihood of measuring their q,u given their plx,sig_plx
    # and the cloud parameters:
    #
    l_i = P_0 * np.exp(-.5*chi2_f)/(2 * np.pi * np.sqrt(detC_f)) +\
            P_1 * np.exp(-.5*chi2_1)/(2 * np.pi * np.sqrt(detC_b1)) +\
            P_2 * np.exp(-.5*chi2_2)/(2 * np.pi * np.sqrt(detC_b2)) +\
            P_3 * np.exp(-.5*chi2_3)/(2 * np.pi * np.sqrt(detC_b3)) +\
            P_4 * np.exp(-.5*chi2_4)/(2 * np.pi * np.sqrt(detC_b4)) +\
            P_5 * np.exp(-.5*chi2_5)/(2 * np.pi * np.sqrt(detC_b5))
    #
    l_i = l_i[l_i!=0]
    logll = np.sum(np.log(l_i))
    #
    if np.isnan(logll):
        logll = - np.inf
    #
    return logll
#

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# That's all.                                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Done.
