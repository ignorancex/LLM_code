
import numpy as np
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp
import concurrent.futures
import math
from collections import namedtuple
import pandas as pd
import os
import csv
from scipy import interpolate


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

def f_H2_KMT(Sigmag, ZZsun):
    """
        Krumholz et al. model for H2 
        Sigmag = surface density of gas on ~kpc scale in Msun/kpc^2
            ZZsun = gas metallicity in units of solar 
    """
    ZdZsun = np.maximum(ZZsun, 0.08)
    x = 3.1/4.1*(1.+3.1*ZdZsun**0.365)
    tc = 3.34e-7* Sigmag * ZdZsun  # assumes clumpiness factor of c=5
    s = np.log(1.+0.6*x + 0.01*x*x)/(0.6*tc) 
    isl2 = (s<2)
    dummy = np.zeros_like(Sigmag)
    dummy[isl2] = (1.0 - 0.75*s[isl2]/(1.+0.25*s[isl2]))
    return dummy


###f_H2_GD14_improved = gd14_improved
###f_H2_GD14new = gd14_new
###f_H2_GD14 = gd14
### the gd14 model is the norm

def f_H2_GD14_improved(Sigmag, ZZsun, UMW):
    """
    Gnedin & Draine 2014 model for H2 
    Sigmag = surface density of gas on ~kpc scale in Msun/kpc^2
        ZZsun = gas metallicity in units of solar 
    """
    DMW = np.minimum(ZZsun,10.)
    DMW2 = DMW*DMW
    gd = np.sqrt(DMW2 + 0.0289) # 0.17**2 = 0.0289
    xd = ((0.001+0.1*UMW)/(1.+DMW2))**(0.5-0.05*DMW)
    alfa = 1.0 + 1.75*xd / (1.0 + xd*xd) * (0.7+0.25*DMW)
    xd = UMW * 0.1
    SigR1 = 1.e6*(.75+40*xd**0.725 / (1.+xd**0.7)) / gd
    Rdum = np.power(Sigmag/SigR1, alfa)
    dummy = Rdum / (1. + Rdum)

    return dummy

def f_H2_GD14new(Sigmag, ZZsun, UMW):
    """
    Gnedin & Draine 2014 model for H2 , 
    namely, improved model from the erratum
    Sigmag = surface density of gas on ~kpc scale in Msun/kpc^2
        ZZsun = gas metallicity in units of solar 
    """
    gd = np.sqrt((0.5*ZZsun)**2 + 0.0289) # 0.17**2 = 0.0289
    Ud = (0.001+0.1*UMW)**0.35
    Ud2 = Ud * Ud
    alfa = 1. + 0.7*Ud/(1.0+Ud2)
    SigR1 = 4.e7/gd * Ud2/(1.0 + Ud2)
    q =  np.power(Sigmag/SigR1, alfa)
    Rdum = q * (1. + 0.25*q) / 1.25
    fH2 = np.maximum(0., Rdum/(Rdum+1.))
    return fH2

def f_H2_GD14(Sigmag, ZZsun, UMW):
    """
    Gnedin & Draine 2014 model for H2 
    corrected as described in the erratum to their paper
    Sigmag = surface density of gas on ~kpc scale in Msun/kpc^2
        ZZsun = gas metallicity in units of solar 
    """
    gd = np.sqrt((0.5*ZZsun)**2 + 0.0289) # 0.17^2 = 0.0289
    Ud = np.sqrt(0.001+0.1*UMW)
    alfa = 0.5 + 1.0/(1.0+np.sqrt(1.67e-3*UMW*ZZsun**2))
    SigR1 = 5.e7/gd * Ud/(1.0 + 1.69*Ud)
    Rdum = np.power(Sigmag/SigR1, alfa)
    fH2 = np.maximum(0., Rdum/(Rdum+1.))
    return fH2

def rho_crit(z, Om0, OmL): 
    """
    critical density of the universe at redshift z in Msun/kpc^3
    assuming H0 = 71 km/s/Mpc - the value of the ELVIS simulations
    """
    return 139.894 * (Om0*(1+z)**3 + OmL)

def Omega(z): 
    z13 = 0.266 * (1.+z)**3 
    return z13 / (z13 + 0.734)

pi2_18 = 18. * np.pi**2


def Delta_vir(z): 
    """
    virial density contrast relative to rho_crit(z) 
    using Bryan & Norman 1998 fitting formula for flat LCDM
    """
    xd = Omega(z) - 1.
    return pi2_18 + (82. - 39. * xd) * xd  


def rhalf(Mh, z, h23i, Om0, OmL):
    '''
    This is in units of kpc, check Eq 19 in GRUMPY model paper. The proportionality constant 0.1 is the chid parameter that is discussed here
    '''
    # need to clean up expression for rd below, this is approximate and uses Delta=200*rho_crit
    rd = 0.1 * 162.63 * (Mh*1e-12)**(1./3) * h23i * (Om0*(1+z)**3 + OmL)**(-1./3)
    return rd


def f_H2_BS16(Sigmag, ZZsun, UMW, clfact=1, fgas=1):
    """
    Bialy & Sternberg 2014, 2016 model for H2 
    Sigmag = surface density of gas on ~kpc scale in Msun/kpc^2
        ZZsun = gas metallicity in units of solar 
    """
    siggt = np.sqrt(ZZsun**2 + 0.0289)
    nave = 1.28 * (Sigmag/1e7)**2 / fgas**2
    Ud = np.maximum(UMW, 1.e-2) #np.clip(UMW, 1.e-5, 1.e4)
    aG = (100/(clfact*nave))*0.59*Ud*(9.9/(1.0+8.9*siggt))**0.37
    Str = (1e6 + 3.684e6*np.log((0.5*aG)+1.0)) / siggt
    
    alfa = 1.25  * (1. + Ud**0.5)
    alfa = np.minimum(alfa, 10.0)
    
    Rdum = np.power(0.5*Sigmag/Str, alfa)
    dummy = Rdum / (1. + Rdum)
    
    return dummy


rdmin, rdmax = 0.001, 7.
pi2 = np.pi * 2.

def MH2(Mh=None,Mg=None,MZg=None, sfr=None, z=None,h23i=None, Om0=None, OmL=None, Zsun=None, Z_IGM =None,SigHIth=None, irstar=0, rpert=0.,h2_model = "gd14"):
    '''
    Function that computes the H2 mass
    '''
    rd = rhalf(Mh, z,h23i, Om0, OmL) * 10.**rpert
    #rd in in kpc
    rd = np.maximum(0.001, rd)
    rd = np.minimum(100., rd)
    #I dont think this later condition of being above 100 kpc disk radius will be met in our dwarf galaxies ...

    #the radius grid
    # rg = np.linspace(rdmin*rd, rdmax*rd, 40) 
    rg = np.linspace(0, rdmax*rd, 50) 

    #Is the downward trend in the profile a result of spline extrapolation

    #see Section 2.3 in Kravtsov & Manwadkar 2022 for details on these formulas
    area2 = 2.*np.pi*rd**2

    #the central surface density
    Sigma0 = Mg / area2
    Sig0d = 1e-6 * Sigma0
    if Sig0d > SigHIth:
        xHI = -np.log(SigHIth / Sig0d)
    else:
        xHI = rdmin 

    #xHI is the radius in units of scale radius rd where the gas density falls below the HI threshold density that is inputted
    #note that SigHIth is in Msun/pc^2 and so we have to convert Sigma0 into same units, hence the factor of 1e-6
    #is gas density is larger than SigHIth, then we have some part of gas disk is in neutral HI

    #this is the gas surface density profile
    sgd_gas = Sigma0 * np.exp(-rg/rd)

    #sgd = MHI / area2 * np.exp(-rg/rd)
    # surface density of SFR relative to MW solar neighborhood
    # see Fig 7 in Kennicutt & Evans 2012
    sig_sfr = sfr / area2 * 1e-9 / 2.5e-3 # relative to MW
    # factor of 0.5 is to account that roughly half of heavy elements are in dust particles
    ZZsun = np.maximum(0.5*MZg / Mg / Zsun, Z_IGM) 
    ##the above quantity is in units of Zsolar. So Z_IGM has to also be in solar units

    #computing the fraction of H2 in HI gas disk
    if h2_model == "gd14":
        fH2 = f_H2_GD14(sgd_gas, ZZsun, sig_sfr)
    if h2_model == "gd14_new":
        fH2 = f_H2_GD14new(sgd_gas, ZZsun, sig_sfr)
    if h2_model == "gd14_improved":
        fH2 = f_H2_GD14_improved(sgd_gas, ZZsun, sig_sfr)
    if h2_model == "kmt12":
        fH2 = f_H2_KMT(sgd_gas, ZZsun)

    #I need to think a little more carefully about the gas density here, is it HI gas or is it just normal gas? Is there a difference???

    ##this is the integrand we will be integrating to get the H2 mass (see Eqauation 30)
    sgd = rg * sgd_gas * fH2

    #is fH2 already becoming zero for values where we do not have any HI?
    
    inz = np.ravel(np.nonzero(sgd))
    nnz = np.size(inz)
    
    #the spacing in the radius grid
    deltar = rg[1] - rg[0]

    MH2 =  pi2 * np.sum(sgd[rg<xHI*rd]*deltar)
    #MH2 is the total H2 mass

    rstar = 0.
    if nnz > 0:
        rstar = rg[inz[-1]]
        if irstar:             
            sgsp = UnivariateSpline(rg, sgd, s=0.0)

            sgsp_gas = UnivariateSpline(rg, sgd_gas, s=0.0)

            #note that rg is not normalized radius, it is radii in physical kpc units
            #however, xHI is normalized to disk radius rd and hence we have to multiply
            MH2 = 2.0 * np.pi * sgsp.integral(rdmin*rd, xHI*rd) #rg[-1])
            MH2half = 0.5*MH2
            rstar = 0.
            k = 0 
            MH2r = 0
            #here trying to find the half mass radius of the H2 profile
            while MH2r < MH2half:
                k += 1
                MH2r = 2.0 * np.pi * sgsp.integral(0., rg[k])
                
            rstar = rg[k] #rg[inz[-1]]
            rout = rg[inz[-1]]
    else:
        rstar, rout = 0., 0.
        sgsp = UnivariateSpline(rg, sgd, s=0.0)
        sgsp_gas = UnivariateSpline(rg, sgd_gas, s=0.0)


    if irstar: 
        return MH2, rstar, rout, sgsp, sgsp_gas, xHI*rd
    else:
        return MH2, rstar

####### reionization modelling ################

def eps(z, beta, gamma):
    '''
    
    '''
    t1 = (gamma * z**(gamma-1) / beta**gamma) * np.exp((z/beta)**gamma)
    t2 = 0.63*(1 + np.exp((z/beta)**gamma))
    t3 = (1 + np.exp((z/beta)**gamma))**2
    ret = (t1 + t2) / t3
    if math.isnan(ret) or ret < 1e-7:
        ret = 0
    return ret

def m_c_modified(z, beta, gamma):
    return  2*8.45e9 * np.exp(-0.63*z) / (1 + np.exp((z/beta)**gamma) )
    #1.69e10 normalization is 2 times the best fit value from the Okamoto+2008 data at z=0

def m_c_simple(z):
    return  8.45e9 * np.exp(-0.63*z)
    #1.69e10 normalization is 2 times the best fit value from the Okamoto+2008 data at z=0

def E(z, Om0, OmL):
    """E(z) is dimensionless Hubble function"""
    return np.sqrt(Om0*(1 + z)**3 + OmL)

def fgcum_uv(Mh, z, beta, gamma):
    '''
    baryonic fraction as a function of Mh and z for a given evolution 
    of characteristic mass scale parametrized by beta and gamma

    Parameters:
    -----------
        z_re: float, model redshift of reionization
        gamma: float, parameter for sharp increase in Mc
    Returns:
       beta: float
    '''
    dummy = m_c_modified(z,beta,gamma)
    alfa = 2. #original is alfa = 2.
    fact = (2.**(alfa/3.)-1.)
    mualfa = (dummy/Mh)**alfa
    fgcum = 1.0/(1.0+fact*mualfa)**(3./alfa)
    return fgcum, fact, mualfa

def step_fgin(Mh, z, zrei):
    '''
    gas inflow baryonic fraction relatve to universla baryonic fraction
    in step reionization model

    Parameters:
    -----------
        Mh: float, halo mass
        z: float, redshift
        beta: float, parameter for when sharp increase occurs in Mc.
        gamma: float, parameter for sharp increase in Mc
        z_re: float, model redshift of reionization        
    Returns:
       beta: float
    '''
    fgin = 1.
    dummy = m_c_simple(z)
    if z <= zrei and Mh < dummy:
        fgin = 0.
    return fgin


def get_beta(zrei, gamma):
    """
    compute the beta parameter for the gas accretion suppression equation. 
    The beta parameter controls at what redshift the sharp increase in Mc occurs.
    It is calculated by constraining the Mc at reionization to Okamoto simulations
    given a certain z_re and gamma


    Parameters:
    -----------
        z_re: float, model redshift of reionization
        gamma: float, parameter for sharp increase in Mc
    Returns:
       beta: float
    """
    # mc_oka = 2*8.45e9 * np.exp(-0.63*9) / (1 + np.exp((9/8.7)**15)) 
    #above is the Mc at time of reionization in Okamoto+2008 simulations. 
    #the above describes the value of Mc at z = 9 in the best fit. 
    # alpha = (2*8.45e9* np.exp(-0.63*z_re) / mc_oka ) - 1
    alpha = 1820*np.exp(-0.63*zrei) - 1  #this expression is just above calc simplified
    beta = zrei * (np.log(alpha))**(-1./gamma)
    return beta

#############################################################
#stochastic SFR functions

def generate_random(nsize=100, sigma_int=0.4, psd_func=None, args=[], 
                    tmin=0., tmax=13.8):

    tgrid = np.linspace(tmin, tmax, nsize)
    # Period
    tperiod = tmax - tmin
    dt = tgrid[1] - tgrid[0]

    # frequency vector
    f = np.fft.fftfreq(nsize, d = dt) 

    # Gaussian random numbers with zero mean and unit variance
    d = np.random.normal(0., 1., nsize)
    dfft = np.fft.fft(d) # Fourier transform them 
   
    tau_break, alpha = args
    kwargs = {'tau_break': tau_break, 'alpha': alpha, 'f': np.abs(f)}
    # Compute normalization of the power spectrum that corresponds to input parameters
    norm = np.sum(psd_func(**kwargs)) / tperiod
    
    # compute power spectrum and renormalize it so that random series has desired variance 
    #psf = psd_func(**kwargs) / tperiod * nsize * (sigma_int**2 / norm)
    psf = psd_func(**kwargs) / tperiod * sigma_int**2 * nsize
    
    # multiply Fourier amplitudes by sqrt(PSD)
    ps = np.sqrt(psf) * dfft
    
    # fix the DC mode to 1
    ps[0] = 1
    # Fourier transform back to real space
    # get rid of the very small imaginary part
    delta = np.real(np.fft.ifft(ps))

    # return delta and norm
    return tgrid, f, delta, norm 

def PSD_new(sigma2=1.0, tau_break=10, alpha=1., f=None):
    """
    PSD resulting from the damped random walk given by Eq. 5 in Caplar & Tacchella 2019
    """
    psd = np.zeros_like(f)
    fa = np.abs(f)
    psd[fa>0] = sigma2 / (1 + (tau_break*np.abs(f[fa>0]))**(alpha))
    psd[fa==0] = 0.
    
    return psd

def PSD(s, dt, nsize):
    '''
    Compute power spectrum and PSD of time series s
    
    Parameters: 
        s: numpy vector of floats, time series values
        dt: time step size on which s is defined
        
    Returns: 
        ps: numpy vector of floats, power spectrum
        psd: numpy vector of floats, PSD 
    '''
 
    # frequency vector
    f = np.fft.fftfreq(nsize, d = dt) 
    c = np.fft.fft(s)
    # Power spectrum 
    ps = (np.real(c)**2 + np.imag(c)**2) / s.size**2 
    ps = 2 * ps[f>0] # only retain the positive frequency
    f = f[f>0]
    # PSD 
    psd = ps * dt * s.size
    
    return f, ps, psd

#############################################################

#This is the non-stochastic SFR function
def gevol_func(t, Mvec, *args):
    """
    galaxy evolution model implemented in a function
    args = []
    fbuni, tausf, Z_IGM = float numbers, universal bar. fraction, depletion time in Gyrs
                            metallicity of IGM (MZ/Mg, not in solar units)
    ztsp = spline object for z(t), where t is in Gyrs
    dlmdltsp = splint object for dln Mh(t)/dlnt                
    """

    Mh, Mg, Ms, MZg, MZs,_,_ = Mvec

    cosmo,params,zspl,dlmdltspl,rpd = args

    # ensure that z>-1, in case integration takes t to z<0 
    z = np.maximum(-0.999, zspl(t))

    # make sure that Mgas and Mhalo are always positive 
    Mg = np.maximum(Mg, 1e-6)
    Mh = np.maximum(Mh, 1e-6)
    
    dMhdtd = Mh * np.maximum(dlmdltspl(t), 0.001) / t
    
    # computing MH2
    try:
        sfr = gevol_func.sfr
    except: 
        sfr = 0.

    
    #the MH2 functions needs Z_IGM in units of Zsolar, so we renormalize it
    M_H2, _ = MH2(Mh=Mh, Mg=Mg, MZg=MZg, sfr=sfr, z=z, h23i=cosmo.h23i, Om0=cosmo.Om0, OmL=cosmo.OmL, 
                      Zsun=params.Zsun, Z_IGM =params.Z_IGM / params.Zsun, SigHIth=params.SigHIth, rpert = rpd,h2_model = params.h2_model)
    
    # evolution of stellar mass
    sfr = M_H2 / params.tausf
    gevol_func.sfr = sfr 
    dMsdtd = params.Rloss1 * sfr

    # wind
    eta_wind = 0. 
    if Ms > 1.:
        # modified version of the Muratov et al. 2015 model 
        # eta_wind = np.maximum(0., 3.6*(Ms*1.e-10)**(-0.35) - 4.5)
        eta_wind = np.maximum(0., params.eta_norm*(Ms/params.eta_mass)**(params.eta_power) - params.eta_c)
        eta_wind = np.minimum(eta_wind, 2000.)

    # simple Mhot for now, can change later (will only affect MW-sized hosts and above)
    Mhot = 4.e11 
    alfa = 4. 
    epsin = 1.0 - 1.0 / (1.0 + (2.**(alfa/3.) - 1.) * (Mhot/Mh)**alfa)**(3./alfa)
    # normalize accretion by a constant and 
    epsin *= params.epsin_pr

    if params.rei_model == "okamoto":
        fgcum, fact, mualfa = fgcum_uv(Mh, z, params.beta, params.gamma)
        X = 3. * fact * mualfa / (1. + fact*mualfa)
        eps_mod = eps(z, params.beta, params.gamma)
        # this is the modified characteristic mass expression 
        Mgin = (cosmo.fbuni * epsin * fgcum * (dMhdtd * (1. + X) 
                - 2 * cosmo.H0iGyr * E(z, cosmo.Om0, cosmo.OmL) * (1.+z) * Mh * X * eps_mod)) #experimenting here 
    elif params.rei_model == "step":
        #step reionization equation
        fgin = step_fgin(Mh,z,params.zrei)
        Mgin = cosmo.fbuni * epsin * fgin * dMhdtd 

    elif params.rei_model == "noreion":
        Mgin = cosmo.fbuni * dMhdtd * epsin

    # forcing gas inflow rate to be positive. 
    Mgin = np.maximum(0, Mgin)
    dMgdtd = Mgin - (params.Rloss1 + eta_wind) * sfr

    Zg = MZg / Mg
    dMZgdtd = params.Z_IGM*Mgin + (params.yZ*params.Rloss1 - (params.Rloss1 + params.zwind*eta_wind)*Zg)*sfr

    dMZsdtd = params.Rloss1 * Zg * sfr

    dMgindtd = Mgin
    dMgoutdtd = eta_wind * sfr

    return dMhdtd, dMgdtd, dMsdtd, dMZgdtd, dMZsdtd,dMgindtd,dMgoutdtd

#This is the stochastic SFR function

def gevol_func_stoch_sfr(t, Mvec, *args):
    """
    galaxy evolution model implemented in a function
    args = []
    fbuni, tausf, Z_IGM = float numbers, universal bar. fraction, depletion time in Gyrs
                            metallicity of IGM (MZ/Mg, not in solar units)
    ztsp = spline object for z(t), where t is in Gyrs
    dlmdltsp = splint object for dln Mh(t)/dlnt                
    """

    Mh, Mg, Ms, MZg, MZs,_,_ = Mvec
    # deltatspl: input new t, output Delta(t) evaluated at this time step
    cosmo,params,zspl,dlmdltspl,rpd,deltatspl = args

    # ensure that z>-1, in case integration takes t to z<0 
    z = np.maximum(-0.999, zspl(t))

    # make sure that Mgas and Mhalo are always positive 
    Mg = np.maximum(Mg, 1e-6)
    Mh = np.maximum(Mh, 1e-6)
    
    dMhdtd = Mh * np.maximum(dlmdltspl(t), 0.001) / t
    
    # computing MH2
    try:
        sfr = gevol_func_stoch_sfr.sfr
    except: 
        sfr = 0.
        
    # retain mean SFR
    try:
        sfr_mean = gevol_func_stoch_sfr.sfr_mean
    except: 
        gevol_func_stoch_sfr.sfr_mean = [0.]
        gevol_func_stoch_sfr.t_sfrmean = [0.]
        
    
    #the MH2 functions needs Z_IGM in units of Zsolar, so we renormalize it
    M_H2, _ = MH2(Mh=Mh, Mg=Mg, MZg=MZg, sfr=sfr, z=z, h23i=cosmo.h23i, Om0=cosmo.Om0, OmL=cosmo.OmL, 
                      Zsun=params.Zsun, Z_IGM =params.Z_IGM / params.Zsun, SigHIth=params.SigHIth, rpert = rpd,h2_model = params.h2_model)
    # evolution of stellar mass
    sfr = M_H2 / params.tausf # params is a dict

    ############# apply stochasticity in SFR
    # compute new delta at the current t (Gyr)
    new_delta = deltatspl(t)
    #new_delta = 0
    sfr = sfr * 10**new_delta
    #sfr_mean = sfr
    # record average SFR over every 0.1 Gyr
    t_sfr_ave = params.t_sfr_ave

    while t - gevol_func_stoch_sfr.t_sfrmean[0] >= t_sfr_ave:
        if len(gevol_func_stoch_sfr.t_sfrmean) == 1:
            break
        gevol_func_stoch_sfr.sfr_mean.pop(0)
        gevol_func_stoch_sfr.t_sfrmean.pop(0)     
    # if the integration goes backward in time 
    while t <= gevol_func_stoch_sfr.t_sfrmean[-1]:
        if len(gevol_func_stoch_sfr.t_sfrmean) == 1:
            break
        gevol_func_stoch_sfr.sfr_mean.pop(-1) # keep throwing out the last time step
        gevol_func_stoch_sfr.t_sfrmean.pop(-1)       
            
    gevol_func_stoch_sfr.sfr_mean.append(sfr)
    gevol_func_stoch_sfr.t_sfrmean.append(t)
    # record the average SFR in 0.1 Gyr time interval
    sfr_mean = np.average(gevol_func_stoch_sfr.sfr_mean)
    # if there's no gas, stop gas outflow
    f_H2 = M_H2/Mg
    if f_H2 < 1.e-5:
        sfr_mean = 0.

    gevol_func_stoch_sfr.sfr = sfr 

    dMsdtd = params.Rloss1 * sfr # a fraction R of the gas converted stars is returned immediately to the ISM

    # wind
    eta_wind = 0. 
    if Ms > 1.:
        # modified version of the Muratov et al. 2015 model 
        # eta_wind = np.maximum(0., 3.6*(Ms*1.e-10)**(-0.35) - 4.5)
        eta_wind = np.maximum(0., params.eta_norm*(Ms/params.eta_mass)**(params.eta_power) - params.eta_c)
        eta_wind = np.minimum(eta_wind, 2000.) # equation 34
        # for z > 3, we suppress feedback (Dave et al. 2019) following MUFASA
#         if z > 3:
#             eta_wind = eta_wind * (4/(1+z))**2

    # simple Mhot for now, can change later (will only affect MW-sized hosts and above)
    Mhot = 4.e11 
    alfa = 4. 
    epsin = 1.0 - 1.0 / (1.0 + (2.**(alfa/3.) - 1.) * (Mhot/Mh)**alfa)**(3./alfa) # equation 4 & 6
    # normalize accretion by a constant and 
    epsin *= params.epsin_pr

    if params.rei_model == "okamoto":
        fgcum, fact, mualfa = fgcum_uv(Mh, z, params.beta, params.gamma)
        X = 3. * fact * mualfa / (1. + fact*mualfa) # equation 13
        eps_mod = eps(z, params.beta, params.gamma)
        # this is the modified characteristic mass expression 
        # equation 2 & 11
        Mgin = (cosmo.fbuni * epsin * fgcum * (dMhdtd * (1. + X) 
                - 2 * cosmo.H0iGyr * E(z, cosmo.Om0, cosmo.OmL) * (1.+z) * Mh * X * eps_mod)) #experimenting here 
    elif params.rei_model == "step":
        #step reionization equation
        fgin = step_fgin(Mh,z,params.zrei)
        Mgin = cosmo.fbuni * epsin * fgin * dMhdtd 

    elif params.rei_model == "noreion":
        Mgin = cosmo.fbuni * dMhdtd * epsin

    # forcing gas inflow rate to be positive. 
    Mgin = np.maximum(0, Mgin)
    #dMgdtd = Mgin - (params.Rloss1 + eta_wind) * sfr # inflow rate - outflow - fraction of ISM turned into stars
    dMgdtd = Mgin - params.Rloss1*sfr - eta_wind * sfr_mean
    
    Zg = MZg / Mg
    # chemical evolution of gas
    # enrichment by IGM, fraction of yield generated by stars returned to ISM, loss of chemicals due to ISM turning into stars and loss of chemicals due to outflows
    dMZgdtd = params.Z_IGM*Mgin + (params.yZ*params.Rloss1*sfr - (params.Rloss1*sfr + params.zwind*eta_wind*sfr_mean)*Zg)
    #dMZgdtd = params.Z_IGM*Mgin + (params.yZ*params.Rloss1 - (params.Rloss1 + params.zwind*eta_wind)*Zg)*sfr
    # chemical evolution of stars, fraction of heavy elements in the ISM turned into stars
    dMZsdtd = params.Rloss1 * Zg * sfr
    # mass inflow rate
    dMgindtd = Mgin
    # mass outflow rate
    #dMgoutdtd = eta_wind * sfr
    dMgoutdtd = eta_wind * sfr_mean
#     if debug == True:
#         print(sfr_mean)
#         print(f'{t} {sfr_mean:.3g} {Mg:.3e} {Ms:.3e} {M_H2:.3e}')

    return dMhdtd, dMgdtd, dMsdtd, dMZgdtd, dMZsdtd,dMgindtd,dMgoutdtd

########################################################
########################################################
########################################################

def integrate_single(info):
    ''' 
    Function that returns Mouti for a single MAH

    For non-stochastic SFR
    '''

    t0 = info['t0']
    m0 = info['m0']
    mstar_ini = info['mstar_ini']
    astart = info['astart']
    touti = info['tout']
    params = info['model_params']
    cosmo = info['cosmo']
    zspl = info['zspl']
    dlmdltspl = info['dlmdltspl']
    rpd = info['rpd']

    cosmo = convert(cosmo)
    params = convert(params)

    zstart = 1. / astart - 1.

    if params.rei_model == "noreion":
        mg0 = cosmo.fbuni * m0

    if params.rei_model == "okamoto":
        mg0 = cosmo.fbuni * m0 * fgcum_uv(m0, zstart, params.beta, params.gamma)[0]

    if params.rei_model == "step":
        mg0 = cosmo.fbuni * m0 * step_fgin(m0, zstart, params.zrei)

    #the starting integration point
    #M200c, Mgas, Mstar, MZg, MZs, Mgin, Mgout
    #the latter two are the quantities for total gas accreted onto halo and total gas driven out in outflows 
    y0 = np.array([m0, mg0, mstar_ini, params.Z_IGM*mg0, 0.,0.,0.]) 

    # solve the system of ODEs
    #the t_eval originally used to be just touti
    sol = solve_ivp(gevol_func, [t0, touti[-1]], y0, t_eval=np.linspace(touti[0],touti[-1],1000), 
                    method='RK45',
                    args=(cosmo,params,zspl,dlmdltspl,rpd),
                    rtol=1.e-5,  first_step=1.e-5)
    #this was originall 1e-5
    Mouti = sol.y 

    # why was the integration done on such a sparse grid?? Maybe because it is faster?

    gevol_func.sfr = 0.
    
    return Mouti


def integrate_single_stoch_sfr(info):
    ''' 
    Function that returns Mouti for a single MAH
    '''

    t0 = info['t0']
    m0 = info['m0']
    mstar_ini = info['mstar_ini']
    astart = info['astart']
    touti = info['tout']
    params = info['model_params']
    cosmo = info['cosmo']
    zspl = info['zspl']
    dlmdltspl = info['dlmdltspl']
    rpd = info['rpd']

    deltatspl = info['deltatspl']

    cosmo = convert(cosmo)
    params = convert(params)

    zstart = 1. / astart - 1.

    if params.rei_model == "noreion":
        mg0 = cosmo.fbuni * m0

    if params.rei_model == "okamoto":
        mg0 = cosmo.fbuni * m0 * fgcum_uv(m0, zstart, params.beta, params.gamma)[0]

    if params.rei_model == "step":
        mg0 = cosmo.fbuni * m0 * step_fgin(m0, zstart, params.zrei)

    #the starting integration point
    #M200c, Mgas, Mstar, MZg, MZs, Mgin, Mgout
    #the latter two are the quantities for total gas accreted onto halo and total gas driven out in outflows 

    y0 = np.array([m0, mg0, mstar_ini, params.Z_IGM*mg0, 0.1,0.1,0.1]) 
    ##Why is the
    sol = solve_ivp(gevol_func_stoch_sfr, [t0, touti[-1]], y0, t_eval=np.linspace(touti[0],touti[-1],1000), 
                    method='RK45',
                    args=(cosmo,params,zspl,dlmdltspl,rpd,deltatspl),
                    rtol=1.e-5,  first_step=1.e-5)

    Mouti = sol.y 
    
    # update the initialization information for each track
    gevol_func_stoch_sfr.sfr = 0.
    gevol_func_stoch_sfr.sfr_mean = [0.]
    gevol_func_stoch_sfr.t_sfrmean = [0.]

    return Mouti

def integrate_model(tmdlmdlt=None, good_index=None, data=None, zspl=None, 
                    params=None, cosmo=None, iniconf=None):
    '''
    This is the actual integration routine.
    We input in the parameters of the characterisitic mass ,beta and gamma/
    tmdlmdlt is given to us by get_tmdlmdlt function in MAH_fitting.py
    '''

    disable_tqdm = iniconf['run params']['disable_tqdm']
    stochastic_sfr = iniconf['run params']['stochastic_sfr']
   
    if disable_tqdm == "False":
        disable_tqdm = False
    if disable_tqdm == "True":
        disable_tqdm = True

    Mout = []
    #along with the distance from host, we will also record the final x,y,z position of the halos
    Mpeak, zpeak = [], []
    rperturb = [] 
    
    #to ensure that results are reproducible
    np.random.seed(238499)

    times, m200c = data['times'], data['m200c']
    tmdlmdlt = np.array(tmdlmdlt)

    ###prepare the list to be fed into parallel function
    all_info = []

    ## Here is the part with non-stochastic SFR (fiducial)
    if stochastic_sfr == "False":

        deltatspl = None

        for i in range(len(tmdlmdlt)):
            #find index of good MAHs
            ind = good_index[i]
            touti = times[ind][1:]
            m200ci = m200c[ind][1:]
            zi = zspl(touti)

            mpeaki = np.max(m200ci)
            ipeak = np.argmax(m200ci)

            t0, m0, dlmdltspl, astart  = tmdlmdlt[i]

            rpd = np.random.normal(0., scale=params.rpert_sig, size=1)   
        
            temp_dict = {"i":i,"t0":t0, 'm0':m0,'astart':astart,'rpd':rpd,"tout":touti,"mstar_ini":params.mstar_ini,'zspl':zspl, 'dlmdltspl':dlmdltspl,'cosmo':cosmo._asdict(),'model_params':params._asdict() }

            all_info.append(temp_dict)
            Mpeak.append(mpeaki)
            zpeak.append(zi[ipeak])
            rperturb.append(rpd)

        #run the function either in parallel or series
        run_parallel = iniconf['run params']['run_mah_parallel']
        ncores = int(iniconf['run params']['ncores'])

        if run_parallel == "True":
            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
                results = list(tqdm(executor.map(integrate_single,all_info), total = len(all_info)))
            Mout = list(results)
        else:
            #serial version
            Mout = []
            for i in tqdm(range(len(all_info)),disable=disable_tqdm):
                Mouti = integrate_single(all_info[i])
                Mout.append(Mouti)


    if stochastic_sfr == "True":
        #stochastic SFR is True

        deltatspl = []

        for i in range(len(tmdlmdlt)):
            #find index of good MAHs
            ind = good_index[i]
            touti = times[ind][1:]
            #print('final dt (Myr)',(touti[-1]-touti[-2])*1000)
            m200ci = m200c[ind][1:]
            zi = zspl(touti)
            
            mpeaki = np.max(m200ci)
            ipeak = np.argmax(m200ci)

            t0, m0, dlmdltspl, astart  = tmdlmdlt[i]
            
            #rpd = np.float(np.random.normal(0, scale=params.rpert_sig, size=1))   
            rpd = np.random.normal(0, scale=params.rpert_sig, size=1) 
            #### delta for stochastic SFR!
            nsize = 1000 #the smallest fluctuation timescale is 10 Myr,the lifetime of star-forming clouds. nsize=10Gyr/10Myr=1000
            tmin, tmax = times[ind].min(), times[ind].max()

            # sigma_int = 0.6
            tgrid, f, delta, norm = generate_random(nsize=nsize, sigma_int=params.sigma_int, tmin=tmin, tmax=tmax, 
                                                    psd_func=PSD_new, args=[params.tau_break, params.alpha])

            # spline interpolation for time vs delta
            dspl = UnivariateSpline(tgrid,delta,s=0)

            temp_dict = {"i":i,"t0":t0, 'm0':m0,'astart':astart,'rpd':rpd,"tout":touti,"mstar_ini":params.mstar_ini,'zspl':zspl, 'dlmdltspl':dlmdltspl,'cosmo':cosmo._asdict(),'model_params':params._asdict(),'deltatspl':dspl}
            
            all_info.append(temp_dict)
            Mpeak.append(mpeaki)
            zpeak.append(zi[ipeak])
            rperturb.append(rpd)
            deltatspl.append(dspl)

        #run the function either in parallel or series
        run_parallel = iniconf['run params']['run_mah_parallel']
        ncores = int(iniconf['run params']['ncores'])

        if run_parallel == "True":
            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
                results = list(tqdm(executor.map(integrate_single_stoch_sfr,all_info), total = len(all_info)))
            Mout = list(results)
        else:
            #serial version
            Mout = []
            for i in tqdm(range(len(all_info)),disable=disable_tqdm):
                Mouti = integrate_single_stoch_sfr(all_info[i])
                Mout.append(Mouti)

    #the deltatspl object is None if stochastic sfr is None
    return Mout, Mpeak, zpeak, rperturb, deltatspl

#### 
#This is the post processing stuff
####


def post_process(good_index=None,good_files=None,tmdlmdlt=None, data=None, params=None, cosmo=None,
                 zspl=None, Mout=None, Mpeak=None, zpeak=None, rperturb=None,deltatspl=None,iniconf=None):

    '''
    Arguments: 
    good_index = the indices of the halo at the very start that fulfill the conditions. We will need this here because we are going to read in from x,y,z data to get final position
                 the good_index elements are in one-to-one correspondance with the Mout elements
    tmdlmdlt = the information of MAH slopes obtained from MAH_fitting.py
    data = dictionary of all the info that was required earlier as well. 
    Mout,Mpeak,zpeak, beta,gamma
    Returns: Dictionary of final masses (z = 0): Mh, Ms, Mg, MZs, MZg, MH2. Mpeak, zpeak, radial distance
    Notes: This function will also write some files to the temp folder. Those folders are useful for fsps calculation
    '''

    run_fsps = iniconf['run params']['run_fsps']   
    disable_tqdm = iniconf['run params']['disable_tqdm']
    if disable_tqdm == "False":
        disable_tqdm = False
    if disable_tqdm == "True":
        disable_tqdm = True

    stochastic_sfr = iniconf['run params']['stochastic_sfr']
    
    track_dir = iniconf['output paths']['final_output_dir']

    times, m200c = data['times'],data['m200c']

    mah_names = data['mah_name']

    if len(good_index) != len(Mout) or len(tmdlmdlt) != len(Mout) or len(mah_names) != len(times) or len(good_files) != len(Mout):
        print( len(good_index),len(Mout),len(tmdlmdlt),len(mah_names),len(good_files) )
        raise ValueError("Post Process: Array length issue")

    rvir, rs, mvir, vmax = data['rvir'], data['rs'], data['mvir'], data['vmax']
    upID, ID = data['upID'], data['ID']
    xc, yc, zc = data['x'], data['y'], data['z']
    vxc, vyc, vzc = data['vx'], data['vy'], data['vz']


    Mhf, Msf, Mgf, MZsf, MZgf, MH2f, MHIf = [],[],[],[],[],[],[]
    Mpeak_new, zpeak_new, rstar_ave = [],[], []
    sfrf = [] #this is the star formation rate at the final moment
    sfrf_inst = [] #this is the instantaneous star formation rate at the final moment

    tau_50s, tau_90s = [],[]

    rvir_f = []
    rs_f = []
    mvir_f = []
    m200c_f = []
    vmax_f = []

    xpos_f = []
    ypos_f = []
    zpos_f = []

    vx_f = []
    vy_f = []
    vz_f = []

    upID_f, ID_f = [],[]

    rperturb_f = []

    all_final_names=[]

    all_zstarts = []

    sub_good_ind = []
    sfh_names = []

    for i in tqdm(range(len(Mout)),disable=disable_tqdm):
        Mouti = Mout[i]
        ind = good_index[i]
        touti, m200ci = times[ind][1:], m200c[ind][1:]
        mhi_name = mah_names[ind]
        #the corresponding good_file name is. This list is also made from mah_names and should have the same content and a one to one matching
        good_file_name_i = good_files[i]
        if mhi_name != good_file_name_i:
            raise ValueError("Post Process: The file names do not match!")

        ##I need to confirm this is the right way to do this

        rviri, rsi, mviri, upIDi, IDi = rvir[ind][1:], rs[ind][1:], mvir[ind][1:], upID[ind][1:], ID[ind][1:]
        xci, yci, zci = xc[ind][1:], yc[ind][1:], zc[ind][1:]
        vmaxi = vmax[ind][1:]
        vxi, vyi, vzi = vxc[ind][1:], vyc[ind][1:], vzc[ind][1:] 

        Mhi, Mgi, Msi, MZgi, MZsi = (Mouti[0,:], Mouti[1,:], Mouti[2,:], Mouti[3,:], Mouti[4,:])
        Mgin_i = Mouti[5,:]
        Mgout_i = Mouti[6,:]

        t0, m0, dlmdltspl, astart = tmdlmdlt[i]
        zstart = 1. / astart - 1.
        rpd = rperturb[i]
        zi = zspl(touti)

        if stochastic_sfr == "True":
            dspl = deltatspl[i]
            args = [cosmo,params,zspl,dlmdltspl,rpd,dspl]
        else:
            args = [cosmo,params,zspl,dlmdltspl,rpd]

        if np.min(Mhi) > 0 and np.isnan(np.sum(Mhi)) == False: 
            sub_good_ind.append(ind) 

            #the virial variables at z = 0 time.
            mvir_f.append(mviri[-1])
            rvir_f.append(rviri[-1]) 
            rs_f.append(rsi[-1])
            m200c_f.append(m200ci[-1])
            vmax_f.append(vmaxi[-1])
            xpos_f.append(xci[-1])
            ypos_f.append(yci[-1])
            zpos_f.append(zci[-1])

            vx_f.append(vxi[-1])
            vy_f.append(vyi[-1])
            vz_f.append(vzi[-1])

            rperturb_f.append(float(rpd))

            ID_f.append(IDi[-1])
            upID_f.append(upIDi[-1])

            all_final_names.append(mhi_name)

            all_zstarts.append(float(zstart))

            dummy1 = []

            rshalf = 0.

            # estimate rhalf roughly
            rh = float(rhalf(Mpeak[i], zpeak[i], cosmo.h23i, cosmo.Om0, cosmo.OmL) * 10.**rperturb[i])
            #rh is Rd (eq 19) estimate roughly from the Mpeak and zpeak of halo.
            #this will provide us a grid on which we will evaluate the H2 profiles to get half mass radius

            area2 = 2.*np.pi*(1e3*rh)**2
            Sigma0 = Mgi[-1] / area2
            if Sigma0 > params.SigHIth:
                xHI = -np.log(params.SigHIth / Sigma0)
                MHI = Mgi[-1] * (1. - np.exp(-xHI) * (1. + xHI))
            else:
                MHI = 1.e-100

            MHIf.append(MHI)

            # rh2g = np.linspace(np.log10(rdmin*rh),np.log10(3*rh), 100) 
            # rh2g = np.logspace(np.log10(rdmin*rh),np.log10(3*rh), 100) 
            rh2g = np.linspace(rdmin*rh,3*rh, 100) 

            #We want linear spacing to compute the radial metallicity gradient as that is easiest
            rh2g_cens = 0.5*(rh2g[1:] + rh2g[:-1])

            #this is the 2d array in which the H2 profiles at each time step will be saved
            mh2_matrix = []
            new_metals = []

            MH2g = np.zeros_like(rh2g)
            #print(MH2g.shape, rh2g.shape, rh, rdmin, rh2g)

            Mhf.append(Mhi[-1]),Msf.append(Msi[-1]); MZsf.append(MZsi[-1])
            Mgf.append(Mgi[-1]); MZgf.append(MZgi[-1])

            Mpeak_new.append(Mpeak[i])
            zpeak_new.append(zpeak[i])

            ##we need to also compute the time evolution of the radius 
            rshalf_evos = [] #this should have the same length as tout for this halo
            sfr_evos = []

            if stochastic_sfr == "True":
                sfr_ave = []
                gevol_to_use = gevol_func_stoch_sfr
                time_array_i = np.linspace(touti[0],touti[-1],1000) 
                redshifts_i =  zspl(time_array_i)
            else:
                gevol_to_use = gevol_func
                # time_array_i = touti
                time_array_i = np.linspace(touti[0],touti[-1],1000) 
                # redshifts_i = zi
                redshifts_i =zspl(time_array_i)
            ## as the main integration function is different if it is stochastic SFR or not, 
            
            if iniconf["run params"]["run_size_model"] == "False":
                rshalf_evos = np.ones(len(time_array_i))*(-99)
                sfr_ave = np.ones(len(time_array_i))*(-99)
                sfr_evos = np.ones(len(time_array_i))*(-99)
                dummy1 = np.ones(len(time_array_i))*(-99)
                
                #compute the H2 mass in the last step
                dMdtvec = gevol_to_use(time_array_i[-1],[Mhi[-1], Mgi[-1], Msi[-1], MZgi[-1], MZsi[-1], Mgin_i[-1],Mgout_i[-1]],*args)
                sfr = dMdtvec[2] / params.Rloss1
                #Remember that the MH2 function needs the Z_IGM parameter in units of Zsun!
                M_H2, rstar, rsout, sgsp, sgsp_gas,  xHI_rad = MH2(Mh=Mhi[-1],Mg=Mgi[-1],MZg=MZgi[-1], sfr=sfr, z=redshifts_i[-1], h23i=cosmo.h23i, Om0=cosmo.Om0, OmL=cosmo.OmL, 
                                                Zsun=params.Zsun, Z_IGM =params.Z_IGM/params.Zsun,SigHIth=params.SigHIth,irstar=1, rpert=rperturb[i],h2_model = params.h2_model)

                dummy1[-1] = M_H2

            else:
                #loop over time to compute the time evolution of the size
                for j, td in enumerate(time_array_i): 
                    ## touti is the times array which is taken from the halo MAH
                    ## however, when we have stochastic SFR, we have a different much denser time array

                    ##NEED TO THINK ABOUT THE BELOW, RIGHT NOW THE THE GEVOL FUNCTION cannot take in pre-computed Mgin and Mgout
                    ##This will significantly speed up the process I think? As they are already computed ... 

                    dMdtvec = gevol_to_use(td,[Mhi[j], Mgi[j], Msi[j], MZgi[j], MZsi[j], Mgin_i[j],Mgout_i[j]],*args)
                    sfr = dMdtvec[2] / params.Rloss1
                    sfr_evos.append(sfr)
                    if stochastic_sfr == "True":
                        if j != 0:
                            #SFR in Msun / Gyr
                            sfr_ave.append( (Msi[j]-Msi[j-1])/(time_array_i[j]-time_array_i[j-1])/params.Rloss1 )
                        else:
                            sfr_ave.append(0)
                    
                    #Remember that the MH2 function needs the Z_IGM parameter in units of Zsun!

                    M_H2, rstar, rsout, sgsp, sgsp_gas,  xHI_rad = MH2(Mh=Mhi[j],Mg=Mgi[j],MZg=MZgi[j], sfr=sfr, z=redshifts_i[j], h23i=cosmo.h23i, Om0=cosmo.Om0, OmL=cosmo.OmL, 
                                                    Zsun=params.Zsun, Z_IGM =params.Z_IGM/params.Zsun,SigHIth=params.SigHIth,irstar=1, rpert=rperturb[i],h2_model = params.h2_model)

                    dummy1.append(M_H2)
                    #dummy1 is the evolution of MH2 mass

                    # ### CALCULATIONS FOR COMPUTING RADIAL METALLICITY PROFILE

                    # #initializing array of zeros to save H2 profile in
                    # MH2_profile = np.zeros(len(rh2g_cens))

                    # # accumulate cumulative MH2 profile by adding current instantaneous MH2(<r) profile
                    # # we want differential profile in each bin and not cumulative profile, hence we compute difference below
                    # for m in range(len(MH2_profile)):
                    #     # MH2_profile[m] += 2*np.pi*sgsp.integral(rdmin*rh, 10.**rh2g[m+1]) -  2*np.pi*sgsp.integral(rdmin*rh, 10.**rh2g[m])
                    #     # MH2_profile[m] += 2*np.pi*sgsp.integral(rdmin*rh, rh2g[m+1]) -  2*np.pi*sgsp.integral(rdmin*rh, rh2g[m])
                    #     MH2_profile[m] += 2*np.pi*sgsp.integral( rh2g[m], rh2g[m+1])

                    # if j > 0:
                    #     dMs = np.maximum(0., Msi[j] - Msi[j-1])
                    #     dMZs = MZsi[j] - MZsi[j-1] 
                    #     ssp_metal = dMZs/dMs
                    # else:
                    #     ssp_metal = 1e-5

                    # #There is no H2 gas beyond xHI. To what radius is xHI normalized to?
                    # MH2_profile[rh2g_cens > xHI_rad] = 0
                    # # mh2_matrix.append(MH2_profile)
                    # # new_metals.append(ssp_metal)
                    # #this stores the list of the metallicities of the new SSPs that are forming

                    ### CALCULATIONS FOR COMPUTING STELLAR HALF MASS RADIUS

                    # accumulate cumulative MH2 profile by adding current MH2(<r) profile (from each time step)
                    for k, rh2d in enumerate(rh2g):
                        #oh we should only integrate to xHI_rad
                        max_rad = np.minimum(xHI_rad, rh2d)
                        MH2g[k] += 2*np.pi*sgsp.integral(rdmin*rh,max_rad)
                        # MH2g[k] += 2*np.pi*sgsp.integral(rdmin*rh,10.**rh2d)
                        # MH2g[k] += 2*np.pi*sgsp.integral(rdmin*rh,rh2d)


                        # #do we need this??
                        #note that we are computing a cumulative distribution so even for radii that are beyond the xHI_rad, we need to evaluate the integral till xHI_rad
                        #so that the cumulative distribution stays flat
                        # #rh is not in log space, but rh2g array is in log space hence 10**rh2d
        
                    #print(MH2g.shape, MH2g)
                    #this is computing the evolution of rshalf
                    #MH2 is the cumulative distribution of 
                    MH2tot_tt = MH2g[-1]
                    if MH2tot_tt > 0:
                        MH2half_tt = 0.5*MH2tot_tt
                        ihalf_tt = np.argwhere(MH2g < MH2half_tt).flatten()
                        rshalf_tt = rh2g[ihalf_tt[-1]]
                        # rshalf_tt = 10.**rh2g[ihalf_tt[-1]]

                        #print('ihtt:', ihalf_tt, MH2tot_tt) 
                    else:
                        rshalf_tt = 0

                    rshalf_evos.append(float(rshalf_tt))

            #### end of computation of sizes+MH2 and their evolution

            rshalf_evos = np.array(rshalf_evos)

            if stochastic_sfr == "True":
                sfr_evos = np.array(sfr_evos)
                # instantaneous SFR at present time (final value)
                sfrf_inst.append(sfr_evos[-1]*1e-9) # convert it to solar mass/yr

            #print("rshalf_evos:", rshalf_evos)

            MH2f.append(dummy1[-1])

            # what is the below stuff used for??
            MH2tot = MH2g[-1]
            if MH2tot > 0.:
                MH2half = 0.5 * MH2tot
                ihalf = np.ravel(np.nonzero(MH2g < MH2half))
                # rshalf = 10.**rh2g[ihalf[-1]]
                rshalf = rh2g[ihalf[-1]]

            else:
                rshalf = 0
            
            if run_fsps == "True":
                #use file name to the 
                entire_name = 'fsps_' + mhi_name

                fsps_input_dir = iniconf['output paths']['fsps_input_dir']
                entire_path = fsps_input_dir + '/' + entire_name +'.dat'

                f = open(entire_path, "w")
                sfh_names.append(entire_name)

                for j, td in enumerate(touti[:-1]): 
                    # output SFH for FSPS with stellar metallicity evolution
                    dt = touti[j+1] - td
                    tjave = td + 0.5*dt
                    dMs = np.maximum(0., Msi[j+1] - Msi[j])
                    dMZs = MZsi[j+1] - MZsi[j] 
                    dummys = np.maximum(dMs, 1.) #dummy
                    dummyz = np.maximum(dMZs, 0.) 
                    #SFR in Msun/yr (so un-noramlized SFH) and metallicty of SSP (aka new stellar population)
                    f.write("%.3f  %.3g   %.3g\n"%(tjave, 1.e-9*dMs/dt/params.Rloss1,dummyz/dummys))
     
                f.close()

            #Keep in mind that this is the 3D half-mass radisus ~ 3D half-light radius .
            rstar_ave.append(rshalf) 
            sfrf.append( 1.e-9*(Msi[-1]-Msi[-2])/(time_array_i[-1]-time_array_i[-2])/params.Rloss1 ) #this the star formation rate SFR at z = 0 

            #using the evolution of Ms, compute tau_50 and tau_90
            norm_Ms = Msi/np.max(Msi)
            if stochastic_sfr=="True":
                # t_eval=np.linspace(touti[0],touti[-1],1000)
                tau_50_i = np.interp(0.5,norm_Ms,time_array_i ) #used to be t_eval
                tau_90_i = np.interp(0.9,norm_Ms,time_array_i )
            else:
                tau_50_i = np.interp(0.5,norm_Ms,time_array_i )
                tau_90_i = np.interp(0.9,norm_Ms,time_array_i )

            tau_50s.append(tau_50_i)
            tau_90s.append(tau_90_i)

            if stochastic_sfr=="True": 
                #THIS ONLY USED TO BE WHEN TRUE
                #The integration was done on a much sparser grid than touti. So for purposes of these tracks we find the values 
                #closest to touti from the integration output array
                #The full SFR evolution is saved in a different array!

                differences = np.abs(time_array_i[:, np.newaxis] - touti)
                min_indices = np.argmin(differences, axis=0)

                dummy1 = np.array(dummy1)

                halo_evol = {'t': touti.tolist(), 'z_redshift': zi.tolist() ,'Mvir': mviri.tolist(),
                            'M200c': m200ci.tolist(),'Mh': Mhi[min_indices].tolist(), 'Mg': Mgi[min_indices].tolist(), 'Ms': Msi[min_indices].tolist(),
                            'MZg': MZgi[min_indices].tolist(), 'MZs': MZsi[min_indices].tolist(), 'MH2': dummy1[min_indices].tolist(),
                            'rsize': rshalf_evos[min_indices].tolist(), 'rvir': rviri.tolist(), 'rs': rsi.tolist(),
                            'upID': upIDi.tolist(), 'ID': IDi.tolist(),
                            'xpos': xci.tolist(), 'ypos': yci.tolist(), 'zpos': zci.tolist(),
                            'vx': vxi.tolist(),' vy': vyi.tolist(), 'vz': vzi.tolist(),
                            'Mgin': Mgin_i[min_indices].tolist(), 'Mgout': Mgout_i[min_indices].tolist(), "sfr" : np.array(sfr_evos)[min_indices] , "sfr_ave" : np.array(sfr_ave)[min_indices]}

                halo_dense_evol = {"t" : time_array_i , "Mh": Mhi.tolist(), "Mg": Mgi.tolist(),"Ms": Msi.tolist(), "MZg": MZgi.tolist() , 
                                    "MZs" : MZsi.tolist(), "MH2": dummy1, "rsize": rshalf_evos.tolist()  , "sfr": sfr_evos , "sfr_ave": sfr_ave, 
                                    "Mgin": Mgin_i.tolist(), "Mgout":Mgout_i.tolist()}

                halo_dense_evol_df = pd.DataFrame(halo_dense_evol )
                dense_track_name = track_dir + "/track_data/" + mhi_name + '_track_stoch_sfr.csv'
                halo_dense_evol_df.to_csv(dense_track_name, index=False )

            else:
            #we were originally only doing the above matching if stochastic, but now going to try what happens when doing

                differences = np.abs(time_array_i[:, np.newaxis] - touti)
                min_indices = np.argmin(differences, axis=0)

                dummy1 = np.array(dummy1)

                halo_evol = {'t': touti.tolist(), 'z_redshift': zi.tolist() ,'Mvir': mviri.tolist(),
                            'M200c': m200ci.tolist(),'Mh': Mhi[min_indices].tolist(), 'Mg': Mgi[min_indices].tolist(), 'Ms': Msi[min_indices].tolist(),
                            'MZg': MZgi[min_indices].tolist(), 'MZs': MZsi[min_indices].tolist(), 'MH2': dummy1[min_indices].tolist(),
                            'rsize': rshalf_evos[min_indices].tolist(), 'rvir': rviri.tolist(), 'rs': rsi.tolist(),
                            'upID': upIDi.tolist(), 'ID': IDi.tolist(),
                            'xpos': xci.tolist(), 'ypos': yci.tolist(), 'zpos': zci.tolist(),
                            'vx': vxi.tolist(),' vy': vyi.tolist(), 'vz': vzi.tolist(),
                            'Mgin': Mgin_i[min_indices].tolist(), 'Mgout': Mgout_i[min_indices].tolist(), "sfr" : np.array(sfr_evos)[min_indices]}


                # halo_evol = {'t': touti.tolist(), 'z_redshift': zi.tolist() ,'Mvir': mviri.tolist(),
                #             'M200c': m200ci.tolist(),'Mh': Mhi.tolist(), 'Mg': Mgi.tolist(), 'Ms': Msi.tolist(),
                #             'MZg': MZgi.tolist(), 'MZs': MZsi.tolist(), 'MH2': dummy1,
                #             'rsize': rshalf_evos.tolist(), 'rvir': rviri.tolist(), 'rs': rsi.tolist(),
                #             'upID': upIDi.tolist(), 'ID': IDi.tolist(),
                #             'xpos': xci.tolist(), 'ypos': yci.tolist(), 'zpos': zci.tolist(),
                #             'vx': vxi.tolist(),' vy': vyi.tolist(), 'vz': vzi.tolist(),
                #             'Mgin': Mgin_i.tolist(), 'Mgout': Mgout_i.tolist(), "sfr" : sfr_evos}

            halo_evol_df = pd.DataFrame(halo_evol)
            #we will add the magnitudes to this evolution table after fsps is run
             
            track_name = track_dir + "/track_data/" + mhi_name + '_track.csv'

            halo_evol_df.to_csv(track_name, index=False )

            # ##### compute the radial profile of H2 at different times for each galaxy
            # #should I do 1d interpolation or spl?
            # # time_evo = t_eval = np.linspace(touti[0],touti[-1],1000)

            # #there was sfh_names here, was that just a mistaken copy paste?
            # mh_spl = interpolate.interp1d(time_array_i, Mhi)
            # ms_spl = interpolate.interp1d(time_array_i,Msi)
            # mg_spl = interpolate.interp1d(time_array_i,Mgi)
            # mgz_spl = interpolate.interp1d(time_array_i,MZgi)

            # print(len(Mpeak), i)

            # mh2_matrix, rad_cens = get_radial_profile(mh_spl = mh_spl, ms_spl = ms_spl, mg_spl=mg_spl, mgz_spl=mgz_spl, 
            #                                             time_steps=time_array_i, Mpeak_val = Mpeak[i], zpeak_val = zpeak[i], rperturb=rperturb[i], zspl =zspl, params=params,cosmo=cosmo)

            # #we will save this in the track info folder in a different file. 
            # mh2_matrix = np.array(mh2_matrix)
            # all_info = np.vstack([rh2g_cens, mh2_matrix])
            # h2_prof_name = track_dir + "/track_data/" + mhi_name + '_h2_profile.npy'
            # np.save(h2_prof_name , all_info)

            # ##here we are saving the gas profile for debugging purpose
            # new_metals = np.array(new_metals)
            # metal_name = track_dir + "/track_data/" + mhi_name + '_metals.npy'
            # np.save(metal_name , new_metals)


    #construct dictionary of the final statistics
    hdists = np.sqrt(np.array(xpos_f)**2 + np.array(ypos_f)**2 + np.array(zpos_f)**2)*1e3
    summary_stats = {'file_name': all_final_names,' Mvir': mvir_f, 'M200c': m200c_f, 
                     'Vmax': vmax_f, 'Mh': Mhf,  
                     'Ms': Msf, 'Mg' : Mgf, 'MZs': MZsf , 'MZg': MZgf, 'MH2': MH2f,
                     'MHI': MHIf, 'Mpeak': Mpeak_new ,'zpeak': zpeak_new ,
                     'rsize': rstar_ave,'rvir': rvir_f, 'rs': rs_f,'sfrf': sfrf, 'zstart': all_zstarts,
                     'xpos': xpos_f ,'ypos': ypos_f , 'zpos': zpos_f, 'vx': vx_f, 'vy': vy_f, 'vz': vz_f,
                     'hdist': hdists, 'tau_50': tau_50s, 'tau_90': tau_90s,
                     'rperturb': rperturb_f, 'upID': upID_f, 'ID': ID_f}

    if iniconf['run params']['run_fsps'] == "True": 
        #if fsps is yes, then save the sfh_names into a text file 
        temp_path = iniconf['output paths']['fsps_input_dir'][:-5] + "/temp.txt"
        ft = open(temp_path,"a")
        for i, sfhi in enumerate(sfh_names): 
            ft.write(sfhi + "\n")

        ft.close()

    return summary_stats,sub_good_ind



