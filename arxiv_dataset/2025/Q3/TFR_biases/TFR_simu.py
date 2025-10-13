"""
Functions to simulate TFR catalogs
"""
import numpy as np
import pandas as pd
from skimage.restoration import richardson_lucy
from scipy.special import erf

""" Sampling and catalog simulations """
def phi(lgx, alpha):
    # Schechter function: dn/dlogL ~ (L/L*)^(alpha+1) exp(-L/L*)
    # lgx = logM-logM* = b logV - b logV*
    return 10.**((alpha+1) * lgx) * np.exp(-10.**lgx)

def schechter_sample(alpha, size=100, lgx_min=-3.5, lgx_max=1.5, resolution=1001, sigx=0.0):
    """ 
    Generate a random sample of log(L/L*) following the Schechter function, phi(logL), 
    using inverse transform sampling method [1,2]. One key feature of this function is that it 
    deconvolves the expected measurement errors in logL/L* (sigx) from the input Schechter 
    function before drawing the random sample. This ensures the resulting distribution matches
    the input Schechter function after adding random Gaussian errors. 

    Schechter function: 
        phi(logL) = ln(10) phi* 10^[(logL-logL*)(a+1)] exp(-10^(logL-logL*))
    in logarithmic:
        log(phi(logL)) = log(ln(10) phi*) + (a+1)(logL-LogL*) - 10^(logL-logL*) log(e)

    Parameters
    ----------
    alpha : float
        The faint-end slope of the Schechter function
        logL* is not an input parameter because the output luminosity array is in log(L/L*)
    size: int, default: 100
        Size of the output sample. 
    lgx_min, lgx_max : float, default: -3.5, +1.5
        Lower and upper bounds for log(L/L*). 
    resolution : int, default: 1001
        Resolution of the cumulative distribution function that is interpolated to 
        perform the inverse transform sampling.
    sigx: float, default: 0.0
        scale of Gaussian error to be deconvolved from the given Schechter function before sampling

    Returns
    -------
    lgx_sample : array_like
        log(L/L*) for samples drawn from the Schechter function.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Inverse_transform_sampling
    [2] https://github.com/skypyproject/skypy/blob/main/skypy/utils/random.py
    """

    # requested range and resolution
    lgx = np.linspace(lgx_min, lgx_max, resolution)
    
    # calculate the probability density function (PDF) from Schechter function
    if sigx > 0: # deconvolve the Schechter function 
        # expand the range to avoid deconvolution artifacts
        lgx2 = np.linspace(lgx_min-10*sigx, lgx_max+3*sigx, 2*resolution)
        dx = lgx2[1]-lgx2[0]
        # Schechter function: 
        #   phi(logL) = ln(10) phi* 10^[(logL-logL*)(a+1)] exp(-10^(logL-logL*))
        pdf = phi(lgx2,alpha) # 10**((alpha+1)*lgx2) * np.exp(-10**lgx2)
        # deconvolve the Schechter function by a Gaussian kernel
        psf_x = np.arange(0,3*sigx,dx) # ensures symmetry
        psf_x = np.concatenate((-1*np.flip(psf_x[1:]),psf_x))
        psf = np.exp(-psf_x**2 / (2 * sigx**2))
        pdf_deconv = richardson_lucy(pdf,psf,num_iter=50,clip=False)
        # interpolate back to the requested range
        pdf = np.interp(lgx, lgx2, pdf_deconv)
    else: # no deconvolution
        pdf = phi(lgx,alpha) # 10**((alpha+1)*lgx) * np.exp(-10**lgx)

    # calculate cumulative distribution function (CDF) by integrating PDF
    cdf = np.copy(pdf) 
    np.cumsum((pdf[1:]+pdf[:-1])/2*np.diff(lgx), out=cdf[1:])
    # normalize CDF
    cdf[0] = 0
    cdf /= cdf[-1]

    # inverse transform sampling from the CDF
    u = np.random.uniform(low=0.0,high=1.0,size=size)
    lgx_sample = np.interp(u, cdf, lgx)
    
    # output log(L/L*) array
    return lgx_sample 

def fz(z, q0=-0.53, j0=1.0):
    """ Taylor expansion term f(z|q0,j0) from Caldwell04 Eq 5 to calculate 
    Luminosity distance from Hubble's law at low z (z << 1)
        DL = cz/H0 * f(z|q0,j0)
    for Omega_m,0 = 0.315, Omega_l,0 = 0.685 (Planck 2018), we have
        q0 = Omega_m,0/2 - Omega_l,0 = -0.53 # cosmological deceleration parameter
        j0 = Omega_m,0 + Omega_l,0 = 1.0 # cosmological jerk parameter
    """
    return 1 + 0.5*(1-q0) * z - (1./6.)*(1-q0-3*q0**2+j0) * z**2

def simu_cat_TFR(czmin,czmax,czbin,
            nn=-1, oversample=1,
            sigm=0.15, sigw=0.045, sigcz=0, 
            beta=3.33, gamma=10.5,
            v_star=0.3, alpha=-1.27,
            ml=5.736):
    """ Simulate a Tully-Fisher galaxy sample for all generative models of the TFR 
    
    Parameters
    ----------
    czmin / czmax / czbin: redshift range and bin size in km/s
    nn : powerlaw index of integrated galaxy volume density vs distance 
        (default: 0 for constant density at all distances)
    oversample : constant factor to change the output sample size

    sigm : intrinsic scatter and measurement combined dispersion in log mass 
        (set to zero for inverse TFR)
    sigw : intrinsic scatter and measurement combined dispersion in log velocity 
        (set to zero for forward TFR)
    sigcz : velocity noise in km/s (default: 0 km/s)

    beta/gamma : BTFR slope and fiducial log mass at log V = 2.5
    
    v_star : characteristic velocity times TFR slope (M_star = v_star + BTFR_M0)
    alpha : faint-end slope of the velocity function (default: M_star=10.8, alpha=-1.27)
    
    ml : survey sensitivity limit in log apparent mass
        if a constant, the selection function is a step function  
        if a two-element list, the selection function is an erf function (ml[0]: mean, ml[1]: width)

    Returns
    -------
    sim : Panda data frame of the simulated catalog

    Tags
    -------
    cz : cosmological redshift
    d_int: 2x log luminosity distance from cz
    Vcmb: observed redshift incl. random Gaussian peculiar velocity (sigcz)
    d: 2x log luminosity distance from Vcmb
    logV_int: edge-on rotation velocity
    logMb_int: BTFR-predicted baryonic mass in log
    logmb_int: logMb_int - d_int
    logmb: apparent baryonic mass incl. random Gaussian scatter (sigm)
    sini: sine of the inclination angle
    logW_int: projected rotation velocity
    logW: projected rotation velocity incl. random Gaussian scatter (sigw)
    W

    Notes on Survey Volume
    -------------
    Because survey volume between cz and cz+dcz is 
        dV = theta^2 cz^2 dcz / H0^3
    if the integrated volume density is flat
        n = dN/dV ~ cz^nn, where nn = 0 
    the number of galaxies in each linear cz bin increases as 
        dN/dcz = dN/dV dV/dcz ~ cz^(2+nn)
    and in each log cz bin as:
        dN/dlog(cz) ~ cz^(3+nn)
    
    Since the sample gets truncated at higher masses at higher cz, 
    the actually observed sample grow slower than dN/dcz ~ cz^(2+nn). 
    By setting nn=-1, one effectively achieves a flat dN/dcz distribution.
    """

    """ cosmology parameters """
    H0 = 70.0 # assumed Hubble constant in km/s/Mpc
    sol= 299792.458 # speed of light

    """ loop through redshift shells """
    czctrs = np.arange(czmin,czmax,czbin)
    for czctr in czctrs:
        # intial sample size
        n0 = int(oversample * (czctr/1600.0)**(2+nn) * czbin)

        """ intrinsic and observed redshift """
        # true cosmological redshift
        cat = pd.DataFrame({'cz':np.full(n0,czctr)})
        # true distance parameter, d := 2 log DL/Mpc
        cat['d_int'] = 2*np.log10(cat['cz']/H0 * fz(cat['cz']/sol))
        # add velocity noise to true redshift
        if sigcz > 0:
            cat['Vcmb'] = cat['cz'] + np.random.normal(scale=sigcz,size=n0)
            cat['d'] = 2*np.log10(cat['Vcmb']/H0 * fz(cat['Vcmb']/sol))
        else:
            cat['Vcmb'] = cat['cz']
            cat['d'] = cat['d_int']

        """ intrinsic b*logV sampled from a Schechter function """
        logX = schechter_sample(alpha, lgx_min=-3.5, lgx_max=1.5, size=n0)
        cat['logV_int'] = (logX + v_star)/beta + 2.5

        """ intrisic mass from TFR, this linear relation makes sampling 
        the velocity function equivalent to sampling the mass function """
        cat['logMb_int'] = gamma + beta*(cat['logV_int'] - 2.5)

        """ intrinsic standard mass at 1 Mpc """
        cat['logmb_int'] = cat['logMb_int'] - cat['d_int']

        """ add scatter to standard mass at 1 Mpc """
        if sigm > 0:
            cat['logmb'] = cat['logmb_int']+np.random.normal(scale=sigm,size=n0)
        else:
            cat['logmb'] = cat['logmb_int']

        """ add random orientation to edge-on velocity """
        # random orientation on the sky (flat PDF in cos i)
        cat['sini'] = np.sqrt(1.0-np.random.uniform(size=n0)**2)
        # projected line widths, including sin i, and measurement errors
        cat['logW_int'] = cat['logV_int'] + np.log10(cat['sini'])

        """ add scatter to edge-on logW """
        if sigw > 0: 
            cat['logW'] = cat['logW_int'] + np.random.normal(scale=sigw,size=n0)
        else: 
            cat['logW'] = cat['logW_int']
        # save linear velocity width
        cat['W'] = 10**cat['logW'] 

        """ apply mass limit """
        if isinstance(ml, float): # step function
            idx = (cat['logmb'] >= ml) 
        elif isinstance(ml, list): # erf function
            idx = (np.random.uniform(size=n0) < (1+erf((cat['logmb']-ml[0])/np.sqrt(2)/ml[1]))/2) 
        # need to reset index to pick the first # of rows
        cat = (cat.loc[idx, :]).reset_index()

        """ save to final catalog """
        if czctr == czctrs[0]: 
            sim = cat
        else:
            sim = pd.concat([sim, cat])

    return sim