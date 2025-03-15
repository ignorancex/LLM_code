import numpy as np

def compute_teq(a, rstar, tstar):
    """ 
    Calculate equilibrium temperature of a planet
    using equation (1) of Enoch et al 2011,
    assuming circular orbit (e=0)
    Teq = Tstar (Rstar/2a)^(1/2)

    Args:
        a     (AU)   : semimajor axis
        rstar (Rsol) : stellar radius
        tstar (K)    : stellar temperature

    Returns:
        equilibrium temperature (K)
    """
    # a     = np.array(a)
    # tstar = np.array(tstar)
    # rstar = np.array(rstar)
    
    # cgs units
    AU    = 150.    *10**(11)        # cm
    Rsun  = 6.956   *10**5  * 10**5  # cm
    
    teq = tstar * ( ( rstar*Rsun / (2.*a*AU) )**(1/2.) ) 
    
    return teq

def compute_stellar_luminosity(rstar, tstar):
    '''
    Calculates stellar luminosity.

    Args:   
        rstar (Rsol) : stellar radius
        tstar (K)    : stellar temperature

    Returns: 
        Lstar (Lsol) :   stellar luminosity
    '''
    # convert to cgs units
    Lsun         = 3.846    *10**(33)        # erg s-1
    Rsun         = 6.956    *10**5  * 10**5  # cm
    a            = 7.57     *10**(-15)      # erg cm-3 K-4
    c            = 2.997925 *10**(10)       # cm s-1
    sigma_planck = a*c/4.                   # erg s-1 cm-2 K-4

    return ( 4 * np.pi * ( rstar * Rsun )**2  * sigma_planck * tstar ** 4 ) / Lsun


def lumi_to_epsilon(lumi, radius, semimajor, Tstar, Rstar):
    ''' 
    Convert Lumi to epsilon using
    L = epsilon * sigma * Tirr**4 * pi * rp**2
    where Tirr = Tstar (Rstar/2a)^(1/2)

    Args:
        lumi  (LJ)         : planet internal/intrinsic luminosity
        radius (RJ)        : planet radius
        semimajor (AU)     : planet semamajor axis 
        Tstar (K)          : stellar temperature
        Rstar (Rsol)       : stellar radius

    Returns:
        epsilon (%) : heating efficiency 
    '''
    # always cgs units ;)
    a            = 7.57     *10**(-15) # erg cm-3 K-4
    c            = 2.997925 *10**(10)  # cm s-1
    sigma_planck = a*c/4.              # erg s-1 cm-2 K-4
    pi           = 3.141592654

    Lsun  = 3.846   *10**(33)            # erg s-1
    LJ    = 8.710   *10**(-10) * Lsun    # erg s-1
    
    rj    = 7.14    *10**4  *10**5   # cm
    AU    = 150.    *10**(11)        # cm
    Rsun  = 6.956   *10**5  * 10**5  # cm
    
    # convert parameters
    aplanet   = semimajor * AU    # cm
    radius    = radius    * rj    # cm
    lumi      = lumi      * LJ    # erg s-1
    Rstar     = Rstar     * Rsun  # cm

    epsilon_num  = lumi * aplanet**2
    epsilon_deno = sigma_planck * pi * radius**2 * Tstar**4 * Rstar**2 

    epsilon = epsilon_num / epsilon_deno * 100

    return epsilon

def stellarparams_completo(Mstar, semi):
    ''' 
    Calculate Teq and Lstar 
    given Mstar and semimajor axis 
    as calculated using the scaling relations in COMPLETO.

    This function is used given samples from MCMC. 

    Args:
        Mstar (Msol) -- usually samples from MCMC
        semi (AU)    -- observed value
    Returns:
        Teq (K) and Lstar (Lsol)
    '''


    # always cgs units ;)
    a            = 7.57     *10**(-15) # erg cm-3 K-4
    c            = 2.997925 *10**(10)  # cm s-1
    sigma_planck = a*c/4.              # erg s-1 cm-2 K-4
    pi           = 3.141592654
    Lsun  = 3.846   *10**(33)            # erg s-1
    Rsun  = 6.956   *10**5  * 10**5  # cm
    AU    = 150.    *10**(11)        # cm

    Lstar = np.zeros(shape = len(Mstar) )
    Tstar = np.zeros(shape = len(Mstar) )
    Rstar = np.zeros(shape = len(Mstar) )

    idx = np.where( Mstar < 0.7)[0]
    Lstar[idx] = 0.628*(Mstar[idx])**2.62
    
    idx = np.where( ~ (Mstar < 0.7))[0]
    Lstar[idx] = (Mstar[idx])**3.92

    Rstar = (Mstar)**0.945 # Rsun
    Tstar = ( Lstar*Lsun/(4*pi*(Rstar*Rsun)**2*sigma_planck) ) ** 0.25 # K !

    Teq = Tstar * ( ( (Rstar*Rsun) / (2*semi*AU) ) ** (0.5) ) 
    
    return Teq, Lstar



# these functions are just hacks 
def lumi_to_mstar_completo(Lstar, Mstar_obs):
    '''
    Convert stellar luminosity to Mstar using scaling relations.
    These are the same relations used in COMPLETO.
    Mstar_obs is an input parameter just because COMPLETO uses two different relations,
    which depend on the stellar mass.

    Important note:
        This function is only used to convert the observed stellar luminosity 
        to a `completo stellar mass`. 
        This a hack. Ultimately we want to use the stellar luminosity to match the correct Teq
        but COMPLETO takes stellar mass as an input parameter. So this is just a hack.

    Args:   
        Lstar     (Lsol) :   stellar luminosity
        Mstar_obs (Msol) :   observed stellar mass
    Returns:
        Mstar_completo (Msol) : stellar mass as computed by COMPLETO
    '''
    if Mstar_obs < 0.7:
        Mstar_completo = ( Lstar / 0.628 ) ** (1./2.62)
    else:
        Mstar_completo = Lstar ** (1./3.92)
    
    return Mstar_completo

def compute_mstar_completo_given_observables(tsobs, tserr, rsobs, rserr, msobs, size=10000):
    '''
    Compute stellar mass as computed in COMPLETO given the observed stellar temperature and radius.

    Args:
        tsobs (K)    : observed stellar temperature 
        tserr (K)    : uncertainties on stellar temperature from observations
        rsobs (Rsol) : observations stellar radius
        rserr (Rsol) : uncertainties on stellar temperature from observations
        msobs (Msol) : observed stellar mass 
        size (int)   : number of samples to draw

    Returns:
        Msample : samples of stellar mass a

    '''
    # Sample stellar temperature and radius given the observed values + their uncertainties
    # assuming a normal distribution with 
    # x ~ N(mean=observed_values, std = observed_uncertainties)
    tsample = np.random.normal(loc = tsobs, scale = tserr, size = size)
    rsample = np.random.normal(loc = rsobs, scale = rserr, size = size)

    # compute stellar luminosity given stellar radius and temperature.
    # this is regarded as the observed stellar luminosity.
    # this step is required just because the TEPCAT catalog
    # does not provide the stellar luminosity + uncertainties
    Lsample = compute_stellar_luminosity(rsample, tsample)
    
    # stellar mass as computed by COMPLETO
    Msample = lumi_to_mstar_completo(Lsample, msobs)

    return Msample





