"""
A series of python functions for computing posterior distributions 
of the black hole parameters involved in tidal disruption events.   

Author: Andrew Mummery, Oxford University. 1st December 2023. 
        andrew.mummery@physics.ox.ac.uk 

Paper: The maximum mass of a black hole which can tidally disrupt a star: 
       measuring black hole spins with tidal disruption events
       Andrew Mummery. 
       Published in MNRAS. 

Please cite this work if you use these scripts. 
"""
import numpy as np 
from numpy import cos, sin, pi, roots
from astropy import constants 
from tqdm import tqdm
import matplotlib.pyplot as plt

Ms = constants.M_sun.value
Rs = constants.R_sun.value
c = constants.c.value
G = constants.G.value

settup_str = """
  ________________
 |_____    _______| ____    _______   __        _______  ______  
       |  |    (_) |  _ \  /  __   \ |  |      |   ____||   (_)|(_)  ______
       |  |    | | | | | ||  |__|  | |  |      |  |___  |   ___|| | |  __  |
       |  |    | | | |_| ||  |  |  | |  |____  |___   | |  |    | | | |  | |
       |__|    |_| |____/ |__|  |__| |_______|  ___|  | |__|    |_| |_|  |_|
                                               |______|

             TidalSpin was created by Andrew Mummery*

              * andrew.mummery@physics.ox.ac.uk
"""

print(settup_str)## Forgive me, I coded a lot of this during a long train journey. 

def main():

    format_plots_nicely()## Default matplotlib looks rubbish. 

    ### Produce 2D mass spin contours with Monte Carlo, and two 1D posteriors. 
    logM = 8.5##Some prior log10(mass) estimate. 
    sigmaM = 0.3##Uncertainty in log_10(black hole mass) from M-sigma relationship

    ## The black hole mass prior used in the paper. 
    prior_MBH = lambda lm: log_norm(lm, logM, sigmaM) * (10**lm/1e8)**0.03 * np.exp(-(10**lm/(6.4e7))**0.49) 

    hm, mass_matrix = get_hills_masses()## Get Hills masses for solar mass star (for comparison)

    p_a, a = spin_posterior(max_mass_matrix=mass_matrix, prior_Mbh=prior_MBH, log_Mbh_min=7, log_Mbh_max=10)## Get 1D spin posterior 
    p_m, m = mass_posterior(max_mass_matrix=mass_matrix, prior_Mbh=prior_MBH, log_Mbh_min=7, log_Mbh_max=10)## Get 1D mass posterior
    a_samples, m_samples, _, _ = monte_carlo_all(max_mass_matrix=mass_matrix, prior_Mbh=prior_MBH, log_Mbh_min=7, log_Mbh_max=10, N_draw=100000)
    ## Monte Carlo 100,000 TDEs to verify 1D posteriors and determine 2D correlations. 

    ## Make plot, similar to figure 10 of the paper. 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax2.axis('off')

    ax3.scatter(m_samples, a_samples, marker='.', color='blue', alpha=0.02, rasterized=True)
    ax3.set_xlabel(r'$\log_{10}M_\bullet/M_\odot$', fontsize=30)
    ax3.set_ylabel(r'$a_\bullet$', fontsize=30)
    ax3.plot(np.log10(hm/Ms), a, ls='-.', c='k')
    ax3.set_ylim(-1, 1)
    ax3.set_xlim(right=9.2)
    ax3_xlim = ax3.get_xlim()


    ax1.hist(m_samples, bins=20, density=True, color='blue')
    ax1.plot(m, p_m, c='r', ls='--')
    ax1.set_ylabel(r'$p(\log_{10} M_\bullet/M_\odot)$', fontsize=30)
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_xlim(ax3_xlim)

    ax4.hist(a_samples, bins=30, density=True, color='blue')
    ax4.plot(a, p_a, ls='--', c='r')

    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel(r'$p(a_\bullet)$', fontsize=30, rotation=270, labelpad=40)
    ax4.set_xlabel(r'$a_\bullet$', fontsize=30)
    ax4.set_xlim(-1, 1)
    ax4.yaxis.tick_right()
    ax4.set_yticks([])
    ax4.set_yticklabels([])

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()



def format_plots_nicely():
    ### Default matplotlib looks rubbish. 
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['font.size'] = 20
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.edgecolor'] = 'k'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['figure.figsize'] = [12, 9]
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['savefig.format'] = 'pdf'
    plt.rcParams['savefig.bbox'] = 'tight'
    return 

    
def get_ibso(a, psi):
    """
    Solves the numerical octic equation to return xi, the dimensionless
    innermost bound spherical orbit radius. 
    See Appendix A of the paper for more details. 
    
    Input: 
        a -- dimensionless black hole spin 
        psi -- asymptotic inclination of star 
    Output:
        xi -- dimensionless ibso radius. 
    
    Note:
        Works in the sign convention -1 < a < 1 and 0 < psi < pi/2. 
        Will likely return nonsense outside of these bounds. 

    """
    cs = [a**8 * cos(psi)**4, 
          0, 
          -2 * a**6 * cos(psi)**2 + 6 * a**6 * cos(psi)**4,
          -8 * a**4 * cos(psi)**2 + 16 * a**4 * cos(psi)**2 * sin(psi)**2, 
          a**4 - 4 * a**4 * cos(psi)**2 + 9 * a**4 * cos(psi)**4,  
          8 * a**2 - 24 * a**2 * cos(psi)**2 - 16 * a**2 * sin(psi)**2,  
          16 - 2 * a**2 + 6 * a**2 * cos(psi)**2, 
          -8, 
          1]
    
    root = roots(cs[::-1])

    if a > 1:
        return np.nan

    real_roots = root[np.isreal(root)]
    if len(real_roots)!=2:
        if a > 0:
            return np.real(real_roots[1])
        return np.real(real_roots[0])
        
    if a > 0:
        return np.real(min(real_roots))
    return np.real(max(real_roots))


def hills_mass(a, psi, eta=1, mstar=1, rstar=1):
    """
    Returns the Hills mass for a given black hole spin and incoming stellar angle. 

    Inputs: 
        a -- dimensionless black hole spin. 
        psi -- asymptotic inclination of star 

        eta -- stellar self gravity parameter (default = 1)
        mstar -- stellar mass (solar mass units, default = 1)
        rstar -- stellar radius (solar radii units, default = 1)
    
    Outputs:
        Mhills -- the Hills mass (solar masses)

    Note:
        Works in the sign convention -1 < a < 1 and 0 < psi < pi/2. 
        Will likely return nonsense outside of these bounds. 

    """
    x = get_ibso(a, psi)
    amp = (2/eta * c**6 * (rstar * Rs)**3/(G**3 * mstar * Ms))**0.5 * 1/x**1.5 
    fac = (1 + 6*x/(x**2 - a**2*cos(psi)**2) + 3*a**2/(2*x**2) - 6*a*sin(psi)/x**1.5 * (x**2/(x**2 - a**2*cos(psi)**2))**0.5)**0.5
    return amp * fac


def get_hills_masses(N_spin = 300, 
                  N_psi = 200):
    """
    Returns the Hills masses for a grid of 
    black hole spins and incoming stellar angle. 

    Inputs: 
        N_spin -- number of spins to grid
        N_psi -- Number of inclinations to grid 

    Outputs:
        max_m -- the maximum Hills mass for each spin 
        
        max_mass_matrix -- the Hills mass for each spin and inclination

    Notes:
        Grids spins from -0.9999 to +0.9999 in linear steps. 
        Grids spins from 0, pi/2 in linear steps. 

        Works in the sign convention -1 < a < 1 and 0 < psi < pi/2. 
        Will likely return nonsense outside of these bounds. 

        This code assumes solar values for M_star, R_star. 
        This code assumes eta = 1. 

        Hills masses can be shifted by a factor 
        f=(R_\star/R_\odot)^{3/2} (M_\odot/M_\star)^{1/2} \eta^{-1/2}
        for other parameters.

    """

    psi_ = np.linspace(0.001, pi/2, N_psi)
    a_ = np.linspace(-0.9999, +0.9999, N_spin)

    print('Generating Hills masses......')
    max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

    for i, a in enumerate(tqdm(a_)):
        for j, psi in enumerate(psi_):
            max_mass_matrix[i, j] = hills_mass(a, psi)

    max_m = np.zeros_like(a_)
    for j in range(len(a_)):
        max_m[j] = max(max_mass_matrix[j, :])

    return max_m, max_mass_matrix
    

def MassRadiusRelation(mstar):
    """
    Returns the (Kippenhahn & Weigert 1990) mass-radius relationship for stars. 

    Input:
        Stellar mass (physical units)
    Output:
        Stellar radii (physical units)
    Notes:
        Intended only for internal code use. 
    """
    rstar = np.zeros_like(mstar)
    i_low = mstar < Ms 
    i_high = ~i_low 
    rstar[i_low] = Rs * np.power(mstar[i_low]/Ms,0.56)
    rstar[i_high] = Rs * np.power(mstar[i_high]/Ms,0.79)
    return rstar

def KroupaIMF(mstar):
    """
    Returns the (Kroupa 2001) initial stellar mass function. 

    Input:
        List of stellar masses (physical units)
    Output:
        Probability density dN_star/dM_star.
    Notes:
        Intended only for internal code use. 
    """
    i_lowest = mstar < 0.08 * Ms 
    i_low = (mstar < 0.5 * Ms) * (~ i_lowest)
    i_med = (mstar < 1 * Ms) * (~ i_lowest) *  (~ i_low)
    i_high = (~ i_lowest) * (~ i_low) * (~i_med)

    P = np.zeros_like(mstar)

    P[i_lowest] = (mstar[i_lowest]/(0.08*Ms))**(-0.3)
    P[i_low] = (mstar[i_low]/(0.08*Ms))**(-1.8)
    P[i_med] = (mstar[i_med]/(0.08*Ms))**(-2.7) * (0.5/0.08)**(-1.8+2.7)
    P[i_high] = (mstar[i_high]/(0.08*Ms))**(-2.3)  * (1/0.08)**(-2.7+2.3) * (0.5/0.08)**(-1.8+2.7)

    return P
    
def TDE_rate_on_stellar_params(mstar):
    """
    Returns the (Wang and Merrit 2004) TDE rate as a function of stellar properties. 

    Input:
        List of stellar masses (physical units)
    Output:
        TDE rate R(M_\star). 
    Notes:
        Intended only for internal code use. 
    """
    rstar = MassRadiusRelation(mstar)
    return np.power(mstar/Ms,-1/3) * np.power(rstar/Rs, 1/4)

def StellarMassDistributionFunction(Mmin=0.01*Ms, Mmax=2.0*Ms, N=10000):
    mstars = np.logspace(np.log10(Mmin), np.log10(Mmax), N)
    dM = (mstars[1:] - mstars[:-1])
    ps = KroupaIMF(mstars) * TDE_rate_on_stellar_params(mstars)
    return ps[:-1]/np.sum(ps[:-1] * dM)


### Log-normal for black hole masses 
def log_norm(x, log_mu, log_sigma):
    ### Mean must be in log10, as must x. 
    return 1/(np.sqrt(2*np.pi)*log_sigma) * np.exp(-(x-log_mu)**2/(2*log_sigma**2))

def spin_posterior_fixed_mass(Mbh, 
                  prior_spin=None, 
                  max_mass_matrix=None, 
                  prior_star=None, 
                  prior_psi=None,

                  N_star = 100, 
                  Mstar_min=0.1, 
                  Mstar_max=1,

                  N_spin = 300, 
                  N_psi = 200, 

                  eta = 1,
                  additional_likelihood=None):
    """
    Returns a black hole spin posterior assuming a fixed black hole mass. Units = physical. 

    Inputs: 
        Mbh -- the black hole mass at which to produce a posterior. 

        prior_spin -- function that accepts dimensionless spin and returns a prior.
        If none, defaults to flat prior. 

        max_mass_matrix -- the ooutput of get_hills_masses().
        If None calculates this matrix in this code. 

        M_star_min, M_star_max -- lower and upper bounds on stellar population. Solar units expected. 

        prior_star, prior_psi -- prior distribtuions of stellar properties and incoming inclinations. 
        Defaults to the priors used in the paper if None. 
        
        N_X -- the number of grid spaces for variable X. 

        eta -- stellar self gravity parameter (see paper).  
        Defaults to 1. 

        additional_likelihood -- a function that accepts M_bh/M_hills and returns a likelihood. 
        Defaults to 1 if unspecified (as in paoer). See Appendix D of paper. 
        Using additional likelihood will result in stronger spin constraints, but must be physically motivated. 
    
    Returns: 
        a -- list of spins at which the spin posterior is computed. 
        p_a -- the values of the posterior. 

    Notes:
        Returned posterior p_a includes both prograde and retrograde spins.  
        If the absolute value of the spin is required use p(|a|) = p(a) + p(-a), 
        or in Python:

        abs_a = a[N_spin/2:]

        p_abs_a = p_a[N_spin/2:] + p_a[N_spin/2 - 1 :: -1]

        Note that this requires N_spin to be an even number. 

    """    
    if prior_spin is None:
        prior_spin = lambda a: np.ones_like(a)## agnostic spin prior. 


    Mstar_min *=  Ms
    Mstar_max *=  Ms

    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    if prior_star is None:
        psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)
    else:
        psd = prior_star(mstars)

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]

    if max_mass_matrix is None:
        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

        print('Generating mass matrix......')
        max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

        for i, a in enumerate(tqdm(a_)):
            for j, psi in enumerate(psi_):
                max_mass_matrix[i, j] = hills_mass(a, psi)
    else:
        N_spin = len(max_mass_matrix[:, 0])
        N_psi =  len(max_mass_matrix[0, :])

        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

    dpsi = psi_[1] - psi_[0]
    da_ = a_[1] - a_[0]

    f = ((MassRadiusRelation(mstars)/Rs)**1.5 * (Ms/mstars)**0.5 * eta**0.5)

    mass_mass_matrix = [[] for _ in range(N_spin)]

    for k in range(N_spin):
        tmp_matrix = np.zeros(N_psi * N_star).reshape(N_psi, N_star)
        for u in range(N_psi):
            tmp_matrix[u, :] = max_mass_matrix[k, u] * f
        mass_mass_matrix[k] = tmp_matrix


    p_a = np.zeros(N_spin)

    if prior_psi is None:
        ppsi = np.cos(psi_)
    else:
        ppsi = prior_psi(psi_)

    if additional_likelihood is None:
        additional_likelihood = lambda x: np.ones_like(x)

    for i, a in enumerate(a_):
        p = 0
        p_ = np.heaviside(mass_mass_matrix[i] - Mbh, 1) * additional_likelihood(Mbh/mass_mass_matrix[i]) 
        p += np.sum(np.sum(p_ * psd * dMs, axis=1) * ppsi * dpsi, axis=0) * prior_spin(a) 
        
        p_a[i] = p

    p_a = p_a
    p_a = p_a/sum(p_a * da_)
    
    return p_a, a_




def spin_posterior(prior_Mbh, log_Mbh_min, log_Mbh_max, 
                  prior_spin=None,
                  max_mass_matrix = None, 
                  prior_star=None, 
                  prior_psi=None,

                  N_star = 100, 
                  Mstar_min=0.1, 
                  Mstar_max=1,

                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 
                  
                  eta = 1, 
                  
                  additional_likelihood=None): 
    
    """
    Inputs: 
        prior_Mbh -- function which accepts the logarithm of the black hole mass and returns a prior. 
        log_Mbh_min, log_Mbh_max -- extent of black hole mass range to compute posterior over. Units = log_10 M_solar.

        prior_spin -- function that accepts dimensionless spin and returns a prior.
        If none, defaults to flat prior. 

        max_mass_matrix -- the ooutput of get_hills_masses().
        If None calculates this matrix in this code. 

        M_star_min, M_star_max -- lower and upper bounds on stellar population. Solar units expected. 

        prior_star, prior_psi -- prior distribtuions of stellar properties and incoming inclinations. 
        Defaults to the priors used in the paper if None. 
        
        N_X -- the number of grid spaces for variable X. 

        eta -- stellar self gravity parameter (see paper).  
        Defaults to 1. 

        additional_likelihood -- a function that accepts M_bh/M_hills and returns a likelihood. 
        Defaults to 1 if unspecified (as in paoer). See Appendix D of paper. 
        Using additional likelihood will result in stronger spin constraints, but must be physically motivated. 
    
    Returns: 
        a -- list of spins at which the spin posterior is computed. 
        p_a -- the values of the posterior. 

    Notes:
        Returned posterior p_a includes both prograde and retrograde spins.  
        If the absolute value of the spin is required use p(|a|) = p(a) + p(-a), 
        or in Python:

        abs_a = a[N_spin/2:]

        p_abs_a = p_a[N_spin/2:] + p_a[N_spin/2 - 1 :: -1]

        Note that this requires N_spin to be an even number. 

    """

    if prior_spin is None:
        prior_spin = lambda a: np.ones_like(a)## agnostic spin prior. 


    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    if prior_star is None:
        psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)
    else:
        psd = prior_star(mstars)

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]

    if max_mass_matrix is None:
        psi_ = np.linspace(0.001, pi/2, N_psi)
        dpsi = psi_[1] - psi_[0]

        a_ = np.linspace(-0.9999, +0.9999, N_spin)
        da_ = a_[1] - a_[0]

        print('Generating mass matrix......')
        max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

        for i, a in enumerate(tqdm(a_)):
            for j, psi in enumerate(psi_):
                max_mass_matrix[i, j] = hills_mass(a, psi)
    
    else:
        N_spin = len(max_mass_matrix[:, 0])
        N_psi =  len(max_mass_matrix[0, :])

        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

        dpsi = psi_[1] - psi_[0]
        da_ = a_[1] - a_[0]


    f = ((MassRadiusRelation(mstars)/Rs)**1.5 * (Ms/mstars)**0.5 * eta**0.5)

    if prior_psi is None:
        ppsi = np.cos(psi_)
    else:
        ppsi = prior_psi(psi_)

    mass_mass_matrix = [[] for _ in range(N_spin)]

    for k in range(N_spin):
        tmp_matrix = np.zeros(N_psi * N_star).reshape(N_psi, N_star)
        for u in range(N_psi):
            tmp_matrix[u, :] = max_mass_matrix[k, u] * f
        mass_mass_matrix[k] = tmp_matrix


    p_a = np.zeros(len(a_))

    log_Ms = np.linspace(log_Mbh_min, log_Mbh_max, N_bh)
    dMbh = (log_Ms[1] - log_Ms[0])

    if additional_likelihood is None:
        additional_likelihood = lambda x: np.ones_like(x)

    print('Getting 1D spin distribution.....')
    for i, a in enumerate(tqdm(a_)):
        p = 0
        for l_m_bh in log_Ms:
            m_bh = 10**l_m_bh * Ms
            p_ = np.heaviside(mass_mass_matrix[i] - m_bh, 1)  * additional_likelihood(m_bh/mass_mass_matrix[i]) 
            p += np.sum(np.sum(p_ * psd * dMs, axis=1) * ppsi * dpsi, axis=0) * prior_Mbh(l_m_bh) * dMbh 
        
        p_a[i] = p * prior_spin(a) 

    p_a = p_a/sum(p_a * da_)
    
    return p_a, a_



def mass_posterior(prior_Mbh, log_Mbh_min, log_Mbh_max,
                  prior_spin=None,
                  max_mass_matrix = None, 
                  prior_star=None, 
                  prior_psi=None,

                  N_star = 100, 
                  Mstar_min=0.1, 
                  Mstar_max=1,

                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 
                  
                  eta = 1,
                  
                  additional_likelihood=None): 

    """
    Computes the posterior black hole mass distribution for a TDE with prior black hole mass estimate.

    Inputs: 
        prior_Mbh -- function which accepts the logarithm of the black hole mass and returns a prior. 
        log_Mbh_min, log_Mbh_max -- extent of black hole mass range to compute posterior over. Units = log_10 M_solar.

        prior_spin -- function that accepts dimensionless spin and returns a prior.
        If none, defaults to flat prior. 

        max_mass_matrix -- the ooutput of get_hills_masses().
        If None calculates this matrix in this code. 

        M_star_min, M_star_max -- lower and upper bounds on stellar population. Solar units expected. 

        prior_star, prior_psi -- prior distribtuions of stellar properties and incoming inclinations. 
        Defaults to the priors used in the paper if None. 
        
        N_X -- the number of grid spaces for variable X. 

        eta -- stellar self gravity parameter (see paper).  
        Defaults to 1. 

        additional_likelihood -- a function that accepts M_bh/M_hills and returns a likelihood. 
        Defaults to 1 if unspecified (as in paoer). See Appendix D of paper. 
        Using additional likelihood will result in stronger spin constraints, but must be physically motivated. 
    
    Returns: 
        log_mbh -- list of log10(black hole masses) at which the posterior is computed.
        p_mbh -- the values of the posterior.

    """
    if prior_spin is None:
        prior_spin = lambda a: np.ones_like(a)## agnostic spin prior. 


    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    if prior_star is None:
        psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)
    else:
        psd = prior_star(mstars)[:-1]

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]

    if max_mass_matrix is None:
        psi_ = np.linspace(0.001, pi/2, N_psi)
        dpsi = psi_[1] - psi_[0]


        a_ = np.linspace(-0.9999, +0.9999, N_spin)
        da_ = a_[1] - a_[0]

        print('Generating mass matrix......')
        max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

        for i, a in enumerate(tqdm(a_)):
            for j, psi in enumerate(psi_):
                max_mass_matrix[i, j] = hills_mass(a, psi)

    else:
        N_spin = len(max_mass_matrix[:, 0])
        N_psi =  len(max_mass_matrix[0, :])

        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

        dpsi = psi_[1] - psi_[0]
        da_ = a_[1] - a_[0]

    f = ((MassRadiusRelation(mstars)/Rs)**1.5 * (Ms/mstars)**0.5 * eta**0.5)

    if prior_psi is None:
        ppsi = np.cos(psi_)
    else:
        ppsi = prior_psi(psi_)

    mass_mass_matrix = [[] for _ in range(N_spin)]

    for k in range(N_spin):
        tmp_matrix = np.zeros(N_psi * N_star).reshape(N_psi, N_star)
        for u in range(N_psi):
            tmp_matrix[u, :] = max_mass_matrix[k, u] * f
        mass_mass_matrix[k] = tmp_matrix


    log_Ms = np.linspace(log_Mbh_min, log_Mbh_max, N_bh)
    dMbh = (log_Ms[1] - log_Ms[0])

    if additional_likelihood is None:
        additional_likelihood = lambda x: np.ones_like(x)
    

    p_bh = np.zeros(len(log_Ms))

    print('Getting 1D mass distribution.....')
    for i, l_m_bh in enumerate(tqdm(log_Ms)):
        p = 0
        m_bh = 10**l_m_bh * Ms
        for j, a in enumerate(a_):
            p_ = np.heaviside(mass_mass_matrix[j] - m_bh, 1) * additional_likelihood(m_bh/mass_mass_matrix[i]) 
            p += np.sum(np.sum(p_ * psd * dMs, axis=1) * ppsi * dpsi, axis=0) * prior_spin(a) * da_
    
        p_bh[i] = p * prior_Mbh(l_m_bh)

    p_bh = p_bh/sum(p_bh * dMbh)
    
    return p_bh, log_Ms


def monte_carlo_all(prior_Mbh, log_Mbh_min, log_Mbh_max,
                  prior_spin=None,
                  max_mass_matrix=None,
                  prior_star=None, 
                  prior_psi=None,
                  
                  N_draw=10000,

                  N_star = 100, 
                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 

                  Mstar_min=0.1, 
                  Mstar_max=1,
                  
                  eta = 1,
                  
                  additional_likelihood=None):
    """
    Runs a monte carlo simulation of length N_draw and returns the parameters of each of the successful TDEs. 

    Inputs: 
        prior_Mbh -- function which accepts the logarithm of the black hole mass and returns a prior. 
        log_Mbh_min, log_Mbh_max -- extent of black hole mass range to compute posterior over. Units = log_10 M_solar.

        prior_spin -- function that accepts dimensionless spin and returns a prior.
        If none, defaults to flat prior. 

        max_mass_matrix -- the ooutput of get_hills_masses().
        If None calculates this matrix in this code. 

        N_draw -- the number of monte-carlo samples to perform. 

        M_star_min, M_star_max -- lower and upper bounds on stellar population. Solar units expected. 

        prior_star, prior_psi -- prior distribtuions of stellar properties and incoming inclinations. 
        Defaults to the priors used in the paper if None. 
        
        N_X -- the number of grid spaces for variable X. 

        eta -- stellar self gravity parameter (see paper).  
        Defaults to 1. 

        additional_likelihood -- a function that accepts M_bh/M_hills and returns a likelihood. 
        Defaults to 1 if unspecified (as in paoer). See Appendix D of paper. 
        Using additional likelihood will result in stronger spin constraints, but must be physically motivated.     

    Returns: 
        a_samples, m_samples, mstar_samples, psi_samples = list of length N_draw of all the parameters from successful TDE samples. 

    Notes:
        Returned a_samples includes both prograde and retrograde spins.  

    """
    if prior_spin is None:
        prior_spin = lambda a: np.ones_like(a)## agnostic spin prior. 
    
    log_Ms = np.linspace(log_Mbh_min, log_Mbh_max, N_bh)
    dMbh = (log_Ms[1] - log_Ms[0])

    PHI_MBH = np.cumsum(prior_Mbh(log_Ms)*dMbh)/np.sum(prior_Mbh(log_Ms)*dMbh)

    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    if prior_star is None:
        psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)
    else:
        psd = prior_star(mstars)[:-1]

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]
    PHI_stars = np.cumsum(psd * dMs)

    if max_mass_matrix is None:
        psi_ = np.linspace(0.001, pi/2, N_psi)
        dpsi = psi_[1] - psi_[0]
        a_ = np.linspace(-0.9999, +0.9999, N_spin)
        da_ = a_[1] - a_[0]
        print('Generating mass matrix......')
        max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

        for i, a in enumerate(tqdm(a_)):
            for j, psi in enumerate(psi_):
                max_mass_matrix[i, j] = hills_mass(a, psi)

    else:
        N_spin = len(max_mass_matrix[:, 0])
        N_psi =  len(max_mass_matrix[0, :])

        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

        dpsi = psi_[1] - psi_[0]
        da_ = a_[1] - a_[0]


    if prior_psi is None:
        ppsi = np.cos(psi_)
    else:
        ppsi = prior_psi(psi_)

    PHI_psis = np.cumsum(ppsi * dpsi)
    
    PHI_A = np.cumsum(prior_spin(a_)*da_)/np.sum(prior_spin(a_)*da_)


    m_samples = np.full(N_draw, np.nan)
    a_samples = np.full(N_draw, np.nan)
    m_star_samples = np.full(N_draw, np.nan)
    psi_samples = np.full(N_draw, np.nan)


    if additional_likelihood is None:
        additional_likelihood = lambda x: np.ones_like(x)

    print('Generating samples......')
    n=0

    pbar = tqdm(total=N_draw, mininterval=1)

    while n < N_draw:
        u = np.random.uniform(0, 1, 1)
        l_m_bh = log_Ms[np.argmin(abs(PHI_MBH - u))]
        m_bh = 10**l_m_bh * Ms

        w = np.random.uniform(0, 1, 1)
        m_star = mstars[np.argmin(abs(PHI_stars - w))]

        v = np.random.uniform(0, 1, 1)
        psi = psi_[np.argmin(abs(PHI_psis - v))]
        j = np.argmin(abs(psi - psi_))

        x = np.random.uniform(0, 1, 1)
        a_bh = a_[np.argmin(abs(PHI_A - x))]
        i = np.argmin(abs(a_ - a_bh))

        m_test = (m_bh * (MassRadiusRelation(m_star)/Rs)**-1.5 * (Ms/m_star)**-0.5 * eta**-0.5)

        p_ = (max_mass_matrix[i, j] > m_test) *  additional_likelihood(m_test/max_mass_matrix[i, j]) 

        y = np.random.uniform(0, 1, 1)

        if p_>y:
            m_samples[n] = l_m_bh 
            a_samples[n] = a_bh
            m_star_samples[n] = m_star
            psi_samples[n] = psi
            n+=1
            pbar.update()


    a_samples = a_samples[a_samples==a_samples]
    m_samples = m_samples[m_samples==m_samples]
    psi_samples = psi_samples[psi_samples==psi_samples]
    m_star_samples = m_star_samples[m_star_samples==m_star_samples]

    return a_samples, m_samples, m_star_samples, psi_samples


def observable_posterior(observe_dist, observe_values,
                  prior_Mbh, log_Mbh_min, log_Mbh_max,
                  prior_spin=None,
                  max_mass_matrix = None,
                  prior_star=None, 
                  prior_psi=None,

                  N_star = 100, 
                  Mstar_min=0.1, 
                  Mstar_max=1,

                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 
                  
                  eta = 1,
                  additional_likelihood=None):

    """
    Computes the posterior of a TDE observable O which depends only on black hole mass. 

    Inputs: 
        observe_dist -- a probability density function that takes in as input a value of the observable, 
        and black hole mass, and returns a probability density. This is p(O|M) in the notation of the paper. 

        observe_values -- a list of values at which to evaluate the posterior. 
    
        prior_Mbh -- function which accepts the logarithm of the black hole mass and returns a prior. 
        log_Mbh_min, log_Mbh_max -- extent of black hole mass range to compute posterior over. Units = log_10 M_solar.

        prior_spin -- function that accepts dimensionless spin and returns a prior.
        If none, defaults to flat prior. 

        max_mass_matrix -- the ooutput of get_hills_masses().
        If None calculates this matrix in this code. 

        M_star_min, M_star_max -- lower and upper bounds on stellar population. Solar units expected. 

        prior_star, prior_psi -- prior distribtuions of stellar properties and incoming inclinations. 
        Defaults to the priors used in the paper if None. 
        
        N_X -- the number of grid spaces for variable X. 

        eta -- stellar self gravity parameter (see paper).  
        Defaults to 1. 

        additional_likelihood -- a function that accepts M_bh/M_hills and returns a likelihood. 
        Defaults to 1 if unspecified (as in paoer). See Appendix D of paper. 
        Using additional likelihood will result in stronger spin constraints, but must be physically motivated.     
    
    Returns: 
        p_observe -- the values of the posterior for the observable O.

    """
    
    p_tde, m_tde = mass_posterior(max_mass_matrix=max_mass_matrix, prior_Mbh=prior_Mbh, prior_spin=prior_spin, prior_star=prior_star, prior_psi=prior_psi,
                                   log_Mbh_min=log_Mbh_min, log_Mbh_max=log_Mbh_max, N_star=N_star, Mstar_min=Mstar_min, 
                                   Mstar_max=Mstar_max, N_bh = N_bh, N_spin=N_spin, N_psi=N_psi, eta=eta, additional_likelihood=additional_likelihood)## Gets 1D mass posterior.
    

    dM = m_tde[1] - m_tde[0]

    p_observe = np.zeros_like(observe_values)

    for k, o in enumerate(observe_values):
        p_observe[k] = sum( observe_dist(o, np.array(m_tde)) * p_tde * dM )## Integrates p(O|M) p(M) dM. 
    
    return p_observe


def monte_carlo_observable(observe_func, 
                  prior_Mbh, log_Mbh_min, log_Mbh_max,
                  prior_spin=None,
                  max_mass_matrix=None,
                  prior_star=None, 
                  prior_psi=None,
                  
                  N_draw=10000,

                  N_star = 100, 
                  N_bh = 150, 
                  N_spin = 300, 
                  N_psi = 200, 

                  Mstar_min=0.1, 
                  Mstar_max=1,
                  
                  eta = 1,
                  additional_likelihood=None):
    """
    Runs a monte carlo simulation of length N_draw and returns the parameters of each of the successful TDEs. 
    Also computes a TDE observable for each TDE, which is assumed to depend only on black hole mass,  
    and returns the monte-carlo distribution of the observable. 

    Inputs: 
        observe_func -- function which takes as input log_mbh and returns an observable. 

        prior_Mbh -- function which accepts the logarithm of the black hole mass and returns a prior. 
        log_Mbh_min, log_Mbh_max -- extent of black hole mass range to compute posterior over. Units = log_10 M_solar.

        prior_spin -- function that accepts dimensionless spin and returns a prior.
        If none, defaults to flat prior. 

        max_mass_matrix -- the ooutput of get_hills_masses().
        If None calculates this matrix in this code. 

        N_draw -- the number of monte-carlo samples to perform. 

        M_star_min, M_star_max -- lower and upper bounds on stellar population. Solar units expected. 

        prior_star, prior_psi -- prior distribtuions of stellar properties and incoming inclinations. 
        Defaults to the priors used in the paper if None. 
        
        N_X -- the number of grid spaces for variable X. 

        eta -- stellar self gravity parameter (see paper).  
        Defaults to 1. 

        additional_likelihood -- a function that accepts M_bh/M_hills and returns a likelihood. 
        Defaults to 1 if unspecified (as in paoer). See Appendix D of paper. 
        Using additional likelihood will result in stronger spin constraints, but must be physically motivated.     
    
    Returns: 
        obs_samples, a_samples, m_samples, mstar_samples, psi_samples -- list of length N_draw of all the parameters from successful TDE samples. 

    Notes:
        Returned a_samples includes both prograde and retrograde spins.  

    """    
    if prior_spin is None:
        prior_spin = lambda a: np.ones_like(a)## agnostic spin prior. 
    
    log_Ms = np.linspace(log_Mbh_min, log_Mbh_max, N_bh)
    dMbh = (log_Ms[1] - log_Ms[0])

    PHI_MBH = np.cumsum(prior_Mbh(log_Ms)*dMbh)/np.sum(prior_Mbh(log_Ms)*dMbh)

    Mstar_min *=  Ms
    Mstar_max *=  Ms


    mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), N_star+1)
    if prior_star is None:
        psd = StellarMassDistributionFunction(Mmin=Mstar_min,Mmax=Mstar_max,N=N_star+1)
    else:
        psd = prior_star(mstars)[:-1]

    dMs = mstars[1:]-mstars[:-1]
    mstars = mstars[:-1]
    PHI_stars = np.cumsum(psd * dMs)

    if max_mass_matrix is None:
        psi_ = np.linspace(0.001, pi/2, N_psi)
        dpsi = psi_[1] - psi_[0]    
        a_ = np.linspace(-0.9999, +0.9999, N_spin)
        da_ = a_[1] - a_[0]

        print('Generating mass matrix......')
        max_mass_matrix = np.zeros(len(a_)*len(psi_)).reshape(len(a_), len(psi_))

        for i, a in enumerate(tqdm(a_)):
            for j, psi in enumerate(psi_):
                max_mass_matrix[i, j] = hills_mass(a, psi)
    else:
        N_spin = len(max_mass_matrix[:, 0])
        N_psi =  len(max_mass_matrix[0, :])

        psi_ = np.linspace(0.001, pi/2, N_psi)
        a_ = np.linspace(-0.9999, +0.9999, N_spin)

        dpsi = psi_[1] - psi_[0]
        da_ = a_[1] - a_[0]


    if prior_psi is None:
        ppsi = np.cos(psi_)
    else:
        ppsi = prior_psi(psi_)

    PHI_psis = np.cumsum(ppsi * dpsi)
    PHI_A = np.cumsum(prior_spin(a_)*da_)/np.sum(prior_spin(a_)*da_)

    if additional_likelihood is None:
        additional_likelihood = lambda x: np.ones_like(x)


    m_samples = np.full(N_draw, np.nan)
    a_samples = np.full(N_draw, np.nan)
    m_star_samples = np.full(N_draw, np.nan)
    psi_samples = np.full(N_draw, np.nan)
    obs_samples = np.full(N_draw, np.nan)

    print('Generating samples......')
    n=0

    pbar = tqdm(total=N_draw, mininterval=1)

    while n < N_draw:
        u = np.random.uniform(0, 1, 1)
        l_m_bh = log_Ms[np.argmin(abs(PHI_MBH - u))]
        m_bh = 10**l_m_bh * Ms

        w = np.random.uniform(0, 1, 1)
        m_star = mstars[np.argmin(abs(PHI_stars - w))]

        v = np.random.uniform(0, 1, 1)
        psi = psi_[np.argmin(abs(PHI_psis - v))]
        j = np.argmin(abs(psi - psi_))

        x = np.random.uniform(0, 1, 1)
        a_bh = a_[np.argmin(abs(PHI_A - x))]
        i = np.argmin(abs(a_ - a_bh))

        m_test = (m_bh * (MassRadiusRelation(m_star)/Rs)**-1.5 * (Ms/m_star)**-0.5 * eta**-0.5)
        p_ = (max_mass_matrix[i, j] > m_test) *  additional_likelihood(m_test/max_mass_matrix[i, j]) 

        if p_:
            m_samples[n] = l_m_bh 
            a_samples[n] = a_bh
            m_star_samples[n] = m_star
            psi_samples[n] = psi
            obs_samples[n] = observe_func(l_m_bh)

            n+=1
            pbar.update()


    a_samples = a_samples[a_samples==a_samples]
    m_samples = m_samples[m_samples==m_samples]
    psi_samples = psi_samples[psi_samples==psi_samples]
    m_star_samples = m_star_samples[m_star_samples==m_star_samples]
    obs_samples = obs_samples[obs_samples==obs_samples]

    return obs_samples, a_samples, m_samples, m_star_samples, psi_samples






if __name__ == "__main__":
    main()