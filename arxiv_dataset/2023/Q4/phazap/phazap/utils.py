import numpy as np

def format_pretty_phase_shift(phase_shift):
    # phase_shift can only be multiple of pi/2
    # NOTE In newer versions of phazap we do not explicitly compute -pi phase shift
    prefactor = int(phase_shift/np.pi * 2) # This should be an integer
    conversion_chart = {
        0: "0",
        1: "+π/2",
        2: "±π",
        -1: "-π/2",
        -2: "-π",
    }
    return conversion_chart[prefactor]

""" Compute p-value """
from scipy.stats import chi2
from scipy.interpolate import interp1d

def p_sigma(dist,df):
    """
    Compute p-value from chi2 distribution

    Parameters
    ----------
    dist : float
        Distance
    df : int
        Degrees of freedom

    Returns
    -------
    p : float
        p-value
    """
    ps = np.logspace(-15,0,10000)
    #df = 1
    chi2s = chi2.isf(ps,df)
    p = interp1d(np.sqrt(chi2s),ps, bounds_error=False, fill_value=min(ps))
    return 1 - p(dist)

""" Compute mode """

#1D Maximum likelihood value from posterior samples
def mode(posterior,bins=30):
    """
    Compute mode of a posterior distribution

    Parameters
    ----------
    posterior : np.ndarray
        Posterior samples
    bins : int
        Number of bins

    Returns
    -------
    mode : float
        Mode of the posterior
    """
    bins, param = np.histogram(posterior,bins=bins)
    return param[np.argmax(bins)]

def modes(posteriors,bins=30):
    """
    Compute modes of a set of posterior distributions

    Parameters
    ----------
    posterior : np.ndarray
        Posterior samples
    bins : int
        Number of bins

    Returns
    -------
    modes : np.ndarray
        Modes of the posteriors
    """
    length = np.shape(posteriors)[-1]
    
    modes_c = np.zeros(length)
    for i in range(length):
        modes_c[i] = mode(posteriors[:,i],bins)
    return modes_c

""" Wrap phases around their modes """

def wrap_phases(parameters_1,parameters_2,nphases):
    """
    Wrap phases around their modes to avoid discontinuities from periodic boundaries

    Parameters
    ----------
    parameters_1 : np.ndarray
        Posterior samples
    parameters_2 : np.ndarray
        Posterior samples
    nphases : int
        Number of phases in the posterior samples.
        It is assumed that the first nphases parameters are phases.

    Returns
    -------
    parameters_1_wrap_mod : np.ndarray
        Wrapped posterior samples of parameter_1. The phases are wrapped around the modes of parameters_1.
    parameters_2_wrap_mod : np.ndarray
        Wrapped posterior samples of parameter_2. The phases are wrapped around the modes of parameters_1.
    """
    #Wrap phases around their modes 
    #to avoid discontinuities from periodic boundaries
    phases_1 = parameters_1[:,:nphases]
    phases_2 = parameters_2[:,:nphases]
    
    mode_1 = modes(phases_1)
    mode_2 = modes(phases_2)
   
    phases_1_wrap = np.mod(phases_1 - mode_1 + np.pi,2*np.pi)
    phases_2_wrap = np.mod(phases_2 - mode_2 + np.pi,2*np.pi) + mode_2 - mode_1
    
    #Put with rest of parameters
    parameters_1_wrap = np.concatenate((phases_1_wrap,parameters_1[:,nphases:]),axis=1)
    parameters_2_wrap = np.concatenate((phases_2_wrap,parameters_2[:,nphases:]),axis=1)
    
    mean_1_wrap = np.mean(parameters_1_wrap,axis=0)
    mean_2_wrap = np.mean(parameters_2_wrap,axis=0)
    
    #Make sure mean is within 0, 2pi
    parameters_1_mod = np.concatenate((np.mod(phases_1_wrap,2*np.pi),parameters_1[:,nphases:]),axis=1)
    parameters_2_mod = np.concatenate((np.mod(phases_2_wrap,2*np.pi),parameters_2[:,nphases:]),axis=1)
    
    mean_1_mod = np.mean(parameters_1_mod,axis=0)
    mean_2_mod = np.mean(parameters_2_mod,axis=0)
    
    parameters_1_wrap_mod = parameters_1_wrap - mean_1_wrap + mean_1_mod
    parameters_2_wrap_mod = parameters_2_wrap - mean_2_wrap + mean_2_mod
    
    return parameters_1_wrap_mod, parameters_2_wrap_mod

def wrap_only_phases(phases_1,phases_2):
    """
    Wrap phases around their modes to avoid discontinuities from periodic boundaries

    Parameters
    ----------
    phases_1 : np.ndarray
        Posterior samples of phases
    phases_2 : np.ndarray
        Posterior samples of phases

    Returns
    -------
    phases_1_wrap : np.ndarray
        Wrapped posterior samples of phases_1. The phases are wrapped around the modes of phases_1.
    phases_2_wrap_mod : np.ndarray
        Wrapped posterior samples of phases_2. The phases are wrapped around the modes of phases_1.
    """
    #Wrap phases around their modes 
    #to avoid discontinuities from periodic boundaries
    
    mode_1 = modes(phases_1)
    mode_2 = modes(phases_2)
   
    phases_1_wrap = np.mod(phases_1 - mode_1 + np.pi,2*np.pi)
    phases_2_wrap = np.mod(phases_2 - mode_2 + np.pi,2*np.pi) + mode_2 - mode_1   
    
    mean_2_wrap = np.mean(phases_2_wrap,axis=0)
    
    #Make sure mean is within 0, 2pi
    phases_2_mod = np.mod(phases_2_wrap,2*np.pi)
    
    mean_2_mod = np.mean(phases_2_mod,axis=0)
    
    phases_2_wrap_mod = phases_2_wrap - mean_2_wrap + mean_2_mod
    
    return phases_1_wrap, phases_2_wrap_mod

def wrap_phase(phases_1):
    """
    Wrap phases around their modes to avoid discontinuities from periodic boundaries

    Parameters
    ----------
    phases_1 : np.ndarray
        Posterior samples of phases

    Returns
    -------
    phases_1_wrap : np.ndarray
        Wrapped posterior samples of phases_1. The phases are wrapped around their modes.
    """
    #Wrap phases around their modes 
    #to avoid discontinuities from periodic boundaries
    
    mode_1 = mode(phases_1)
   
    phases_1_wrap = np.mod(phases_1 - mode_1 + np.pi,2*np.pi)
    
    return phases_1_wrap