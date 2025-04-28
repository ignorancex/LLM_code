import numpy as np
from . import utils

"""Tension between data sets"""

#Distance between data sets in Gaussian limit
def distance(parameters_1,parameters_2):
    """
    Compute distance between two sets of parameters.
    The definition is given in Eq. 10 of https://arxiv.org/pdf/2308.06616.pdf

    Parameters
    ----------
    parameters_1 : array_like
        First set of parameters.
    parameters_2 : array_like
        Second set of parameters.
    
    Returns
    -------
    distance : float
        Distance between the two sets of parameters.
    """    
    if len(parameters_1) == 0:
        return 0
    elif np.shape(parameters_1)[1] == 1:
        return distance1D(parameters_1,parameters_2)
    
    #Difference means
    mean_1 = np.mean(parameters_1,axis=0)
    mean_2 = np.mean(parameters_2,axis=0)
    Dphi_12 = mean_1 - mean_2
        
    #Compute covariance
    cov_1 = np.cov(parameters_1.T)
    cov_2 = np.cov(parameters_2.T)
    
    #Compute distance
    c12_inv = np.linalg.inv(cov_1 + cov_2)  
    return np.sqrt(np.matmul(Dphi_12,np.matmul(c12_inv,Dphi_12)))

#Distance in 1D
def distance1D(parameter_1,parameter_2):
    """
    Compute distance between two sets of parameters in 1D.
    The definition is given in Eq. 10 of https://arxiv.org/pdf/2308.06616.pdf

    Parameters
    ----------
    parameter_1 : array_like
        First set of parameters.
    parameter_2 : array_like
        Second set of parameters.

    Returns
    -------
    distance : float
        Distance between the two sets of parameters.
    """
    #Difference means
    mean_1 = np.mean(parameter_1,axis=0)
    mean_2 = np.mean(parameter_2,axis=0)
    Dphi_12 = mean_1 - mean_2
        
    #Compute covariance
    cov_1 = np.var(parameter_1)
    cov_2 = np.var(parameter_2)
    
    #Compute distance
    c12_inv = 1/(cov_1 + cov_2)  
    return np.sqrt(Dphi_12*c12_inv*Dphi_12)

def distance_phases(parameters_1,parameters_2,nphases):
    """
    Compute distance between two sets of parameters accounting for the wrap around of the phases.

    Parameters
    ----------
    parameters_1 : array_like
        First set of parameters.
    parameters_2 : array_like
        Second set of parameters.
    nphases : int
        Number of phases. It is assumed that the first nphases parameters are phases.

    Returns
    -------
    distance : float
        Distance between the two sets of parameters accounting for the wrap around of the phases.
    """

    #Wrap phases
    parameters_1_wrap, parameters_2_wrap = utils.wrap_phases(parameters_1,parameters_2,nphases)

    #Compute distance
    return distance(parameters_1_wrap,parameters_2_wrap)

def distance_phases_with_shift(parameters_1,parameters_2,nphases):
    """
    Compute distance between two sets of parameters with possible lensing phase shifts.
    Details in Eq. 12 of https://arxiv.org/pdf/2308.06616.pdf

    Parameters
    ----------
    parameters_1 : array_like
        First set of parameters.
    parameters_2 : array_like
        Second set of parameters.
    nphases : int
        Number of phases.
    
    Returns
    -------
    distances : array_like
        Distances between the two sets of parameters with possible lensing phase shifts.
    """
    #Add possible lensing phase shifts
    phase_shifts = np.array([0,1,2,-1])*np.pi/2
    distances = 0.*phase_shifts
    
    #Loop around phase shifts
    length = np.shape(parameters_1)[1]
    for i, phase_shift in enumerate(phase_shifts):
        parameters_2_shifted = parameters_2 + phase_shift*np.hstack((np.ones(nphases),np.zeros(length-nphases)))
        parameters_2_shifted[:,:nphases] = np.mod(parameters_2_shifted[:,:nphases],2*np.pi)

        distances[i] = distance_phases(parameters_1,parameters_2_shifted,nphases)
        
    return distances

"""Tension between data sets: only periodic phases"""

def distance_only_phases(phases_1,phases_2):
    """
    Compute distance between two sets of phases accounting for the wrap around of the phases.  

    Parameters
    ----------
    phases_1 : array_like
        First set of phases.
    phases_2 : array_like
        Second set of phases.

    Returns
    -------
    distance : float
        Distance between the two sets of phases accounting for the wrap around of the phases.
    """   
    #Wrap phases
    phases_1_wrap, phases_2_wrap = utils.wrap_only_phases(phases_1,phases_2)

    #Compute distance
    return distance(phases_1_wrap,phases_2_wrap)

def distance_only_phases_with_shift(parameters_1,parameters_2):
    """
    Compute distance between two sets of phasess with possible lensing phase shifts.

    Parameters
    ----------
    parameters_1 : array_like
        First set of parameters.
    parameters_2 : array_like
        Second set of parameters.
        
    Returns
    -------
    distances : array_like
        Distances between the two sets of phases with possible lensing phase shifts.
    """
    #Add possible lensing phase shift
    phase_shifts = np.array([0,1,2,-1])*np.pi/2
    distances = 0.*phase_shifts
    
    length = np.shape(parameters_1)[1]
    for i, phase_shift in enumerate(phase_shifts):
        parameters_2_shifted = np.mod(parameters_2 + phase_shift*np.ones(length),2*np.pi)

        distances[i] = distance_only_phases(parameters_1,parameters_2_shifted)
        
    return distances

"""Volume calculation"""

#Volume
def volume(parameters): 
    """
    Volume of a set of parameters.
    Definition in Eq. 17 of https://arxiv.org/pdf/2308.06616.pdf

    Parameters
    ----------
    parameters : array_like
        Set of parameters.
    
    Returns
    -------
    volume : float
        Volume of the set of parameters.
    """
    return np.sqrt(np.linalg.det(np.cov(parameters.T)))

def volume_phases(parameters,nphases):
    """
    Volume of a set of parameters accounting for the wrap around of the phases.

    Parameters
    ----------
    parameters : array_like
        Set of parameters.
    nphases : int
        Number of phases. It is assumed that the first nphases parameters are phases.

    Returns
    -------
    volume : float
        Volume of the set of parameters accounting for the wrap around of the phases.
    """
    phases = parameters[:,:nphases]
    
    #Wrap phases  around their modes 
    #to avoid discontinuities from periodic boundaries
    mode = utils.modes(phases)
    
    phases_wrap = np.mod(phases - mode + np.pi,2*np.pi) + mode - np.pi
    
    #Put with rest of parameters
    parameters_wrap = np.concatenate((phases_wrap,parameters[:,nphases:]),axis=1)
    
    return np.sqrt(np.linalg.det(np.cov(parameters_wrap.T)))

def volume_only_phases(phases_1,phases_2):
    """
    Volume of a set of parameters accounting for the wrap around of the phases.
    
    Parameters
    ----------
    phases_1 : array_like
        First set of phases.
    phases_2 : array_like
        Second set of phases.

    Returns
    -------
    volume : float
        Volume of the set of parameters accounting for the wrap around of the phases.
    """
    if len(phases_1) == 0:
        return 0
    elif np.shape(phases_1)[1] == 1:
        return np.std(phases_1), np.std(phases_2)
    
    #Wrap phases
    phases_1_wrap, phases_2_wrap = utils.wrap_only_phases(phases_1,phases_2)

    #Compute volume
    return volume(phases_1_wrap), volume(phases_2_wrap)

"""Number of effective degrees of freedom"""

def n_eff(phases_1,phases_2,nphases,prior_range):
    """
    Compute number of effective degrees of freedom. 
    Definition in Eq. 11 of https://arxiv.org/pdf/2308.06616.pdf

    Parameters
    ----------
    phases_1 : array_like
        First set of phases.
    phases_2 : array_like
        Second set of phases.
    nphases : int
        Number of phases.
    prior_range : float
        Prior range.

    Returns
    -------
    n_eff : float
        Number of effective degrees of freedom.
    """

    #Flat prior on the phases
    var_prior = (prior_range**2) / 12
    cov_prior = np.diag(np.ones(nphases))*var_prior
    
    cov_1 = np.cov(phases_1.T)
    cov_2 = np.cov(phases_2.T)
    
    cov_inv = np.linalg.inv(cov_prior + cov_1 + cov_2)
    cov_mult = np.matmul(cov_inv,cov_prior)
    
    return np.trace(cov_mult)

def n_eff_1D(phase_1,phase_2,prior_range):
    """
    Compute number of effective degrees of freedom in 1D.
    Definition in Eq. 11 of https://arxiv.org/pdf/2308.06616.pdf

    Parameters
    ----------
    phase_1 : array_like
        First set of phases.
    phase_2 : array_like
        Second set of phases.
    prior_range : float
        Prior range.

    Returns
    -------
    n_eff : float
        Number of effective degrees of freedom.
    """
    #Flat prior on the phases
    var_prior = (prior_range**2) / 12
    cov_prior = 1.*var_prior
    
    cov_1 = np.var(phase_1)
    cov_2 = np.var(phase_2)
    
    cov_inv =1./(cov_prior + cov_1 + cov_2)
    cov_mult = cov_inv*cov_prior
    
    return cov_mult

def n_eff_pair_1D(phases_1,phases_2,prior_range): 
    """
    Compute number of effective degrees of freedom in 1D for a pair.
    Definition in Eq. 11 of https://arxiv.org/pdf/2308.06616.pdf

    Parameters
    ----------
    phases_1 : array_like
        First set of phases.
    phases_2 : array_like
        Second set of phases.
    prior_range : float
        Prior range.

    Returns
    -------
    n_eff : float
        Number of effective degrees of freedom.
    """
    #Wrap phases
    phases_1_wrap, phases_2_wrap = utils.wrap_only_phases(phases_1,phases_2)
    
    return n_eff_1D(phases_1_wrap,phases_2_wrap,prior_range)

def n_eff_pair(phases_1,phases_2,nphases,prior_range):
    """
    Compute number of effective degrees of freedom for a pair.
    Definition in Eq. 11 of https://arxiv.org/pdf/2308.06616.pdf

    Parameters
    ----------
    phases_1 : array_like
        First set of phases.
    phases_2 : array_like
        Second set of phases.
    nphases : int
        Number of phases.
    prior_range : float
        Prior range.

    Returns
    -------
    n_eff : float
        Number of effective degrees of freedom.
    """
    if nphases == 0:
        return 0
    elif nphases == 1:
        return n_eff_pair_1D(phases_1,phases_2,prior_range)
    
    #Wrap phases
    phases_1_wrap, phases_2_wrap = utils.wrap_only_phases(phases_1,phases_2)
    
    return n_eff(phases_1_wrap,phases_2_wrap,nphases,prior_range)