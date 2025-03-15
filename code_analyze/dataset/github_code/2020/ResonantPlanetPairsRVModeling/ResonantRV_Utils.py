import numpy as np
from RadvelNbodyReboundModel import NbodyModel
from ResonantPairModel import ACRModel, ACRModelPrior,ACRModelPriorTransform,RadvelModelPriorTransform
from scipy.interpolate import interp1d,RectBivariateSpline
import radvel
import pandas as pd
import pickle
import rebound as rb
import matplotlib.pyplot as plt
import os
from radvel.orbit import timeperi_to_timetrans,timetrans_to_timeperi
_here = os.path.dirname(os.path.abspath(__file__))
ACR_DATA_FILE_PATH = _here +"/ACR_locations_data.pkl" 
class acr_function():
    """
    Class for computing planet eccentricities
    for different ACR configurations.
    """
    def __init__(self,res_j,res_k,ACR_data_file=ACR_DATA_FILE_PATH):
        """
        Parameters
        ----------
        res_j : int
            Together with res_k specifies the particular MMR
            as the j_res:j_res-k_res resonance.
        res_k : int
            Order of the resonance
        ACR_data_file : str
            File path for binary file containting dictionary of ACR curve data.
        """
        with open(ACR_data_file,'rb') as fi:
            all_acr_loc_dict = pickle.load(fi)
        acr_loc_dict = all_acr_loc_dict[(res_j,res_k)]
        gammas = np.array(list(acr_loc_dict.keys()))
        Ngamma = len(gammas)
        Nt = len(acr_loc_dict[gammas[0]][0])
        t = np.linspace(0,1,Nt)
        alpha_res = ((res_j-res_k)/res_j)**(2/3)
        e1_arr = np.zeros((Ngamma,Nt))
        e2_arr = np.zeros((Ngamma,Nt))
        for i,gamma in enumerate(gammas):
            beta1 = 1/(1+gamma)
            beta2 = gamma * beta1
            raw_e1_data,raw_e2_data  = acr_loc_dict[gamma]
            scaled_amd = beta1 * np.sqrt(alpha_res) * raw_e1_data**2 + beta2 * raw_e2_data**2
            scaled_amd /= np.max(scaled_amd)

            # re-sample eccentricities uniformly in sqrt(amd)
            e1_arr[i] = interp1d(np.sqrt(scaled_amd) , raw_e1_data )(t)
            e2_arr[i] = interp1d(np.sqrt(scaled_amd) , raw_e2_data )(t)

        
        self.e1_rbs = RectBivariateSpline(gammas,t,e1_arr)
        self.e2_rbs = RectBivariateSpline(gammas,t,e2_arr)

    def __call__(self,gamma,t):
        """
        Arguments
        ---------
        gamma : float
            Planets' mass ratio = M_out / M_in
        t : float
            Paramterization of the ACR configuration's AMD.
            t=0 corresponds to circular orbits and t=1 corresponds
            to the point where ACR curves of different mass ratios
            all meet.
        """
        return self.e1_rbs(gamma,t)[0,0],self.e2_rbs(gamma,t)[0,0]

def get_acr_like(observations_df,j,k,ACR_data_file=ACR_DATA_FILE_PATH):
    """
    Get a radvel likelihood object for an ACR radial velocity model.
    
    Arguments
    ---------
    observations_df : pandas DataFrame
        Dataframe containing the radial observations.
        Required columns:
            'instrument','time','velocity','uncertainty'

    j : int
        Speficies the j:j-k resonance for the ACR model

    k : int
        Order of resonance for ACR model.

    ACR_data_file : str
        File string designating file containing ACR data.

    Returns
    -------
    radvel CompositeLikelihood
    """
    acrfn = acr_function(j,k,ACR_data_file=ACR_data_file)
    mdl = ACRModel(j,k , acrfn, time_base=observations_df.time.median())
    likes=[]
    instruments = observations_df.instrument.unique()
    for i,instrument in enumerate(instruments):
        data = observations_df.query('instrument==@instrument')[['time','velocity','uncertainty']].values
        inst_like = radvel.likelihood.RVLikelihood(mdl,*(data.T),suffix=instrument)
        gammainst = 'gamma{}'.format(instrument)
        jitinst = 'jit{}'.format(instrument)
        inst_like.params[gammainst].value=0
        inst_like.params[jitinst].value=0
        likes.append(inst_like)
    like = radvel.likelihood.CompositeLikelihood(likes)
    like.params['dvdt'].vary = False
    like.params['curv'].vary = False
    return like

def get_full_like(observations_df):
    """
    Get a radvel likelihood object for a two-planet radial velocity model.
    
    Arguments
    ---------
    observations_df : pandas DataFrame
        Dataframe containing the radial observations.
        Required columns:
            'instrument','time','velocity','uncertainty'

    Returns
    -------
    radvel CompositeLikelihood
    """
    # Set up radvel likelihood object
    fitbasis = 'per tc e w k'
    params = radvel.Parameters(2,basis=fitbasis)
    mdl = radvel.RVModel(params,time_base=observations_df.time.median())
    likes=[]
    instruments = observations_df.instrument.unique()
    for i,instrument in enumerate(instruments):
        data = observations_df.query('instrument==@instrument')[['time','velocity','uncertainty']].values
        inst_like = radvel.likelihood.RVLikelihood(mdl,*(data.T),suffix=instrument)
        # ... setting pars to 0 is for some reason 
        # neccessary to stop the code from eating shit...
        for key,par in inst_like.params.items():
            par.value = 0
        likes.append(inst_like)
    like = radvel.likelihood.CompositeLikelihood(likes)
    like.params['dvdt'].vary = False
    like.params['curv'].vary = False
    return like

def get_nbody_like(observations_df,Mstar = 1.0, meters_per_second=True):
    """
    Get a radvel likelihood object for an N-body radial velocity model.
    
    Arguments
    ---------
    observations_df : pandas DataFrame
        Dataframe containing the radial observations.
        Required columns:
            'instrument','time','velocity','uncertainty'

    Returns
    -------
    radvel CompositeLikelihood
    """
    # Set up radvel likelihood object
    mdl =  NbodyModel(2,
            Mstar = Mstar, 
            time_base=observations_df.time.median(),
            meters_per_second=meters_per_second
            )
    likes=[]
    instruments = observations_df.instrument.unique()
    for i,instrument in enumerate(instruments):
        data = observations_df.query('instrument==@instrument')[['time','velocity','uncertainty']].values
        inst_like = radvel.likelihood.RVLikelihood(mdl,*(data.T),suffix=instrument)
        # ... setting pars to 0 is for some reason 
        # neccessary to stop the code from eating shit...
        inst_like.params['gamma{}'.format(instrument)].value = np.mean(data[:,1])
        inst_like.params['jit{}'.format(instrument)].value = np.mean(data[:,2])
        likes.append(inst_like)
    like = radvel.likelihood.CompositeLikelihood(likes)
    like.params['dvdt'].vary = False
    like.params['curv'].vary = False
    return like

def synthpars_to_sim(spars,t0=0,Mstar = 1,DataInKilometersPerSecond=False):
    """
    Get a rebound simulations from a set of radvel parameters in the 'synth' basis.

    Arguments
    ---------
    spars : radvel.Parameters
        A set of radvel parameters in the synth basis. 
        These can be generated from a likelihood object using
        Likelihood.model.get_synthparams()
    t0 : float, optional
        Simulation's epoch (stored as sim.t).
        Default is 0
    Mstar : float, optional
        Stellar mass in solar masses.
        Default is 1.
    DataInMetersPerSecond : boole, (default=False)
        Semi-amplitudes are in meters per second.

    Returns
    -------
    rebound.Simulation object
    """
    # AU/day to m/s
    Kfactor = 1731.46*1e3
    if DataInKilometersPerSecond:
        Kfactor /= 1e3
    sim = rb.Simulation()
    sim.units = ('day','AU','Msun')
    sim.add(m=Mstar)
    sim.t = t0
    for i in range(1,3):
        P = spars['per{}'.format(i)].value
        e = spars['e{}'.format(i)].value
        omega = spars['w{}'.format(i)].value
        Tp = spars['tp{}'.format(i)].value
        M = np.mod(2 * np.pi * (t0 - Tp) / P,2*np.pi)
        k = spars['k{}'.format(i)].value
        a = P**(2/3) * (sim.G * (Mstar) * (2*np.pi)**(-2))**(1/3)
        mass = Mstar * np.sqrt(1-e*e) * (k / Kfactor) *  (P / 2 / np.pi / a)
        sim.add(m=mass,P=P,e=e,inc = np.pi / 2, omega=omega,M=M)
    sim.move_to_com()
    return sim

def add_telescope_jitter_priors(post,low=1e-3,high=30):
    """
    Add logarithmic priors on instrumental jitter term to a radvel
    posterior object.

    Arguments
    ---------
    post : radvel.posterior.Posterior 
        Object for which to add priors
    low : float, optional
        Prior lower limit set to low * med(unc) where
        med(unc) is the median uncertainty of the instrument's 
        measurments. Default is 1e-3
    high : float, optional
        Prior upper limit set to high * med(unc) where
        med(unc) is the median uncertainty of the instrument's 
        measurments. Default is 30
    """
    telvec = post.likelihood.telvec
    unc = post.likelihood.yerr
    jitpriors = []
    for inst in set(telvec):
        msk = telvec == inst
        med_unc = np.median(unc[msk])
        jit_lo = low * med_unc
        jit_hi = high * med_unc
        parstring = "jit{}".format(inst)
        prior = radvel.prior.Jeffreys(parstring,jit_lo,jit_hi)
        jitpriors.append(prior)
    post.priors += jitpriors
def plot_fit(like,ax=None,Npt = 200):

    if ax is None:
        fig,ax = plt.subplots(1,figsize=(12,4))
        fig.set_tight_layout(True)

    x = like.x
    y = like.model(like.x)+like.residuals()
    # Note:
    #  like.errorbars() values include contribution of jitter terms
    #  while like.yerr values are the raw reported uncertainties.
    yerr = like.errorbars()

    for tel in set(like.telvec):
        msk = like.telvec==tel
        ax.errorbar(x[msk],y[msk],yerr=yerr[msk], fmt='o',label=tel)
    min_time= np.min(like.x)
    max_time = np.max(like.x)
    ti = np.linspace(min_time,max_time,Npt)
    ax.plot(ti, like.model(ti))
    ax.set_xlabel('Time')
    ax.set_ylabel('RV')

def get_planet_mass_from_mstar_K_P_e(mstar,K,P,e):
    """
    Compute planet mass from input stellar mass, semi-amplitude, period and eccentricity.

    Parameters
    ----------
    mstar : real
        Stellar mass in solar masses.
    K : real
        RV semi-amplitude in m/s.
    P : real
        Orbital period in days.
    e : real
        eccentricity

    Returns
    -------
    real
        Planet mass in solar masses
    """

    G = 0.0002959107392494323
    k = K / 1731456.836805555556
    Kcubed= k*k*k
    x = np.sqrt(1-e*e)
    x3 = x*x*x
    Y = x3 * P * Kcubed / (2*np.pi*G)
    return np.real(np.roots([-1,Y,Y*2*mstar,Y*mstar**2])[0])

def process_acr_posterior_dataframe(acr_df,acrlike,Mstar = 1,meters_per_second = True):
    """
    Process a Pandas dataframe containing the posterior samples from a radvel MCMC fit with
    an ACR model. Columns for orbital parameters and planet masses are are added to the dataframe. 

    Arguments
    ---------
    acr_df : pandas.Dataframe
        Dataframe containing posterior samples
    acrlike : radvel likelihood
        The likelihood object used to compute the ACR model posterior sample.
    Mstar : float (optional)
        Host star mass in solar masses.
        Used for converting semi-amplitudes and periods to planet masses. 
        Default value is 1
    meters_per_second : bool (optional)
        Whether semi-amplitude values are reported in meters per second or km/s.
        True (default) if values are given in m/s.
    """
    acr_df['timebase'] = acrlike.model.time_base
    acr_df['t'] = acr_df.stcosw**2 + acr_df.stsinw**2
    acr_df['Mstar'] = Mstar
    acrfn = acrlike.model._acr_curves_fn
    # Planet 1
    acr_df['pomega1'] = np.arctan2(acr_df.stsinw,acr_df.stcosw)
    acr_df['e1'] = acrfn.e1_rbs(acr_df.m2_by_m1,acr_df.t,grid=False)
    acr_df['tp1'] = timetrans_to_timeperi(acr_df.tc1,acr_df.per1,acr_df.e1,acr_df.pomega1)
    acr_df['M1'] = 2*np.pi*(acr_df.timebase - acr_df.tp1) / acr_df.per1
    acr_df['l1'] = np.mod(acr_df.M1 + acr_df.pomega1,2*np.pi)
    k1 = acr_df['k1'].values
    
    if not meters_per_second:
        acr_df['m1'] = np.vectorize(get_planet_mass_from_mstar_K_P_e)(
            acr_df['Mstar'].values,
            1e3*k1,
            acr_df['per1'].values,
            acr_df['e1'].values
        )
    else:
        acr_df['m1'] = np.vectorize(get_planet_mass_from_mstar_K_P_e)(
            acr_df['Mstar'].values,
            k1,
            acr_df['per1'].values,
            acr_df['e1'].values
        )

    # Resonance-related variables
    acr_df['jres'] = acrlike.params['jres'].value
    acr_df['kres'] = acrlike.params['kres'].value
    acr_df['angle_n'] = acrlike.params['angle_n'].value
    # Planet 2
    acr_df['m2'] = acr_df.m1 * acr_df.m2_by_m1
    acr_df['e2'] = acrfn.e2_rbs(acr_df.m2_by_m1,acr_df.t,grid=False)
    acr_df['pomega2'] = np.mod(acr_df['pomega1'] + np.pi,2*np.pi)
    j,k = acr_df['jres'].values,acr_df['kres'].values
    
    acr_df['per2'] = j * acr_df.per1 / (j-k)
    M1 = acr_df['M1']
    angle_n = acr_df['angle_n']
    
    acr_df['M2'] = (1-k/j) * acr_df.M1 + ((1-j+k+2*acr_df.angle_n) / j) * np.pi    
    acr_df['tp2'] = acr_df.timebase - acr_df.per2 * acr_df.M2 / 2 / np.pi
    
    for i in range(1,3):
        rt_e = acr_df['e{}'.format(i)].apply(np.sqrt)
        pmg = acr_df['pomega{}'.format(i)]    
        acr_df['secosw{}'.format(i)]= rt_e * np.cos(pmg)
        acr_df['sesinw{}'.format(i)]= rt_e * np.sin(pmg)

def get_planet_K_from_mstar_mplanet_P_e(mstar,mplanet,P,e):
    """
    Compute RV semi-amplitude in units [m/s] from input 
    stellar mass, mplanet, period and eccentricity.

    Parameters
    ----------
    mstar : real
        Stellar mass in solar masses.
    mplanet : real
        Planet mass in solar masses.
    P : real
        Orbital period in days.
    e : real
        Orbit eccentricity

    Returns
    -------
    K : real
        RV semi-amplitude in m/s
    """
    G = 0.0002959107392494323
    k = (2 * np.pi * G / P)**(1/3) * mplanet / np.sqrt(1-e*e) / (mstar + mplanet)**(2/3)
    return 1731456.836805555556 * k


from scipy.special import erf,erfc
def rv_posterior_predictive_distribution_credible_regions(model_like,df,tmin,tmax,Ntimes=200,Nsample=1000,**kwargs):
    r"""
    Compute credible regions of the posterior predictive distribution of the RV
    signal using a radvel model, a dataframe of posterior samples, and a range
    of times.
    
    Arguemnts
    ---------
    model_like : radvel.likelihood
        Likelihood used for the forward-modeling of the RV signal.
    df : pandas.DataFrame
        Dataframe containing posterior sample of 
        parameters taken by model_like
    tmin : float
        Minimum time of time range for computing RV signal predictive posterior
    tmax : float
        Maximum time of time range for computing RV signal predictive posterior
    Ntimes : int, optional
        Number of times to sample between tmin and tmax.
        Default value is 200.
    Nsample : int, optional
        Number of posterior samples to generate RV signals for.
        Default value is 1000.

    Other Arguments
    ---------------
    levels : array-like, optional
        Credible region levels to return.
        Default values correspond to 1,2, and 3$\sigma$
    full_sample : bool, optional
        Return the underlying sample of RV signals in addition to credible regions.
        Default is False.

    Returns
    -------
    time : ndarray (N,)
        Time values of posterior predictive distribution values.
    lower : ndarray (M,N)
        Lower values bounding the credible regions given by 'levels' of the poserior
        predictive distribution.
        Shape is (M,N) where M is the number of credible regions and N is the number
        of times.
    upper : ndarray (M,N)
        Upper values bounding the credible regions given by 'levels' of the poserior
        predictive distribution.
        Shape is (M,N) where M is the number of credible regions and N is the number
        of times.
    normalized_residual_info : ndarray (3,Nobs)
        Information on the normalized residuals of the fit to the observations.
        Contains the median and ±1sigma values of the normalized residuals for 
        each observation point.
    sample : ndarray, (N,Nsample), optional
        If 'full_sample' is True, contains the full sample of the posterior predictive
        distribution used to compute the credible regoins.
    """
    levels = kwargs.pop('levels',np.array([erf(n/np.sqrt(2)) for n in range(1,4)]))
    full_sample = kwargs.pop('full_sample',False)
    rv_out = np.zeros((Nsample,Ntimes))
    Nobs = len(model_like.x)
    normalized_resids = np.zeros((Nobs,Nsample))
    times = np.linspace(tmin,tmax,Ntimes)
    for k in range(Nsample):
        rv_out[k] = np.infty
        # Avoid unstable posterior points
        while np.any(np.isinf(rv_out[k])):
            i = np.random.randint(0,len(df)-1)
            pars = df.iloc[i]
            vpars = pars[model_like.list_vary_params()]
            model_like.set_vary_params(vpars)
            rv_out[k] = model_like.model(times)
            ebs = model_like.errorbars()
            normalized_resids[:,k] = model_like.residuals() / ebs
    lo,hi = [],[]
    for lvl in levels:
        hi.append(np.quantile(rv_out.T,0.5 + lvl/2, axis=1))
        lo.append(np.quantile(rv_out.T,0.5 - lvl/2, axis=1))
    normalized_residual_quantiles = np.quantile(normalized_resids,(0.5,erf(1),erfc(1)),axis=1)
    if full_sample:
            return times,lo,hi,normalized_residual_quantiles,rv_out
    return times,lo,hi,normalized_residual_quantiles
