import numpy as np
import radvel
import pickle
import pandas as pd
import sys
import re
import dynesty
import os
from ResonantRV_Utils import get_acr_like, get_full_like,get_nbody_like,add_telescope_jitter_priors
from ResonantPairModel import ACRModelPrior, ACRModelPriorTransform
from ResonantPairModel import RadvelModelPriorTransform,NbodyModelPriorTransform

# Fit ACR model for second-order resonance? 
do_second_order = False
do_nbody = True

DATADIR = "./saves/"
AllObservations = pd.read_pickle("./data/New_All_Observations.pkl")

# Set system and (possibly) resonant configuration to fit
I = int(sys.argv[1])
auto_determine_acr = True
if len(sys.argv) > 2:
    auto_determine_acr = False
    j_first_order = int(sys.argv[2])
    if do_second_order:
        j_second_order = int(sys.argv[3])

# Read in observations
system = AllObservations.system.unique()[I]
Observations = AllObservations.query('system==@system')



dname = re.sub("HD ","HD",system)
dname = re.sub("BD\+20 ","BD+20_",dname)
savedir = DATADIR+dname + "/"
if not os.path.exists(savedir):
    os.makedirs(savedir)

# Set up file names for saved data
HostStarInfoFile = "./data/HostStarInfo.pkl"

full_model_mcmc_posterior_file = savedir + "full_model_mcmc_posterior.pkl"
full_model_nested_sampling_results_file = savedir + "full_model_nested_sampling_result.pkl"

nbody_model_mcmc_posterior_file = savedir + "nbody_model_mcmc_posterior.pkl"
nbody_model_nested_sampling_results_file = savedir + "nbody_model_nested_sampling_result.pkl"

acr_model_mcmc_posterior_file = savedir +  "acr_model_mcmc_posterior_{}to{}_angle_n_{}.pkl"
acr_model_nested_sampling_results_file_string = savedir +  "acr_model_nested_sampling_{}to{}_angle_n_{}.pkl"

##########################################################
                #####################
                #  Full model fit   #
                #####################
##########################################################
print("Initiaiting full model fits...")

# SETUP

# Construct radvel likelihood object
full_model_like = get_full_like(Observations)

# Initialize parameters to max likelihood value.
###############################################
# Read in saved data
with open("./data/FullModel_MaxLikelihood_Parameters.pkl","rb") as fi:
    all_pars = pickle.load(fi)

# Set parameter values
best_pars = all_pars[system]
for key,par in full_model_like.params.items():
    par.value = best_pars[key].value

# Check value of logprob
print("Max likelihood params set with lnprob = {:.1f}".format(full_model_like.logprob()))

# Set up posterior object
#########################

# determine semi-amplitudes for setting priors
k1best = full_model_like.params['k1'].value
k2best = full_model_like.params['k2'].value

# Set up posterior object for MCMC
full_model_post = radvel.posterior.Posterior(full_model_like)

# Set priors
full_model_priors = [
    radvel.prior.Jeffreys('k1',0.2 * k1best, 5 * k1best ),
    radvel.prior.Jeffreys('k2',0.2 * k2best, 5 * k2best ),
    radvel.prior.EccentricityPrior(2)
]
full_model_post.priors += full_model_priors
add_telescope_jitter_priors(full_model_post)

# Re-determine max-likelihood
from scipy.optimize import minimize
print("Before fit: logprob: {:.2f}, loglike: {:.2f}".format(full_model_post.logprob(),full_model_like.logprob()))
minresult = minimize(full_model_post.neglogprob_array,full_model_post.get_vary_params())
print("After fit: logprob: {:.2f}, loglike: {:.2f}".format(full_model_post.logprob(),full_model_like.logprob()))

# Nested sampling
try:
    with open(full_model_nested_sampling_results_file,"rb") as fi:
        full_model_nested_results =  pickle.load(fi)
    print("Nested sampling results read from saved file.")
except FileNotFoundError:
    print("No nested sampling save file found.")
    print("Running full model nested sampling fit...")
    full_model_prior_transform = RadvelModelPriorTransform(Observations,full_model_like)
    full_model_nested_sampler = dynesty.NestedSampler(
        full_model_like.logprob_array,
        full_model_prior_transform,
        full_model_prior_transform.Npars,
        sample='rwalk'
    )
    full_model_nested_sampler.run_nested()
    full_model_nested_results = full_model_nested_sampler.results
    with open(full_model_nested_sampling_results_file,"wb") as fi:
        pickle.dump(full_model_nested_results,fi)

# MCMC
########
try:
    full_model_mcmc_results = pd.read_pickle(full_model_mcmc_posterior_file)
    print("MCMC results read from saved file.")
except FileNotFoundError:
    print("No MCMC save file found.")
    print("Running full model MCMC fit...")
    full_model_mcmc_results = radvel.mcmc(full_model_post)
    full_model_mcmc_results.to_pickle(full_model_mcmc_posterior_file)


##########################################################
                #####################
                #  Nbody model fit   #
                #####################
##########################################################
if do_nbody:
    print("Initializing N-body model fits...")
    with open(HostStarInfoFile,"rb") as fi:
        HostStarData=pickle.load(fi)
    Mstar = HostStarData[system][0]
    mps =  system != 'HD 45364'
    nbody_model_like = get_nbody_like(Observations,Mstar,meters_per_second=mps)
    spars = best_pars.basis.to_synth(best_pars)
    nbody_model_like.model.set_pars_from_synth_pars( spars)
    for tel in set(nbody_model_like.telvec):
        nbody_model_like.params['gamma{}'.format(tel)].value = spars['gamma{}'.format(tel)].value
        nbody_model_like.params['jit{}'.format(tel)].value = spars['jit{}'.format(tel)].value
    keplogprob = full_model_like.logprob()
    nblogprob = nbody_model_like.logprob()
    print("logP: {:.2f} (Kepler) {:.2f} (Nbody)".format(keplogprob,nblogprob))

    # Set up posterior object for MCMC
    nbody_model_post = radvel.posterior.Posterior(nbody_model_like)
    m1best = nbody_model_like.params['m1'].value
    m2best = nbody_model_like.params['m2'].value
    
    # Set priors
    nbody_model_priors = [
        radvel.prior.Jeffreys('m1',0.2 * m1best, 5 * m1best ),
        radvel.prior.Jeffreys('m2',0.2 * m2best, 5 * m2best ),
        radvel.prior.EccentricityPrior(2)
    ]
    nbody_model_post.priors += nbody_model_priors
    add_telescope_jitter_priors(nbody_model_post)

    # MCMC
    ########
    try:
        nbody_model_mcmc_results = pd.read_pickle(nbody_model_mcmc_posterior_file)
        print("N-body MCMC results read from saved file.")
    except FileNotFoundError:
        print("No N-body MCMC save file found.")
        print("Running N-body model MCMC fit...")
        nbody_model_mcmc_results = radvel.mcmc(nbody_model_post)
        nbody_model_mcmc_results.to_pickle(nbody_model_mcmc_posterior_file)
    
    # Nested sampling
    try:
        with open(nbody_model_nested_sampling_results_file,"rb") as fi:
            nbody_model_nested_results =  pickle.load(fi)
        print("Nested sampling results read from saved file.")
    except FileNotFoundError:
        print("No N-body nested sampling save file found.")
        print("Running N-body model nested sampling fit...")
        nbody_model_prior_transform = NbodyModelPriorTransform(Observations,nbody_model_like)
        nbody_model_nested_sampler = dynesty.NestedSampler(
            nbody_model_like.logprob_array,
            nbody_model_prior_transform,
            nbody_model_prior_transform.Npars,
            periodic = nbody_model_prior_transform.periodic_indicies,
            sample='rwalk'
        )
        nbody_model_nested_sampler.run_nested()
        nbody_model_nested_results = nbody_model_nested_sampler.results
        with open(nbody_model_nested_sampling_results_file,"wb") as fi:
            pickle.dump(nbody_model_nested_results,fi)
##########################################################
                #####################
                #  ACR model fit   #
                #####################
##########################################################
print("Initiaiting ACR fits...")

# Determine nearest first-order resonance based on 
period_ratio = full_model_mcmc_results['per2'] / full_model_mcmc_results['per1']
period_ratio = period_ratio.median()
if auto_determine_acr:
    j_first_order = np.int(np.round(1+1/(period_ratio-1)))
# override...
if system=='HD 99706':
    j_first_order = 3


# Set up ACR likelihoods
first_order_acr_model_like = get_acr_like(Observations,j_first_order,1)
for key,par in first_order_acr_model_like.params.items():
    if key in full_model_like.params.keys():
        par.value = full_model_like.params[key].value

### Nested sampling ###

# define prior transforms
first_order_acr_model_prior_transform = ACRModelPriorTransform(Observations,first_order_acr_model_like)


### First-order res ###
sampler_first_order_acr_model = dynesty.NestedSampler(
    first_order_acr_model_like.logprob_array,
    first_order_acr_model_prior_transform,
    first_order_acr_model_prior_transform.Npars,
    sample='rwalk'
)

best_logz = -np.inf
best_angle_n = -1
print("Running ACR nested sampling fits...")
for angle_n in range(j_first_order):
    print("\t angle_n {} of {}".format(angle_n,j_first_order))
    # Reset sampler, change angle_n parameter
    sampler_first_order_acr_model.reset()
    first_order_acr_model_like.params['angle_n'].value = angle_n
    
    # Run sampler, save results.
    filestring = acr_model_nested_sampling_results_file_string.format(j_first_order,j_first_order - 1, angle_n)
    try:
        with open(filestring,"rb") as fi:
            acr_model_nested_results = pickle.load(fi)
        print("Nested sampling results read from saved file.")

    except FileNotFoundError:
        print("No nested sampling save file found.")
        print("Running ACR model nested sampling fit...")
        sampler_first_order_acr_model.run_nested()
        acr_model_nested_results = sampler_first_order_acr_model.results
        with open(filestring,"wb") as fi:
            pickle.dump(acr_model_nested_results,fi)
    
    # Test logz to see if it is the new maximum.
    logz = acr_model_nested_results['logz'][-1]
    print("logz={:.1f} for angle_n={}".format(logz,angle_n),end=' ')
    if logz > best_logz:
        best_logz = logz
        best_angle_n = angle_n

        # Save best parameters to use as initialization point of MCMC
        imax = np.argmax(acr_model_nested_results.logl)
        first_order_acr_model_best_pars = acr_model_nested_results.samples[imax]
        print("(New best)")
    else:
        print("")

#  MCMC
# Set priors
first_order_acr_model_like.params['angle_n'].value = best_angle_n
first_order_acr_model_post = radvel.posterior.Posterior(first_order_acr_model_like)
acrpriors = [
    ACRModelPrior(),
    radvel.prior.Jeffreys('k1',0.2 * k1best, 5 * k1best ),
    radvel.prior.Jeffreys('m2_by_m1',0.1,10)
]
first_order_acr_model_post.priors += acrpriors
add_telescope_jitter_priors(first_order_acr_model_post)
first_order_acr_model_post.set_vary_params(first_order_acr_model_best_pars)

# Re-determine max-likelihood
print("Finding ACR model  max-likelihood...")
print("Before fit: logprob: {:.2f}, loglike: {:.2f}".format(first_order_acr_model_post.logprob(),first_order_acr_model_like.logprob()))
minresult = minimize(first_order_acr_model_post.neglogprob_array,first_order_acr_model_post.get_vary_params())
print("After fit: logprob: {:.2f}, loglike: {:.2f}".format(first_order_acr_model_post.logprob(),first_order_acr_model_like.logprob()))

filestring = acr_model_mcmc_posterior_file.format(j_first_order,j_first_order-1,best_angle_n)
try:
    first_order_acr_model_mcmc_results = pd.read_pickle(filestring)
    print("MCMC results read from saved file.")
except FileNotFoundError:
    print("No MCMC save file found.")
    print("Running ACR model MCMC fit...")
    first_order_acr_model_mcmc_results = radvel.mcmc(first_order_acr_model_post)
    first_order_acr_model_mcmc_results.to_pickle(filestring)

if do_second_order:

    if auto_determine_acr:
        j_second_order = np.int(np.round(2+2/(period_ratio-1)))
        # Check if nearest second order is a 'proper' second order resonance
        if j_second_order%2==0:
            jplus = j_second_order + 1
            jminus = j_second_order - 1
            delta_plus = np.abs( period_ratio * (jplus - 2 ) / jplus - 1 )
            delta_minus = np.abs(period_ratio * (jminus - 2 ) / jminus - 1)
            if delta_plus < delta_minus:
                j_second_order = jplus
            else:
                j_second_order = jminus

    second_order_acr_model_like = get_acr_like(Observations,j_second_order,2)
    for key,par in second_order_acr_model_like.params.items():
        if key in full_model_like.params.keys():
            par.value = full_model_like.params[key].value

    second_order_acr_model_prior_transform = ACRModelPriorTransform(Observations,second_order_acr_model_like)
    sampler_second_order_acr_model = dynesty.NestedSampler(
        second_order_acr_model_like.logprob_array,
        second_order_acr_model_prior_transform,
        second_order_acr_model_prior_transform.Npars,
        sample='rwalk'
    )
    
    best_logz = -np.inf
    best_angle_n = -1
    print("Running ACR nested sampling fits for second-order resonance...")
    for angle_n in range(j_second_order):
        print("\t angle_n {} of {}".format(angle_n,j_second_order))
        # Reset sampler, change angle_n parameter
        sampler_second_order_acr_model.reset()
        second_order_acr_model_like.params['angle_n'].value = angle_n
        
        # Run sampler, save results.
        filestring = acr_model_nested_sampling_results_file_string.format(j_second_order,j_second_order - 2, angle_n)
        try:
            with open(filestring,"rb") as fi:
                acr_model_nested_results = pickle.load(fi)
            print("Nested sampling results read from saved file.")
    
        except FileNotFoundError:
            print("No nested sampling save file found.")
            print("Running ACR model nested sampling fit...")
            sampler_second_order_acr_model.run_nested()
            acr_model_nested_results = sampler_second_order_acr_model.results
            with open(filestring,"wb") as fi:
                pickle.dump(acr_model_nested_results,fi)
        
        # Test logz to see if it is the new maximum.
        logz = acr_model_nested_results['logz'][-1]
        print("logz={:.1f} for angle_n={}".format(logz,angle_n),end=' ')
        if logz > best_logz:
            best_logz = logz
            best_angle_n = angle_n
    
            # Save best parameters to use as initialization point of MCMC
            imax = np.argmax(acr_model_nested_results.logl)
            second_order_acr_model_best_pars = acr_model_nested_results.samples[imax]
            print("(New best)")
        else:
            print("")
    
    #  MCMC
    # Set priors
    second_order_acr_model_like.params['angle_n'].value = best_angle_n
    second_order_acr_model_post = radvel.posterior.Posterior(second_order_acr_model_like)
    acrpriors = [
        ACRModelPrior(),
        radvel.prior.Jeffreys('k1',0.2 * k1best, 5 * k1best ),
        radvel.prior.Jeffreys('m2_by_m1',0.1,10)
    ]
    second_order_acr_model_post.priors += acrpriors
    add_telescope_jitter_priors(second_order_acr_model_post)
    second_order_acr_model_post.set_vary_params(second_order_acr_model_best_pars)
    
    # Re-determine max-likelihood
    print("Finding ACR model  max-likelihood...")
    print("Before fit: logprob: {:.2f}, loglike: {:.2f}".format(second_order_acr_model_post.logprob(),second_order_acr_model_like.logprob()))
    minresult = minimize(second_order_acr_model_post.neglogprob_array,second_order_acr_model_post.get_vary_params())
    print("After fit: logprob: {:.2f}, loglike: {:.2f}".format(second_order_acr_model_post.logprob(),second_order_acr_model_like.logprob()))
    
    filestring = acr_model_mcmc_posterior_file.format(j_second_order,j_second_order-2,best_angle_n)
    try:
        second_order_acr_model_mcmc_results = pd.read_pickle(filestring)
        print("MCMC results read from saved file.")
    except FileNotFoundError:
        print("No MCMC save file found.")
        print("Running ACR model MCMC fit...")
        second_order_acr_model_mcmc_results = radvel.mcmc(second_order_acr_model_post)
        second_order_acr_model_mcmc_results.to_pickle(filestring)
