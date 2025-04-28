# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:25:33 2023

@author: David J. Kedziora
"""

import numpy as np
from lmfit import Parameters, Minimizer
# from scipy import optimize as sco

def func_pulsed(params, x, y = None, duration = 1):
    # Define the pulsed-laser fitting function for the delay histogram.
    # This describes the Poisson-based mean rate of two-photon event detection.
    # Is based on Eq. (2) of: https://doi.org/10.1063/1.5143786
    # x: domain (delays)
    # y: events per bin to be fitted
    # duration: time in seconds over which events have a chance to be detected
    # mr_bg: mean rate (per second) due to background (noise)
    # period_pulse: pulse period of the stimulating laser
    # delay_mpe: the delay value at which multi-photon emission (MPE) occurs
    # mr_env: mean rate (per second) at the function envelope, i.e. the non-MPE peaks
    # g2zero: the ratio of the MPE mean rate to the envelope mean rate, i.e. g2(0)
    # factor_env: the inverted decay scale for the envelope amplitude 
    # decay_peak: the decay scale for an individual peak
    
    rate_bg = params["rate_bg"]
    period_pulse = params["period_pulse"]
    delay_mpe = params["delay_mpe"]
    rate_env = params["rate_env"]
    g2zero = params["g2_zero"]
    factor_env = params["factor_env"]
    decay_peak = params["decay_peak"]
    
    # Convert mean rate per second to the mean number of events observed.
    bg = duration*rate_bg
    env = duration*rate_env
    
    tau = x - delay_mpe
    tau_m = np.mod(tau, period_pulse)
    
    fit = bg + env*np.exp(-factor_env*np.abs(tau)) * (np.cosh((tau_m - period_pulse/2)/decay_peak) / np.sinh(period_pulse/(2*decay_peak)) 
       - (1 - g2zero)*np.exp(-np.abs(tau)/decay_peak))
    
    if y is None:
        return fit
    else:
        return fit - y
    
def estimate_g2zero_pulsed(in_sr_sample, in_sr_delays, in_knowns, use_poisson_likelihood, 
                           in_duration = 1, do_ignore_bg = False):
    # Take a distribution of detected events defined over a domain of delays.
    # They are detected over a certain duration of seconds.
    # Fit a function of Poisson-based means to the data.
    
    period_pulse = in_knowns["period_pulse"]
    range_delays = in_sr_delays[in_sr_delays.size-1] - in_sr_delays[0]
    step_delays = in_sr_delays[1] - in_sr_delays[0]
    num_events = np.sum(in_sr_sample)
    
    # Guess conservative initial values for mean rate per second.
    # If background dominates, the rate should be uniform.
    # If delta-function peaks dominate, the rates will be their amount inverted. 
    rate_uniform = num_events/len(in_sr_delays)/in_duration
    rate_peaked = num_events/np.floor(range_delays/period_pulse)/in_duration
    
    params = Parameters()
    if do_ignore_bg:
        params.add("rate_bg", value = 0, min = 0, max = np.inf, vary = False)
    else:
        params.add("rate_bg", value = rate_uniform, min = 0, max = np.inf)
    params.add("period_pulse", value = period_pulse, min = 0, max = np.inf, vary = False)
    params.add("delay_mpe", value = np.mean(in_knowns["delay_mpe"]), 
               min = in_knowns["delay_mpe"][0], max = in_knowns["delay_mpe"][-1])
    params.add("rate_env", value = rate_peaked, min = 0, max = np.inf)
    params.add("g2_zero", value = 0.5, min = 0, max = 1)
    params.add("factor_env", value = 0, min = 0, max = np.inf, vary = False)
    params.add("decay_peak", value = step_delays, min = step_delays/10, max = np.inf)
    
    # See the supplementary material of: https://doi.org/10.1063/1.5143786
    # The objective to minimise is based on Eq. (S25) and Eq. (S26).
    if use_poisson_likelihood:
        # r is assumed to be the residual fit - data, hence re-adding the data.
        # The new objective should be fit - data*np.log(fit).
        mini = Minimizer(func_pulsed, params, fcn_args = (in_sr_delays, in_sr_sample, in_duration),
                         reduce_fcn = lambda r : 
                             ((r + in_sr_sample) 
                              - in_sr_sample*np.log(r + in_sr_sample)).sum())
    else:
        # The default objective is just the sum of the residual squared.
        mini = Minimizer(func_pulsed, params, fcn_args = (in_sr_delays, in_sr_sample, in_duration))
        
    # Run a five-parameter fit with the Powell method.
    # Then run a five-parameter 'Trust Region Reflective' fit, just for the errors.
    mini.calc_covar = False
    fitted_params = mini.minimize(method="powell")
    mini.calc_covar = True
    fitted_params = mini.minimize(method="least_squares", params=fitted_params.params)
    
    return fitted_params

# def func_pulsed_integral(params, a, b):
#     # Calculates integral of fitting function from a to b.
#     # Only valid if factor_env is zero.
#     factor_env = params["factor_env"]
#     if not factor_env == 0:
#         raise Exception("Error: A decaying envelope is not appropriate for this integral.")
        
#     def func_int(params, tau):
#         bg = params["bg"]
#         period_pulse = params["period_pulse"]
#         amp_env = params["amp_env"]
#         amp_ratio = params["amp_ratio"]
#         decay_peak = params["decay_peak"]
        
#         tau_m = np.mod(tau, period_pulse)
#         term_1a = np.sinh((tau_m - period_pulse/2)/decay_peak) / np.sinh(period_pulse/(2*decay_peak))
#         term_1b = 2*np.floor_divide(tau, period_pulse)
#         # term_2a = (amp_ratio - 1) * np.exp(-tau/decay_peak)/2
#         # term_2b = (np.exp(tau/decay_peak)+1)**2 - np.sign(tau)*(np.exp(tau/decay_peak)-1)**2 - 2
#         term_2a = (amp_ratio - 1)
#         term_2b = np.heaviside(-tau, 0.5)*np.exp(tau/decay_peak)
#         term_2c = np.heaviside(tau, 0.5)*(2-np.exp(-tau/decay_peak))
#         func = bg * tau + amp_env * decay_peak * (term_1a + term_1b + term_2a * (term_2b + term_2c))
#         # func = bg * tau + amp_env * decay_peak * (term_2a * (term_2b + term_2c))
#         return func
    
#     delay_mpe = params["delay_mpe"]
#     tau_a = a - delay_mpe
#     tau_b = b - delay_mpe
    
#     integral = func_int(params, tau_b) - func_int(params, tau_a)
    
#     return integral

# def cdf_root(x, y, in_params, in_domain_start):
#     # Define a function where the root indicates the following...
#     # The x value of the histogram integral corresponding to a specified y value.
#     return func_pulsed_integral(in_params, in_domain_start, x) - y

# def inverse_sample(y, in_params, in_domain_start, in_domain_end, in_total_int):
#     # Randomly sample a two-photon event from the pulsed function by inverse-sampling its integral.
    
#     # The quasi-linearity of the integral allows for decent initial guesses.
#     x_guess = (y / in_total_int) * (in_domain_end - in_domain_start)
    
#     root_result = sco.root_scalar(cdf_root, args=(y, in_params, in_domain_start),
#                                   bracket = [in_domain_start, in_domain_end],
#                                   x0 = x_guess)
    
#     return root_result.root