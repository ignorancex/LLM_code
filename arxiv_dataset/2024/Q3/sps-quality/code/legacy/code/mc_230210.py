# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:56:24 2023

@author: David J. Kedziora
"""

import os

import pandas as pd
import numpy as np
from lmfit import fit_report

from time import time
import multiprocessing as mp
from functools import partial

import load
import calc
import plot

import matplotlib.pyplot as plt

def main():
    #%% This script loads a set of datasets.
    # It creates 'best' least-squares (LS) and maximum a posteriori (MAP) fits.
    # It then samples distributions defined by these parameters and fits them again.
    # Thus, this Monte Carlo (MC) method gives an idea of parameter uncertainty.
    
    folder_data = "../data/"
    folder_plots = "../results/"
    folder_saves = "../saves/"
    
    # Only full filename prefixes that contain a listed substring will be loaded.
    full_filename_requirements = ["SEQUR"]
    # full_filename_requirements = ["10uW"]
    # full_filename_requirements = ["1p2uW"]
    
    random_seed = 0
    
    # Given a sample size, how many such samples to take during the MC process.
    sample_size_iterations = [(1000000, 100),
                              (100000, 300),
                              (10000, 1000),
                              (1000, 3000)]
    
    constants = {}
    constants["duration_snapshot"] = 10     # Single sampling by detectors; unit s.
    constants["unit_delay"] = 1e-9          # SI unit for delays; 1 ns.
    
    knowns = {}
    knowns["period_pulse"] = 1/80e6         # Inverse of laser frequency in Hz.
    knowns["delay_mpe"] = [55e-9, 65e-9]    # Delay range where multi-photon events occur.
    
    # Set a hardware dependent sample size above which sampling uses multiprocessing.
    # This avoids multiprocessing overhead on small samplings.
    sample_size_threshold_single_process = 3000
    print("CPU Count: %i" % mp.cpu_count())
    
    #%% Loop through all the datasets available, as long as they meet requirement.
    full_filename_prefixes = load.get_full_filename_prefixes(folder_data)
    
    for full_filename_prefix in full_filename_prefixes:
        
        # Do not load filenames that do not contain a required substring.
        if not ((full_filename_requirements is None) or (len(full_filename_requirements) == 0)):
            is_filename_skippable = True
            
            for requirement in full_filename_requirements:
                if requirement in full_filename_prefix:
                    is_filename_skippable = False
                    
            if is_filename_skippable:
                continue
        
        # Load the experiment.
        df_events, sr_delays, range_snapshots = load.load_experiment(folder_data, full_filename_prefix, constants)
        
        # Prepare save destination for any plots and results.
        plot_prefix = folder_plots + full_filename_prefix
        save_prefix = folder_saves + full_filename_prefix + "_seed_" + str(random_seed)
        
        folder_prefix, filename_prefix = os.path.split(full_filename_prefix)
        for folder_base in [folder_plots, folder_saves]:
            if not os.path.exists(os.path.join(folder_base, folder_prefix)):
                os.makedirs(os.path.join(folder_base, folder_prefix))
                
        # Set up a zoom for histogram plots.
        xlim_closeup = [np.mean(knowns["delay_mpe"]) - knowns["period_pulse"]*3/2,
                        np.mean(knowns["delay_mpe"]) + knowns["period_pulse"]*3/2]
        
        # Determine fitting parameters of interest.
        param_ids = ["amp_env", "amp_ratio", "bg", "decay_peak", "delay_mpe"]
        param_fit_ids = ["value", "stderr"]
        mc_param_fits = dict()
                
        #%% Prepare to fit the full sample for the 'best' parameters.
        sr_best = df_events.sum(axis=1)
        print("This dataset details %i two-photon events over %i seconds." 
              % (sr_best.sum(), len(range_snapshots)*constants["duration_snapshot"]))
        sr_best /= sr_best.sum()
        
        # Extract details of the delay domain and create a series of bin edges.
        # Also create a series of delay-bin centres; use this correction for fitting.
        d_delays = sr_delays[1] - sr_delays[0]
        n_delays = len(sr_delays)
        sr_edges = pd.concat([sr_delays, pd.Series([n_delays*d_delays])], ignore_index=True)
        sr_centres = sr_delays + d_delays/2
        
        # Fit by optimising two different definitions of error.
        # See calc.py for more information.
        for use_poisson_likelihood in [False]:#, True]:
            fit_prefix = "fits"
            fit_label = "Fit"
            if use_poisson_likelihood:
                fit_prefix = "pnoise_fits"
                fit_label = "PNoise Fit"
        
            # Perform the actual 'best' fit.
            fit_best = calc.estimate_g2zero_pulsed(sr_best, sr_centres, knowns, use_poisson_likelihood)
                            
            sr_fit = calc.func_pulsed(fit_best.params, sr_centres)
            plot.plot_event_histogram(sr_best, sr_centres, constants, 
                                      plot_prefix + "_best_fit",
                                      in_label = "Full Experiment: %i Events" % df_events.sum(axis=1).sum(),
                                      in_hist_comp = sr_fit, 
                                      in_label_comp = "Best Fit",
                                      in_xlim_closeup = xlim_closeup)
            # TODO: Actually log this. Its display is currently almost useless.
            print(fit_report(fit_best))
            
            # Plot the integral of the histogram function with 'best' parameters.
            sr_fit_int = calc.func_pulsed_integral(fit_best.params, sr_edges[0], sr_edges)
            
            fig_int, ax_int = plt.subplots()
            ax_int.plot(sr_edges, sr_fit_int, label="~CDF")
            ax_int.set_xlabel("Delay (%ss)" % constants["unit_delay"])
            ax_int.set_ylabel("Integral")
            ax_int.legend()
            fig_int.savefig(plot_prefix + "_best_fit_integrated.png", bbox_inches="tight")
            
            plt.close(fig_int)
            
            # Calculate total integral of the continuous distribution.
            total_int = calc.func_pulsed_integral(fit_best.params, sr_edges[0], sr_edges.iloc[-1])
            
            #%% Begin sample generation.
            for num_events, num_iterations in sample_size_iterations:
                print("Attempting to generate %i samples of size %i." % (num_iterations, num_events))
                
                np.random.seed(seed = random_seed)
                
                num_iterations_done = 0
                try:
                    # Attempt to read in previously saved fits.
                    # NOTE: There is no check that they are aligned.
                    for param_id in param_ids:
                        mc_param_fits[param_id] = pd.read_pickle(save_prefix + "_mc_" + fit_prefix
                                                                 + "_sample_size_" + str(num_events)
                                                                 + "_" + param_id + ".pkl")
                    # Remember, the zeroth row is the best fit.
                    num_iterations_done = mc_param_fits[param_id].shape[0] - 1
                    
                    print("Previously saved MC %s for seed %i loaded." % (fit_prefix.replace("_", " "), random_seed))
                    print("Note: %i samples out of a desired %i are already generated." 
                          % (num_iterations_done, num_iterations))
                except:
                    # Create almost-empty dataframes to store parameter fits for the MC samples.
                    # Define the zeroth row as the best fit.
                    for param_id in param_ids:
                        mc_param_fits[param_id] = pd.DataFrame(np.zeros([1, len(param_fit_ids)]))
                        mc_param_fits[param_id].columns = param_fit_ids
                        mc_param_fits[param_id]["value"][0] = fit_best.params[param_id].value
                        mc_param_fits[param_id]["stderr"][0] = fit_best.params[param_id].stderr
                
                if num_iterations > num_iterations_done:
                    
                    # Extend the dataframes by however many iterations need to be done.
                    num_iterations_left = num_iterations - num_iterations_done
                    for param_id in param_ids:
                        mc_param_fits[param_id] = pd.concat([mc_param_fits[param_id], 
                                                             pd.DataFrame(np.zeros([num_iterations_left, len(param_fit_ids)]), 
                                                                          columns=param_fit_ids)], 
                                                            ignore_index=True)
                        
                    print("The first %i samples of the generation process will be skipped." % (num_iterations_done))
                    t = time()
                    
                    for id_iteration in range(num_iterations):
                        
                        # Uniformly sample possible integral values within the delay domain.
                        y_sample = np.random.uniform(low=0.0, high=total_int, size=num_events)
                        
                        if id_iteration < num_iterations_done:
                            continue
                        
                        results = np.zeros(num_events)
                        label_process = "single"
                        # Use multiprocessing if generating a lot of data.
                        if num_events > sample_size_threshold_single_process:
                            label_process = "multi"
                            with mp.Pool(processes = (mp.cpu_count() - 1)) as pool:
                                results = pool.map(partial(calc.inverse_sample,
                                                           in_params = fit_best.params,
                                                           in_domain_start = sr_edges[0],
                                                           in_domain_end = sr_edges[0] + sr_edges.iloc[-1],
                                                           in_total_int = total_int),
                                                   y_sample)
                        
                        else:
                            for k in range(num_events):
                                y_random = y_sample[k]
                                
                                results[k] = calc.inverse_sample(y_random, 
                                                                 in_params = fit_best.params, 
                                                                 in_domain_start = sr_edges[0],
                                                                 in_domain_end = sr_edges[0] + sr_edges.iloc[-1],
                                                                 in_total_int = total_int)
                        
                        # The binned results should have a length one less than the number of bin edges.
                        array_sample = np.histogram(results, bins=sr_edges)[0].astype(float)
                        array_sample /= np.sum(array_sample)
                        
                        # Fit the sample and store the results.
                        # Remember that the zeroth row is the best fit.
                        fit_sample = calc.estimate_g2zero_pulsed(array_sample, sr_centres, knowns, use_poisson_likelihood)
                        for param_id in param_ids:
                            mc_param_fits[param_id]["value"][1 + id_iteration] = fit_sample.params[param_id].value
                            mc_param_fits[param_id]["stderr"][1 + id_iteration] = fit_sample.params[param_id].stderr
                        
                    print("%i two-photon events inverse-sampled from integral (%s-process), repeated %i times: %f s" 
                          % (num_events, label_process, num_iterations_left, time() - t))
                    
                    plot.plot_event_histogram(array_sample, sr_centres, constants, 
                                              plot_prefix + "_example_mc_" + str(num_events),
                                              in_label = "Sample: %i Events" % num_events,
                                              in_hist_comp = sr_fit, 
                                              in_label_comp = "Best Fit (Sample Source)",
                                              in_xlim_closeup = xlim_closeup)
                    
                    # Save the results in pickled format.
                    # Compress the sizes by downcasting appropriately.
                    for param_id in param_ids:
                        pd.to_numeric(mc_param_fits[param_id]["value"], downcast = "float")
                        pd.to_numeric(mc_param_fits[param_id]["stderr"], downcast = "float")
                        mc_param_fits[param_id].to_pickle(save_prefix + "_mc_" + fit_prefix
                                                          + "_sample_size_" + str(num_events)
                                                          + "_" + param_id + ".pkl")
                        
    # Return the last fits.
    return mc_param_fits
        
if __name__ == '__main__':
    mc_param_fits = main()