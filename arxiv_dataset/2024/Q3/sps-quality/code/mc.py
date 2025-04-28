# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 01:00:55 2023

@author: David J. Kedziora
"""

import os

import pandas as pd
import numpy as np
from lmfit import fit_report

from time import time

import load
import calc
import plot

#%% This script loads a set of datasets.
# It creates 'best' least-squares (LS) and maximum a posteriori (MAP) fits.
# It then samples distributions defined by these parameters and fits them again.
# Thus, this Monte Carlo (MC) method gives an idea of parameter uncertainty.

folder_data = "../data/"
folder_plots = "../results/"
folder_saves = "../saves/"

# Only full filename prefixes that contain a listed substring will be loaded.
full_filename_requirements = ["SEQUR"] #["2p5uW_4000cps"]

# Forcibly overwrite the best fits.
# This requires recalculation of its integral.
g_overwrite = None #1e-2
bg_overwrite = None #0

random_seed = 0

constants = {}
constants["duration_snapshot"] = 10     # Single sampling by detectors; unit s.
constants["unit_delay"] = 1e-9          # SI unit for delays; 1 ns.

knowns = {}
knowns["period_pulse"] = 1/80e6         # Inverse of laser frequency in Hz.
knowns["delay_mpe"] = [55e-9, 65e-9]    # Delay range where multi-photon events occur.

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
    param_ids = ["rate_env", "g2_zero", "rate_bg", "decay_peak", "delay_mpe"]
    param_fit_ids = ["value", "stderr"]
    mc_param_fits = dict()
            
    #%% Prepare to fit the full sample for the 'best' parameters.
    sr_best = df_events.sum(axis=1)
    duration_best = len(range_snapshots)*constants["duration_snapshot"]
    sum_per_sec = sr_best.sum()/duration_best
    print("This dataset details %i two-photon events over %i seconds, i.e. %f per second." 
          % (sr_best.sum(), duration_best, sum_per_sec))
    
    # Extract details of the delay domain and create a series of bin edges.
    # Also create a series of delay-bin centres; use this correction for fitting.
    d_delays = sr_delays[1] - sr_delays[0]
    n_delays = len(sr_delays)
    sr_edges = pd.concat([sr_delays, pd.Series([n_delays*d_delays])], ignore_index=True)
    sr_centres = sr_delays + d_delays/2
    
    # Fit by optimising two different definitions of error.
    # Also fit histograms with background deleted for the least-squares case.
    # See calc.py for more information.
    ignore_bg = False
    for use_poisson_likelihood in [False, True, "no_bg"]:
        fit_prefix = "ls_fit"
        fit_label = "LS Fit"
        if use_poisson_likelihood:
            fit_prefix = "p_fit"
            fit_label = "P Fit"
        if use_poisson_likelihood == "no_bg":
            use_poisson_likelihood = False
            ignore_bg = True
            fit_prefix = "no_bg_ls_fit"
            fit_label = "No BG LS Fit"
        if not g_overwrite is None:
            fit_prefix += "_g_%.0e" % g_overwrite
            fit_label += " G %.0e" % g_overwrite
        if not bg_overwrite is None:
            fit_prefix += "_bg_%.0e" % bg_overwrite
            fit_label += " BG %.0e" % bg_overwrite
    
        # Perform the actual 'best' fit.
        print("Generating the best %s for the full dataset." % fit_label)
        fit_best = calc.estimate_g2zero_pulsed(sr_best, sr_centres, knowns, use_poisson_likelihood,
                                               in_duration = duration_best)
        
        sr_fit_best = calc.func_pulsed(fit_best.params, sr_centres,
                                       duration = duration_best)
        plot.plot_event_histogram(sr_best, sr_centres, constants, 
                                  plot_prefix + "_best_" + fit_prefix,
                                  in_label = "Full Experiment: %i Events" % sr_best.sum(),
                                  in_hist_comp = sr_fit_best, 
                                  in_label_comp = "Best " + fit_label,
                                  in_xlim_closeup = xlim_closeup)
        # TODO: Actually log this. Its streamed display is inconvenient.
        print(fit_report(fit_best))

        if not g_overwrite is None:
            print("Overwriting 'g'. Be aware that events will truly be approximate.")
            fit_best.params["g2_zero"].value = g_overwrite
            fit_best.params["g2_zero"].stderr = 0
        if not bg_overwrite is None:
            print("Overwriting 'r_b'. Be aware that events will truly be approximate.")
            fit_best.params["rate_bg"].value = bg_overwrite
            fit_best.params["rate_bg"].stderr = 0
        if any(item is not None for item in [g_overwrite, bg_overwrite]):
            sr_per_sec = calc.func_pulsed(fit_best.params, sr_centres)
            sum_per_sec_new = sr_per_sec.sum()
            print("Prior to the overwrite, best fit suggested %f events per second." % sum_per_sec)
            print("The overwritten fit now suggests %f events per second. Replacing." % sum_per_sec_new)
            sum_per_sec = sum_per_sec_new
        
        if ignore_bg:
            bg_rate = fit_best.params["rate_bg"].value
            bg_events = n_delays*fit_best.params["rate_bg"]*duration_best
            bg_ratio = bg_events/sr_best.sum()
            print("At a best-fit BG rate of %.2E, %i out of %i events (%.2f) are BG noise."
                  % (bg_rate, bg_events, sr_best.sum(), bg_ratio))
            
            print("Zeroing out the BG rate from the best fit.")
            fit_best.params["rate_bg"].value = 0
            fit_best.params["rate_bg"].stderr = 0
            print("Amending average event rate by %.2f: %.2E -> %.2E" %
                  (1 - bg_ratio, sum_per_sec, sum_per_sec*(1 - bg_ratio)))
            sum_per_sec *= (1 - bg_ratio)
        
        num_samples = 250
        for num_events in [0, 1000, 10000, 100000, 1000000]:
            
            num_events_label = str(num_events)
            if num_events == 0:
                num_events = sr_best.sum()
                num_events_label = "same"
            
            if ignore_bg:
                print("Amending number of events per MC sample by %.2f: %i -> %i" %
                      (1 - bg_ratio, num_events, num_events*(1 - bg_ratio)))
                num_events *= (1 - bg_ratio)
        
            try:
                # Attempt to read in previously saved fits.
                # NOTE: There is no check that they are aligned.
                for param_id in param_ids:
                    mc_param_fits[param_id] = pd.read_pickle(save_prefix + "_mc_" + fit_prefix
                                                              + "_sample_size_" + num_events_label
                                                              + "_" + param_id + ".pkl")
                
                print("Previously saved MC %ss for seed %i (%i samples, ~%i events each) loaded." 
                      % (fit_label, random_seed, num_samples, num_events))
            except:
                # Create almost-empty dataframes to store parameter fits for the MC samples.
                # Define the zeroth row as the best fit.
                for param_id in param_ids:
                    mc_param_fits[param_id] = pd.DataFrame(index=range(1 + num_samples), columns=param_fit_ids, dtype=float)
                    mc_param_fits[param_id]["value"][0] = fit_best.params[param_id].value
                    mc_param_fits[param_id]["stderr"][0] = fit_best.params[param_id].stderr
            
                duration_events = num_events/sum_per_sec
                sr_expected = calc.func_pulsed(fit_best.params, sr_centres,
                                               duration = duration_events)
                
                np.random.seed(seed = random_seed)
                
                df_mc = pd.DataFrame(index=sr_centres.index, columns=range(num_samples), dtype=float)
                
                # Each row Poisson-samples how many events are likely to be detected for a given delay.
                t = time()
                for id_delay in sr_delays.index:
                    df_mc.iloc[id_delay] = np.random.poisson(sr_expected[id_delay], num_samples)
                    
                print("%i Poisson-based samples generated, each representing a duration of %f s (~%i events): %f s" 
                      % (num_samples, duration_events, num_events, time() - t))
                
                # Now fit each column, where the column is one sample across all the delays.
                t = time()
                for id_sample in range(num_samples):
                    fit_sample = calc.estimate_g2zero_pulsed(df_mc[id_sample], sr_centres, knowns, use_poisson_likelihood,
                                                             in_duration = duration_events, do_ignore_bg = ignore_bg)
                    # Store the results.
                    # Remember that the zeroth row is the best fit.
                    for param_id in param_ids:
                        mc_param_fits[param_id]["value"][1 + id_sample] = fit_sample.params[param_id].value
                        mc_param_fits[param_id]["stderr"][1 + id_sample] = fit_sample.params[param_id].stderr
                    
                # Plot the last fitted sample as an example.
                sr_fit_sample = calc.func_pulsed(fit_sample.params, sr_centres,
                                                 duration = duration_events)
                plot.plot_event_histogram(df_mc[id_sample], sr_centres, constants,
                                          plot_prefix + str(id_sample) + "_example_mc_" + fit_prefix + "_" + num_events_label,
                                          in_label = "Sample: %i Events, %i s" % (np.sum(df_mc[id_sample]), 
                                                                                  np.round(duration_events)),
                                          in_hist_comp = [sr_expected, sr_fit_sample],
                                          in_label_comp = ["Scaled Best " + fit_label + " (Source)",
                                                           "New " + fit_label],
                                          in_xlim_closeup = xlim_closeup)
                    
                print("%i samples fitted with the %s method: %f s" 
                      % (num_samples, fit_label, time() - t))
                
                # Save the results in pickled format.
                # Compress the sizes by downcasting appropriately.
                for param_id in param_ids:
                    pd.to_numeric(mc_param_fits[param_id]["value"], downcast = "float")
                    pd.to_numeric(mc_param_fits[param_id]["stderr"], downcast = "float")
                    mc_param_fits[param_id].to_pickle(save_prefix + "_mc_" + fit_prefix
                                                      + "_sample_size_" + num_events_label
                                                      + "_" + param_id + ".pkl")