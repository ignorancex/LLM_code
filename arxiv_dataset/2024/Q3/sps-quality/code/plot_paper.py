# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:00:21 2023

@author: David J. Kedziora
"""

# import os

import pandas as pd
import numpy as np
# from lmfit import fit_report

# from time import time
# import multiprocessing as mp
# from functools import partial

import load
# import calc
import plot

import seaborn as sns
import matplotlib.pyplot as plt

folder_data = "../data/"
folder_plots = "../results/"
folder_saves = "../saves/"

random_seed = 0

constants = {}
constants["duration_snapshot"] = 10     # Single sampling by detectors; unit s.
constants["unit_delay"] = 1e-9          # SI unit for delays; 1 ns.

knowns = {}
knowns["period_pulse"] = 1/80e6         # Inverse of laser frequency in Hz.
knowns["delay_mpe"] = [55e-9, 65e-9]    # Delay range where multi-photon events occur.

#%% Examine comparison between normal fit and Poisson-noise fit.

folder_prefix = "InGaAs-GaAs QDs\\FI-SEQUR project demonstrator sample\\"
filename_prefixes = ["1p2uW_3000cps_time bin width 128 ps",
                     "2p5uW_4000cps",
                     "4uW_4100cps",
                     "8uW_5100cps",
                     "10uW_6000cps",
                     "10uW_12000cps",
                     "20uW_7000cps",
                     "30uW_7000cps"]
filename_labels = ["1p2uW", "2p5uW", "4uW", "8uW", "10uW-", "10uW+", "20uW", "30uW"]
total_event_list = []

# Determine fitting parameters of interest.
fit_prefixes = ["ls_fit", "p_fit", "no_bg_ls_fit", "ls_fit_g_1e-02_bg_0e+00", "ls_fit_g_1e-03_bg_0e+00"]
fit_labels = ["LS", "P", "No BG", "Low G (0.01)", "Low G (0.001)"]
num_event_ids = ["1000000", "100000", "0", "10000", "1000"]
num_event_labels = ["1e6", "1e5", "Same", "1e4", "1e3"]

param_ids = ["g2_zero", "delay_mpe", "rate_bg", "rate_env", "decay_peak"]
param_fit_ids = ["value", "stderr"]
mc_results = dict()

plot_prefix = folder_plots

# Set up a zoom for histogram plots.
xlim_closeup = [np.mean(knowns["delay_mpe"]) - knowns["period_pulse"]*3/2,
                np.mean(knowns["delay_mpe"]) + knowns["period_pulse"]*3/2]

#%% Compile all data in one structure.

for filename_prefix, filename_label in zip(filename_prefixes, filename_labels):

    full_filename_prefix = folder_prefix + filename_prefix

    df_events, sr_delays, range_snapshots = load.load_experiment(folder_data, full_filename_prefix, constants)
    
    sr_best = df_events.sum(axis=1)
    total_event_list.append(sr_best.sum())
    duration_best = len(range_snapshots)*constants["duration_snapshot"]
    sum_per_sec = sr_best.sum()/duration_best
    print("This dataset details %i two-photon events over %i seconds, i.e. %f per second." 
          % (sr_best.sum(), duration_best, sum_per_sec))
    
    # Extract details of the delay domain and create a series of bin edges.
    # Also create a series of delay-bin centres; use this correction for fitting.
    d_delays = sr_delays[1] - sr_delays[0]
    n_delays = len(sr_delays)
    range_delays = (sr_delays.iloc[-1] + d_delays) - sr_delays[0]
    sr_edges = pd.concat([sr_delays, pd.Series([range_delays])], ignore_index=True)
    sr_centres = sr_delays + d_delays/2
    print("%s: %.2E range with bin size %.2E" % (filename_label, range_delays, d_delays))
    
    if filename_label == "4uW":
        plot.plot_event_history(df_events, range_snapshots, plot_prefix + "signal_" + filename_label)
        plot.plot_event_histogram(sr_best, sr_centres, constants, 
                                  plot_prefix + "example_" + filename_label,
                                  in_label = "After %s s: %i Events" % (duration_best, sr_best.sum()),
                                  in_hist_comp = df_events.iloc[:,0], 
                                  in_label_comp = "After %s s: %i Events" % (constants["duration_snapshot"], df_events.iloc[:,0].sum()),
                                  in_xlim_closeup = xlim_closeup,
                                  do_only_closeup = False,
                                  do_logarithmic_scale = True)
    
    # Identify save/load destination for any plots and results.
    # plot_prefix = folder_plots + full_filename_prefix
    save_prefix = folder_saves + full_filename_prefix + "_seed_" + str(random_seed)

    mc_results[filename_label] = dict()
    for fit_prefix in fit_prefixes:
        mc_results[filename_label][fit_prefix] = dict()
        for num_events in num_event_ids:
            temp_label = str(num_events)
            if num_events == "0":
                temp_label = "same"
            mc_results[filename_label][fit_prefix][num_events] = dict()
            for param_id in param_ids:
                try:
                    mc_results[filename_label][fit_prefix][num_events][param_id] = pd.read_pickle(save_prefix + "_mc_" + fit_prefix
                                                                                                + "_sample_size_" + temp_label
                                                                                                + "_" + param_id + ".pkl")
                except:
                    continue
            
                if num_events == "1000000":
                    best_value = mc_results[filename_label][fit_prefix][num_events][param_id]["value"][0]
                    best_stderr = mc_results[filename_label][fit_prefix][num_events][param_id]["stderr"][0]
                    print("Best %s %s %s: \\num{%.3E} (\\pm %.2f\\%%)" % (filename_label, fit_prefix, param_id,
                                                              best_value, 100.0*best_stderr/best_value))
                    
                if param_id == "g2_zero" and num_events in ["1000000", "100000", "0", "10000", "1000"]:
                    val_mean = mc_results[filename_label][fit_prefix][num_events]["g2_zero"]["value"][1:].mean()
                    val_std = mc_results[filename_label][fit_prefix][num_events]["g2_zero"]["value"][1:].std()
                    print("Histograms %s %s %s: %.3E, %.3E" % (filename_label, fit_prefix, num_events,
                                                             val_mean, val_std))
            
# # Clean.
# is_stderr_too_big = mc_results["pnoise_fits"]["100"]["bg"]["stderr"]>1
# for param_id in param_ids:
#     df_temp = mc_results["pnoise_fits"]["100"][param_id]
#     df_temp.drop(df_temp[is_stderr_too_big].index, inplace=True)

#%% Create plots as required.
# plotted_filename_labels = ["1p2uW", "30uW"]#,"1p2uW", "2p5uW", "4uW", "8uW", "10uW-", "10uW+", "20uW", "30uW"]
# plotted_fit_prefixes = ["no_bg_ls_fit", "ls_fit"]
# plotted_num_event_ids = ["10000"]
    
def plot_scatter(plotted_filename_labels, plotted_fit_prefixes, plotted_num_event_ids,
                 filename_plot,
                 do_reverse_hues = False, do_no_filename = False,
                 do_reverse_legend = False, do_move_legend = False,
                 x_lim = None, y_lim = None):
    for param_id in param_ids:
        
        df_plot = pd.DataFrame(columns=param_fit_ids)
        
        for count_filename in range(len(filename_labels)):
            filename_label = filename_labels[count_filename]
            total_events = total_event_list[count_filename]
            if filename_label in plotted_filename_labels:
                filename_label_used = filename_label + " "
                if do_no_filename:
                    filename_label_used = ""
                
                for count_fit_prefix in range(len(fit_prefixes)):
                    fit_prefix = fit_prefixes[count_fit_prefix]
                    if fit_prefix in plotted_fit_prefixes:
                        fit_label = fit_labels[count_fit_prefix]
                        
                        if not fit_prefix == "no_bg_ls_fit":
                            # First row of any sample size is the original best fit.
                            df_temp = mc_results[filename_label][fit_prefix][num_event_ids[0]][param_id]
                            df_plot = pd.concat([df_plot,
                                                 df_temp.loc[0].to_frame().T.assign(Fit=filename_label_used + fit_label + " Best", Best="1")], 
                                                ignore_index=True)
                        
                        for count_num_events in range(len(num_event_ids)):
                            num_events = num_event_ids[count_num_events]
                            num_event_label = num_event_labels[count_num_events]
                            if num_events in plotted_num_event_ids:
                                df_temp = mc_results[filename_label][fit_prefix][num_events][param_id]
                                df_plot = pd.concat([df_plot,
                                                     df_temp.loc[1:].assign(Fit=filename_label_used + fit_label + " " + num_event_label, Best="0")],
                                                    ignore_index=True)
        
        g = None
        if do_reverse_hues:
            g = sns.JointGrid(data=df_plot.iloc[::-1], x="value", y="stderr", hue="Fit", ratio=3)
        else:
            g = sns.JointGrid(data=df_plot, x="value", y="stderr", hue="Fit", ratio=3)
        
        g.plot_joint(sns.scatterplot)
        if do_reverse_legend:
            handles, labels = g.ax_joint.get_legend_handles_labels()
            g.ax_joint.legend(handles[::-1], labels[::-1], title="Fit")
        sns.boxplot(df_plot, y=g.hue, x=g.x, ax=g.ax_marg_x)
        g.ax_marg_y.remove()
        if not x_lim is None:
            g.ax_joint.set_xlim(x_lim)
            if x_lim[1] < 0.1:
                g.ax_joint.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        if not y_lim is None:
            g.ax_joint.set_ylim(y_lim)
        if param_id == "g2_zero":
            g.set_axis_labels(xlabel="Value: g", ylabel="Standard Error: g")
            if do_move_legend:
                sns.move_legend(g.ax_joint, "upper left", bbox_to_anchor=(1, 1))
            else:
                sns.move_legend(g.ax_joint, "best")
            g.savefig(plot_prefix + filename_plot + ".png", bbox_inches="tight")
        plt.close()

plot_scatter(plotted_filename_labels = ["2p5uW"],
             plotted_fit_prefixes = ["ls_fit_g_1e-02_bg_0e+00"],
             plotted_num_event_ids = ["1000000", "100000", "10000", "1000"],
             filename_plot = "mc_low_g_1e-2_spread",
             do_reverse_hues = True,
             do_no_filename = True,
             x_lim = (0, 2e-2),
             y_lim = (0, 0.16))

plot_scatter(plotted_filename_labels = ["2p5uW"],
             plotted_fit_prefixes = ["ls_fit_g_1e-03_bg_0e+00"],
             plotted_num_event_ids = ["1000000", "100000", "10000", "1000"],
             filename_plot = "mc_low_g_1e-3_spread",
             do_reverse_hues = True,
             do_no_filename = True,
             x_lim = (0, 2e-3),
             y_lim = (0, 0.16))

plot_scatter(plotted_filename_labels = ["2p5uW"],
             plotted_fit_prefixes = ["ls_fit"],
             plotted_num_event_ids = ["1000000", "100000", "0", "10000", "1000"],
             filename_plot = "mc_2p5uW_spread",
             do_reverse_hues = True)

plot_scatter(plotted_filename_labels = ["1p2uW", "2p5uW", "4uW", "8uW", "10uW-", "10uW+", "20uW", "30uW"],
             plotted_fit_prefixes = ["ls_fit"],
             plotted_num_event_ids = ["100000"],
             filename_plot = "mc_sample_variance",
             do_move_legend = True)   
         
plot_scatter(plotted_filename_labels = ["10uW-", "30uW"],
             plotted_fit_prefixes = ["ls_fit", "p_fit"],
             plotted_num_event_ids = ["1000000"],
             filename_plot = "issues_poisson")

plot_scatter(plotted_filename_labels = ["1p2uW", "30uW"],
             plotted_fit_prefixes = ["no_bg_ls_fit", "ls_fit"],
             plotted_num_event_ids = ["10000"],
             filename_plot = "mc_no_bg",
             do_reverse_hues = True,
             # do_reverse_legend = True,
             do_move_legend = True)
        
#%% Examine expanding averages.

num_perms = 1500

fig_expand, ax_expand = plt.subplots()

for filename_label, total_event_number in zip(filename_labels, total_event_list):
    if filename_label not in ["1p2uW", "2p5uW", "4uW", "8uW", "20uW", "30uW"]:
        continue
    
    sr_1000 = mc_results[filename_label]["ls_fit"]["1000"]["g2_zero"]["value"][1:]
    sr_same = mc_results[filename_label]["ls_fit"]["0"]["g2_zero"]["value"][1:]
    val_best = mc_results[filename_label]["ls_fit"]["0"]["g2_zero"]["value"][0]
    
    np.random.seed(seed = random_seed)
    
    df_perms = pd.DataFrame(index=range(len(sr_1000)), columns=range(num_perms), dtype=float)
    
    for count_perm in range(num_perms):
        df_perms[count_perm] = sr_1000.sample(n=250, ignore_index=True)
    
    df_expand = df_perms.expanding().mean()
    
    expand_mean = df_expand.T.mean()
    expand_std = df_expand.T.std()
    ax_expand.fill_between((df_expand.index+1)*1000, expand_mean - expand_std, expand_mean + expand_std, alpha=0.2)
    ax_expand.plot((df_expand.index+1)*1000, expand_mean, label = filename_label)
    
    ax_expand.plot((df_expand.index+1)*1000, sr_1000.expanding().mean(), "k:")
    
    ax_expand.plot(total_event_number, sr_same.mean() + sr_same.std(), "k_")
    ax_expand.plot([total_event_number, total_event_number], 
                   [sr_same.mean() - sr_same.std(), sr_same.mean() + sr_same.std()], "k-")
    ax_expand.plot(total_event_number, sr_same.mean() - sr_same.std(), "k_")
    ax_expand.plot(total_event_number, val_best, "k+")
    
ax_expand.set_xlim([0, 75000])
ax_expand.set_ylim([0.1, 0.9])
ax_expand.set_xlabel("Detected Events")
ax_expand.set_ylabel("Value: g")
handles, labels = ax_expand.get_legend_handles_labels()
ax_expand.legend(reversed(handles), reversed(labels), loc = "upper right", framealpha = 1)
fig_expand.savefig(plot_prefix  + "expanding.png", bbox_inches="tight")
plt.close(fig_expand)