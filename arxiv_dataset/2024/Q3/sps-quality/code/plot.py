# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:52:15 2022

@author: David J. Kedziora
"""

import matplotlib.pyplot as plt
import numpy as np

import numpy.matlib

def plot_event_history(in_df_events, in_axis_time, in_save_prefix):
    # Create a histogram of events across snapshots and plot it.
    df_hist_snapshot = in_df_events.sum(axis=0)
    
    fig_hist_snapshot, ax_hist_snapshot = plt.subplots()
    ax_hist_snapshot.autoscale(enable=True, axis="x", tight=True)
    ax_hist_snapshot.set_xlabel("Time (s)")
    ax_hist_snapshot.set_ylabel("Total Events Per Snapshot (" 
                                + str(in_axis_time[1] - in_axis_time[0]) + " s)")
    ax_hist_snapshot.plot(in_axis_time, df_hist_snapshot, label="Data")
    ax_hist_snapshot.legend()
    fig_hist_snapshot.savefig(in_save_prefix + "_events_per_snapshot.png", 
                              bbox_inches="tight")
    plt.close(fig_hist_snapshot)


def plot_event_histogram(in_hist, in_axis_delays, in_constants, 
                         in_save_prefix, in_label = "Data",
                         in_hist_comp = None, in_label_comp = None,
                         in_xlim_closeup = None, do_only_closeup = True,
                         do_logarithmic_scale = False):
    # Plot a delay-based histogram.
    # Optionally compare it against another function on the same axis.
    # Both in_hist_comp and in_label_comp can be lists.
    
    unit_delay = in_constants["unit_delay"]
    
    fig_sample, ax_sample = plt.subplots()
    ax_sample.plot(in_axis_delays/unit_delay, in_hist, label = in_label)
    if (in_hist_comp is not None) and (in_label_comp is not None):
        if isinstance(in_hist_comp, list):
            for i in range(len(in_hist_comp)):
                ax_sample.plot(in_axis_delays/unit_delay, in_hist_comp[i],
                               label=in_label_comp[i])
        else:
            ax_sample.plot(in_axis_delays/unit_delay, in_hist_comp,
                           label=in_label_comp)
    ax_sample.set_xlabel("Delay (%ss)" % unit_delay)
    events_per_bin = (in_axis_delays[1]-in_axis_delays[0])/unit_delay
    ax_sample.set_ylabel("Detected Events (Bin Size: %s %ss)" % (events_per_bin, unit_delay))
    ax_sample.autoscale(enable=True, axis="x", tight=True)
    ax_sample.legend()
    if do_logarithmic_scale:
        plt.yscale("log")
        ax_sample.legend(loc="upper right", framealpha=1)
    else:
        _, ylim_max = ax_sample.get_ylim()
        ax_sample.set_ylim([0, ylim_max])
    if not (do_only_closeup and in_xlim_closeup is not None):
        fig_sample.savefig(in_save_prefix + "_hist.png",
                           bbox_inches="tight")
    plt.close(fig_sample)
    
    if in_xlim_closeup is not None:
        ax_sample.set_xlim(np.array(in_xlim_closeup)/unit_delay)
        ax_sample.legend()
        fig_sample.savefig(in_save_prefix + "_hist_closeup.png",
                           bbox_inches="tight")
        plt.close(fig_sample)


def plot_param_fits(in_df_param_fits, in_save_prefix):
    # Scatter-plot fitted parameters for different sample sizes.
    # The dataframe must contain these columns: size, value, stderr.
    fig_scatter, ax_scatter = plt.subplots()
    xlim_min = np.min(in_df_param_fits["value"])
    xlim_max = np.max(in_df_param_fits["value"])
    ylim_max = xlim_max - xlim_min
    
    vlim_max = np.log2(np.max(in_df_param_fits["size"]))
    scatter_plot = ax_scatter.scatter(in_df_param_fits["value"], in_df_param_fits["stderr"],
                                      c = np.log2(in_df_param_fits["size"]),
                                      vmin = 0, vmax = vlim_max)
    
    # Plot the points that are beyond the axis limits.
    remnants = in_df_param_fits[in_df_param_fits["stderr"] > ylim_max].copy()
    remnants["stderr"] = ylim_max
    ax_scatter.scatter(remnants["value"], remnants["stderr"], 
                       c = np.log2(remnants["size"]), marker = "^",
                       vmin = 0, vmax = vlim_max)
    
    last_value = in_df_param_fits["value"].iloc[-1]
    last_stderr = in_df_param_fits["stderr"].iloc[-1]
    
    # Plot lines beyond which the estimates are incorrect compared to last fit.
    ax_scatter.plot([last_value - last_stderr, last_value - last_stderr - ylim_max],
                    [0, ylim_max], ":")
    ax_scatter.plot([last_value + last_stderr, last_value + last_stderr + ylim_max],
                    [0, ylim_max], ":")
    
    ax_scatter.set_xlim([xlim_min, xlim_max])
    ax_scatter.set_ylim([0, ylim_max])
    ax_scatter.set_xlabel("Value")
    ax_scatter.set_ylabel("Standard Error")
    fig_scatter.colorbar(scatter_plot, ax = ax_scatter, 
                         label = r"$log_2x$ for $x$ snapshots per sample")
    # plt.colorbar(scatter_plot)
    fig_scatter.savefig(in_save_prefix + ".png",
                        bbox_inches="tight")
    plt.close(fig_scatter)
    
    
def plot_param_fit_averages(in_df_param_fits, in_param_id, in_save_prefix, in_constants):
    # Plot how quickly cumulative averages for fits of different sample sizes stabilise.
    # The dataframe must contain these columns: size, value, stderr.
    
    # The last row is the fitted parameter for a maximally sized sample.
    # That value will be considered the 'best' estimate.
    max_size_sample = in_df_param_fits["size"].iloc[-1]
    value_best = in_df_param_fits["value"].iloc[-1]
    label_best = "Best Est."
    
    number_plots = int(np.log2(max_size_sample))
    
    # Set up the right number of axes.
    # TODO: Improve the size hard coding.
    fig_averages = plt.figure(figsize=(8, number_plots + 2))
    gs = fig_averages.add_gridspec(number_plots, hspace = 0)
    axes_averages = gs.subplots(sharex=True, sharey=True)
    
    for c in range(0, number_plots):
        size_sample = 2**c
        fits_for_size = in_df_param_fits[in_df_param_fits["size"] == size_sample]
        cumulative_average = fits_for_size["value"].expanding().mean()
        cumulative_time = size_sample * (fits_for_size["id"] + 1) * in_constants["duration_snapshot"]
        if c == 1:
            label_best = None
        axes_averages[c].plot([0, max_size_sample * in_constants["duration_snapshot"]], 
                              [value_best, value_best], ":", label = label_best)
        axes_averages[c].plot(cumulative_time, cumulative_average, "-", 
                              label = "Window: " 
                              + str(size_sample * in_constants["duration_snapshot"]) 
                              + " s")
        
        axes_averages[c].ticklabel_format(axis = "y", style = "sci", scilimits = (-2,2))
        axes_averages[c].set_xlim([0, max_size_sample * in_constants["duration_snapshot"]])
        axes_averages[c].legend()
        axes_averages[c].label_outer()
        
        # # Set a standard axis size.
        # # TODO: Improve the size hard coding.
        # axes_averages[c].figure.set_size_inches(6, 1)
        
    fig_averages.supxlabel("Time (s)")
    fig_averages.supylabel("Cumulative Average of Parameter Fits: " + in_param_id)
    fig_averages.savefig(in_save_prefix + "_cumulative_average.png", bbox_inches="tight")
    plt.close(fig_averages)
    


def plot_g2zero_comparison(in_spreads, in_spread_labels, in_save_prefix):

    fig_comp, ax_comp = plt.subplots()
    
    # The spreads argument must be a list of sub-lists.
    # Each sub-list of values is drawn as a column of points in the plot.
    # The sub-list itself should have 1, 3 or 5 values.
    # The middle value is the mean and the adjacent values are one std away.
    # The outer values are the lowest and highest values of the metric.
    # Extra labels must have one item per each sub-list.
    
    c = 0
    label_avg = "Avg"
    label_std = "Avg +- Std"
    label_min = "Min"
    label_max = "Max"
    for spread in in_spreads:
        num_vals = len(spread)
        id_mean = int((num_vals - 1)/2)
        ax_comp.plot(c, spread[id_mean], linestyle = "None", 
                     color = "k", marker = "x", label = label_avg)
        label_avg = None
        if num_vals > 1:
            id_std_low = id_mean - 1
            id_std_high = id_mean + 1
            ax_comp.plot(c, spread[id_std_low], linestyle = "None", 
                         color = "k", marker = "+", label = label_std)
            ax_comp.plot(c, spread[id_std_high], linestyle = "None",
                         color = "k", marker = "+")
            label_std = None
        if num_vals > 3:
            ax_comp.plot(c, spread[-1], linestyle = "None",
                         color = "k", marker = "1", label = label_max)
            ax_comp.plot(c, spread[0], linestyle = "None",
                         color = "k", marker = "2", label = label_min)
            label_min = None
            label_max = None
        c += 1
    
    ax_comp.set_xlim([-0.5, c - 0.5])
    ax_comp.set_xticks(range(c))
    ax_comp.set_xticklabels(in_spread_labels, rotation=45)
    
    ax_comp.legend()
    fig_comp.savefig(in_save_prefix + "_g2zero_comparison.png", bbox_inches="tight")
    plt.close(fig_comp)


# TODO: Standardise code to work with dataframes or numpy arrays.
def plot_traces(in_trace_matrix, in_axis_time, 
                in_save_prefix, in_var_name, in_save_suffix = None,
                in_extra_metrics = None, in_extra_colors = None, in_extra_labels = None,
                in_ylim = None, do_mean_trace_instead = False):
    
    # Extra metrics must be a list of sub-lists.
    # Each sub-list of values is drawn as horizontal lines on the trace plot.
    # The sub-list itself should have 1, 3 or 5 values.
    # The middle value is the mean and the adjacent values are one std away.
    # The outer values are the lowest and highest values of the metric.
    # Extra colors and extra labels must have one item per each sub-list.
    
    # Input trace matrix should have a row for each trajectory.
    # This code transposes it.
    fig_traces, ax_traces = plt.subplots()
    
    colors = plt.cm.viridis(np.linspace(0, 1, in_trace_matrix.shape[0]))
    
    # Plot the mean trace with std.
    if do_mean_trace_instead:
        trace_avg = np.mean(in_trace_matrix)
        trace_std = np.std(in_trace_matrix)
        ax_traces.plot(in_axis_time, trace_avg, color="black", 
               label="%i Trace Mean" % in_trace_matrix.shape[0])
        ax_traces.plot(in_axis_time, trace_avg+trace_std, color="black", linestyle = "--",
                       label="%i Trace Mean +- Std" % in_trace_matrix.shape[0])
        ax_traces.plot(in_axis_time, trace_avg-trace_std, color="black", linestyle = "--")
    else:
        ax_traces.set_prop_cycle("color", colors)
        ax_traces.plot(in_axis_time, np.transpose(np.array(in_trace_matrix)))
    xlim = [in_axis_time[0], in_axis_time[-1]]
    
    # Plot the extra metrics on top of the traces.
    if in_extra_metrics is not None:
        c = 0
        for metric_spread in in_extra_metrics:
            num_vals = len(metric_spread)
            id_mean = int((num_vals - 1)/2)
            label_metric = in_extra_labels[c]
            if num_vals > 1:
                id_std_low = id_mean - 1
                id_std_high = id_mean + 1
                ax_traces.plot(xlim,
                               [[metric_spread[id_std_low], metric_spread[id_std_high]],
                                [metric_spread[id_std_low], metric_spread[id_std_high]]],
                               color = in_extra_colors[c], linestyle = "--")
                label_metric += " + std"
            if num_vals > 3:
                ax_traces.plot(xlim,
                               [[metric_spread[0], metric_spread[-1]],
                                [metric_spread[0], metric_spread[-1]]],
                               color = in_extra_colors[c], linestyle = ":")
                label_metric += " + extrema"
            ax_traces.plot(xlim, 
                           [metric_spread[id_mean], metric_spread[id_mean]], 
                           color = in_extra_colors[c], linestyle = "-",
                           label = label_metric)
            c += 1
            
    ax_traces.set_xlabel("Total Detection Time (s)")
    ax_traces.set_ylabel(in_var_name)
    ax_traces.set_xlim(xlim)
    if not in_ylim is None:
        ax_traces.set_ylim(in_ylim)
    savefile = in_save_prefix + "_trace_" + in_var_name
    if not in_save_suffix is None:
        savefile += "_" + in_save_suffix
    if do_mean_trace_instead:
        savefile += "_mean.png"
    else:
        savefile += ".png"
    if in_extra_labels is not None:
        ax_traces.legend()
    fig_traces.savefig(savefile, bbox_inches="tight")
    plt.close(fig_traces)