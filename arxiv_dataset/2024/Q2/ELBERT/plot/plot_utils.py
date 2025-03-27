import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# for smoothing
from scipy.ndimage.filters import gaussian_filter1d


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def collect_data_csv(exp_path, 
                     seeds,
                     timesteps=245, 
                     read_sd=False,
                     seed_str='expindex'
):
    results = np.zeros((seeds, timesteps, 7))
    actual_len = timesteps
    count = 0
    for seed in range(seeds):
        try:
            data_pth  = os.path.join("%s_%s_%d" % (exp_path, seed_str, seed) ,'eval.csv')
            data = pd.read_csv(data_pth, sep=',', header=0)
            num_samples = data['num_timesteps'].to_numpy()
            return_arr = data['return'].to_numpy()
            bias = data['bias'].to_numpy()

            actual_len = min(bias.size, timesteps)
            results[count, :actual_len, 0] = num_samples[:actual_len]
            results[count, :actual_len, 1] = return_arr[:actual_len]
            results[count, :actual_len, 2] = bias[:actual_len]

            if read_sd:
                supply_max = data['supply_max'].to_numpy()
                supply_min = data['supply_min'].to_numpy()
                demand_max = data['demand_max'].to_numpy()
                demand_min = data['demand_min'].to_numpy()
                results[count, :actual_len, 3] = supply_max[:actual_len]
                results[count, :actual_len, 4] = supply_min[:actual_len]
                results[count, :actual_len, 5] = demand_max[:actual_len]
                results[count, :actual_len, 6] = demand_min[:actual_len]
            count += 1
        except:
            print("%s, exp index %d failed!" % (exp_path, seed))
    results = results[:count,:actual_len,:]
    return results


def bootstrapping(data, num_per_group, num_of_group):
    new_data = np.array([np.mean(np.random.choice(data, num_per_group, replace=True)) for _ in range(num_of_group)])
    return new_data


def generate_confidence_interval(ys, number_per_g = 30, number_of_g = 100, low_percentile = 10, high_percentile = 90):
    means = []
    mins =[]
    maxs = []
    for i,y in enumerate(ys.T):
        y = bootstrapping(y, number_per_g, number_of_g)
        means.append(np.mean(y))
        mins.append(np.percentile(y, low_percentile))
        maxs.append(np.percentile(y, high_percentile))
    return np.array(means), np.array(mins), np.array(maxs)


def plot_ci(plt, x, y, num_runs, num_dots, linestyle='-', linewidth=3, transparency=0.2, c='red', sigma=5.0, label=None):
    assert (x.ndim==1) and (y.ndim==2)
    assert(x.size==num_dots) and (y.shape==(num_runs,num_dots))
    y_mean, y_min, y_max = generate_confidence_interval(y)
    y_mean = gaussian_filter1d(y_mean, sigma=sigma)
    y_max = gaussian_filter1d(y_max, sigma=sigma)
    y_min = gaussian_filter1d(y_min, sigma=sigma)
    plt.plot(x, y_mean, linestyle=linestyle, linewidth=linewidth, color=c, label=label)
    plt.fill_between(x, y_min, y_max, alpha=transparency, color=c)
    return


def scatter_ci(plt, x, y, num_runs, num_dots, marker='.', s=5, c='red', sigma=5.0, label=None, early_return=False):
    assert (x.ndim==2) and (y.ndim==2)
    assert(x.shape==(num_runs,num_dots)) and (y.shape==(num_runs,num_dots))
    x_mean, x_min, x_max = generate_confidence_interval(x)
    y_mean, y_min, y_max = generate_confidence_interval(y)
    x_mean = gaussian_filter1d(x_mean, sigma=sigma)
    y_mean = gaussian_filter1d(y_mean, sigma=sigma)
    x_max = gaussian_filter1d(x_max, sigma=sigma)
    y_max = gaussian_filter1d(y_max, sigma=sigma)
    x_min = gaussian_filter1d(x_min, sigma=sigma)
    y_min = gaussian_filter1d(y_min, sigma=sigma)
    if early_return:
        return x_mean[-1], y_mean[-1]
    x_error = [[x_mean[-1]-x_min[-1]], [x_max[-1]-x_mean[-1]]]
    y_error = [[y_mean[-1]-y_min[-1]], [y_max[-1]-y_mean[-1]]]
    plt.errorbar(x_mean[-1], y_mean[-1], xerr=x_error, yerr=y_error, fmt='o', lw=1, ms=s, c=c, mec=c, mfc=c, label=label)
    return


def load_and_plot(exp_dict,
                  plot_type,
                  xlim=None, 
                  ylim=None,
                  timesteps=490,
                  legend_loc=None,
                  sigma_smooth=5.0,
                  seed_str='expindex'
):
    assert plot_type=='Rewards' or plot_type=='Bias', "plot_type should be either 'Rewards' or 'Bias'." 
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
    })
    fig = plt.figure(figsize=(6.5, 6))

    plt1 = fig.add_subplot(1, 1, 1)
    plt1.set_ylabel(plot_type, fontsize=20)
    plt1.set_xlabel('Training timestep', fontsize=20)

    for model in exp_dict.keys():
        data = collect_data_csv(exp_dict[model]["dir"], exp_dict[model]["seeds"], timesteps, seed_str=seed_str)
        indexes = data[:, :, 0]
        values = data[:, :, 1] if plot_type=='Rewards' else data[:, :, 2]
        values = gaussian_filter1d(values, sigma=sigma_smooth)
        num_runs = indexes.shape[0]
        num_dots = indexes.shape[1]
        if plot_type=="Bias": values = np.abs(values)
        plot_ci(plt1, indexes[0], values, num_runs=num_runs, num_dots=num_dots, linewidth=2, c=exp_dict[model]["color"], label=exp_dict[model]["label"])
    plt1.grid()
    if ylim is not None: 
        plt1.set_ylim((ylim[0], ylim[1]))
        yticks = np.linspace(ylim[0], ylim[1], 6)
        plt1.set_yticks(yticks, fontsize=12)
        yticklabels = ["$%.2f$"%tick for tick in yticks]
        plt1.set_yticklabels(yticklabels, fontsize=12)
    if xlim is not None: 
        plt1.set_xlim((xlim[0], xlim[1]))
        xticks = np.linspace(xlim[0], xlim[1], 6)
        plt1.set_xticks(xticks, fontsize=12)
        plt1.tick_params(axis='x', labelsize=12)
    if legend_loc is not None: 
        plt1.legend(loc=legend_loc, fontsize=16)

    return fig


def load_and_plot_sd(exp_dict,
                     exp_name,
                     xlim=None, 
                     ylim=None,
                     timesteps=245,
                     legend_loc=None,
                     sigma_smooth=5.0,
                     seed_str='expindex'
):
    assert (exp_name in exp_dict.keys()), "exp_name '%s' is not the key of exp_dict." % exp_name

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
    })
    fig = plt.figure(figsize=(6.5, 6))

    plt1 = fig.add_subplot(1, 1, 1)
    plt1.set_ylabel('Supply and denamd', fontsize=20)
    plt1.set_xlabel('Training timestep', fontsize=20)

    data = collect_data_csv(exp_dict[exp_name]['dir'], exp_dict[exp_name]['seeds'], timesteps, read_sd=True, seed_str=seed_str)
    indexes = data[:, :, 0]
    s0 = gaussian_filter1d(data[:, :, 3], sigma=sigma_smooth)
    s1 = gaussian_filter1d(data[:, :, 4], sigma=sigma_smooth)
    d0 = gaussian_filter1d(data[:, :, 5], sigma=sigma_smooth)
    d1 = gaussian_filter1d(data[:, :, 6], sigma=sigma_smooth)
    num_runs = indexes.shape[0]
    num_dots = indexes.shape[1]
    plot_ci(plt1, indexes[0], s0, num_runs=num_runs, num_dots=num_dots, linewidth=2, c='blue', label='Supply_0')
    plot_ci(plt1, indexes[0], s1, num_runs=num_runs, num_dots=num_dots, linewidth=2, c='red', label='Supply_1')
    plot_ci(plt1, indexes[0], d0, num_runs=num_runs, num_dots=num_dots, linewidth=2, linestyle='--', c='blue', label='Demand_0')
    plot_ci(plt1, indexes[0], d1, num_runs=num_runs, num_dots=num_dots, linewidth=2, linestyle='--', c='red', label='Demand_1')
    plt1.grid()
    if ylim is not None: 
        plt1.set_ylim((ylim[0], ylim[1]))
        yticks = np.linspace(ylim[0], ylim[1], 6)
        plt1.set_yticks(yticks, fontsize=12)
        plt1.tick_params(axis='y', labelsize=12)
        #yticklabels = ["$%.2f"%tick for tick in yticks]
        #plt1.set_yticklabels(yticklabels, fontsize=12)
    if xlim is not None: 
        plt1.set_xlim((xlim[0], xlim[1]))
        xticks = np.linspace(xlim[0], xlim[1], 6)
        plt1.set_xticks(xticks, fontsize=12)
        plt1.tick_params(axis='x', labelsize=12)
    if legend_loc is not None: 
        plt1.legend(loc=legend_loc, fontsize=16)
    
    return fig


def load_and_scatter(exp_dict,
                     rewards_lim=None,
                     bias_lim=None, 
                     timesteps=245, 
                     legend_loc=None, 
                     s=5, 
                     sigma_smooth=5.0,
                     seed_str='expindex'):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
    })
    fig = plt.figure(figsize=(6.5, 6))

    plt1 = fig.add_subplot(1, 1, 1)
    plt1.set_xlabel('Bias', fontsize=20)
    plt1.set_ylabel('Rewards', fontsize=20)

    for model in exp_dict.keys():
        data = collect_data_csv(exp_dict[model]["dir"], exp_dict[model]["seeds"], timesteps, seed_str=seed_str)
        indexes = data[:, :, 0]
        rewards = data[:, :, 1]
        bias = data[:, :, 2]
        rewards = gaussian_filter1d(rewards, sigma=sigma_smooth)
        bias = gaussian_filter1d(bias, sigma=sigma_smooth)
        num_runs = indexes.shape[0]
        num_dots = indexes.shape[1]
        scatter_ci(plt1, bias, rewards, num_runs=num_runs, num_dots=num_dots, marker=exp_dict[model]["marker"], s=s, c=exp_dict[model]["color"], label=exp_dict[model]["label"])
    plt1.grid()
    if rewards_lim is not None: 
        plt1.set_ylim((rewards_lim[0], rewards_lim[1]))
        yticks = np.linspace(rewards_lim[0], rewards_lim[1], 6)
        plt1.set_yticks(yticks, fontsize=12)
        yticklabels = ["$%.2f$"%tick for tick in yticks]
        plt1.set_yticklabels(yticklabels, fontsize=12)
    if bias_lim is not None: 
        plt1.set_xlim((bias_lim[0], bias_lim[1]))
        xticks = np.linspace(bias_lim[0], bias_lim[1], 6)
        plt1.set_xticks(xticks, fontsize=12)
        xticklabels = ["$%.2f$"%tick for tick in xticks]
        plt1.set_xticklabels(xticklabels, fontsize=12)
    if legend_loc is not None:
        plt1.legend(loc=legend_loc, fontsize=16)
    
    return fig
