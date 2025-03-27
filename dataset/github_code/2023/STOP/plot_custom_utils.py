import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

import pdb
from matplotlib import rcParams
import os
import itertools
import yaml

from scipy.stats import binom

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

def compute_stats(method, errors, plotting_stat = 'abs', print_log = False):

    errors = np.array(errors)
    n = len(errors) # trials

    print ('number of trials for {} {}'.format(method, n))
    if errors.ndim == 2:
        if plotting_stat == 'iqm_abs':
            errors_sorted = np.sort(errors, axis = 0)
            n = errors.shape[0]
            errors = errors_sorted[int(np.floor(n/4)):int(np.ceil(3*n/4)), :]
            n = errors.shape[0]
            mean = np.mean(errors, axis = 0)
            std = np.std(errors, axis = 0)
        else:
            mean = np.nanmean(errors, axis = 0)
            std = np.nanstd(errors, axis = 0)
    else:
        if plotting_stat == 'mse':
            mean = np.mean(np.square(errors))
            std = np.std(np.square(errors))
        elif plotting_stat == 'abs':
            n = len(errors)
            mean = np.mean(errors)
            std = np.std(errors)
        elif plotting_stat == 'iqm_abs':
            # IQM
            vals_sorted = np.sort(errors)
            errors = vals_sorted[int(np.floor(n/4)):int(np.ceil(3*n/4))]
            n = len(errors)
            mean = np.mean(errors)
            std = np.std(errors)
        
    yerr = 1.96 * std / np.sqrt(float(n))
    ylower = mean - yerr
    yupper = mean + yerr

    stats = {
        'mean': mean,
        'yerr': yerr,
        'ylower': ylower,
        'yupper': yupper
    }
    if print_log and errors.ndim == 1:
        print ('num trials for {}: {}, mean {}, ylower {}, yupper {}'.format(method, n, mean, ylower, yupper))
    return stats

def get_student_interval(data, z_score = 1.96):
    algorithms = sorted(list(data.keys()))

    means = {}
    ints = {}
    for algo in algorithms:
        algo_data = data[algo]
        algo_data = np.squeeze(algo_data, axis = 1)

        n = algo_data.shape[0]
        out = np.empty((2, algo_data.shape[1]))
        mean = np.nanmean(algo_data, axis = 0)
        means[algo] = mean

        std = np.nanstd(algo_data, axis = 0)
        yerr = z_score * std / np.sqrt(float(n))
        out[0] = mean - yerr
        out[1] = mean + yerr

        ints[algo] = out
    return means, ints 

def get_tolerance_interval(data, alpha = 0.05, beta = 0.9):
    algorithms = sorted(list(data.keys()))

    means = {}
    tol_ints = {}
    for algo in algorithms:
        algo_data = data[algo]
        algo_data = np.squeeze(algo_data, axis = 1)

        n = algo_data.shape[0]
        l, u = _get_tolerance_indices(n, alpha, beta)

        out = np.empty((2, algo_data.shape[1]))

        for i in range(algo_data.shape[1]):
            s = np.sort(algo_data[:, i])

            out[0, i] = s[l]
            out[1, i] = s[u]

        mean = np.nanmean(algo_data, axis = 0)
        means[algo] = mean
        tol_ints[algo] = out
    return means, tol_ints 

def _get_tolerance_indices(n: int, alpha: float, beta: float):
    # we cannot jit compile most things from scipy.stats
    # so perform a callback to the python interpreter to obtain this value
    y = _ppf(n, alpha, beta)

    nu = int(n - y)

    # figure out indices
    if nu % 2 == 0:
        l = int(nu / 2)
        u = int(n - (nu / 2)) - 1
    else:
        nu1 = (nu / 2) - (1 / 2)
        l = int(nu1)
        u = int(n - (nu1 + 1))

    return l, u

def _ppf(n: int, alpha: float, beta: float):
    return binom.ppf(1 - alpha, n, beta)

def run_err_deviation_lesser(scores, tau):
    return np.mean(scores < tau)

def run_err_deviation_greater(scores, tau):
    return np.mean(scores > tau)

def decorate_axis(ax, wrect=10, hrect=10, labelsize='large'):
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)
    # Pablos' comment
    ax.spines['left'].set_position(('outward', hrect))
    ax.spines['bottom'].set_position(('outward', wrect))

def plot_score_hist(score_dict, bins=20, figsize=(28, 14), 
                    fontsize='xx-large'):

    algorithms = sorted(list(score_dict.keys()))
    N = len(algorithms)
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)
    for i in range(N):
        score_matrix = score_dict[algorithms[i]]
        ax[i].set_title(algorithms[i], fontsize=fontsize)
        sns.histplot(score_matrix[:, 0], bins=bins, ax=ax[i], kde=False)

        decorate_axis(ax[i], wrect=5, hrect=5, labelsize='xx-large')
        ax[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        ax[i].set_ylabel('Count', size=fontsize)
        ax[i].grid(axis='y', alpha=0.1)
    return fig


