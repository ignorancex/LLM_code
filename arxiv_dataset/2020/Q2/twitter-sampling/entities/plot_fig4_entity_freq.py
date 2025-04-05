#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot entity frequency distribution for user posting and hashtag.

Usage: python plot_fig4_entity_freq.py
Input data files: ../data/[app_name]_out/[ts|user|hashtag]_*.txt
Time: ~6M
"""

import sys, os, platform
from collections import defaultdict, Counter
import numpy as np
from scipy.special import comb
from scipy import optimize, stats

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from powerlaw import Fit, plot_ccdf

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.vars import ColorPalette
from utils.plot_conf import hide_spines


def binomial(n, k, rho):
    # n: number of trials, k: number of success (being sampled)
    return comb(n, k, exact=True) * (rho ** k) * ((1 - rho) ** (n - k))


def mse_loss(x, A, b):
    # mean square error
    y = np.dot(A, x) - b
    return np.dot(y, y)


def infer_missing_num(sample_data, rho, m):
    n = len(sample_data)
    X = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if j >= i:
                X[i, j] = binomial(j+1, i+1, rho)

    y = np.array(sample_data[: m])

    bounds = [(0, None)] * n
    init_values = np.array(sample_data)
    constraints = tuple([{'type': 'ineq', 'fun': lambda x: x[i] - x[i + 1]} for i in range(n-1)])

    optimizer = optimize.minimize(mse_loss, init_values,
                                  method='SLSQP',
                                  args=(X, y),
                                  bounds=bounds,
                                  constraints=constraints,
                                  options={'maxiter': 100, 'disp': False})
    inferred_num_missing = sum([binomial(i + 1, 0, rho) * optimizer.x[i] for i in range(n)])
    return int(inferred_num_missing)


def main():
    timer = Timer()
    timer.start()

    app_name = 'cyberbullying'
    archive_dir = '../data/{0}_out'.format(app_name)
    entities = ['user', 'hashtag']
    rho = 0.5272

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))

    cc4 = ColorPalette.CC4
    blue = cc4[0]

    for ax_idx, entity in enumerate(entities):
        sample_datefile = open(os.path.join(archive_dir, '{0}_{1}_all.txt'.format(entity, app_name)), 'r', encoding='utf-8')
        complete_datefile = open(os.path.join(archive_dir, 'complete_{0}_{1}.txt'.format(entity, app_name)), 'r', encoding='utf-8')

        sample_entity_freq_dict = defaultdict(int)
        complete_entity_freq_dict = defaultdict(int)
        uni_random_entity_freq_dict = defaultdict(int)

        if entity == 'user':
            for line in sample_datefile:
                sample_entity_freq_dict[line.rstrip().split(',')[1]] += 1
            for line in complete_datefile:
                complete_entity_freq_dict[line.rstrip().split(',')[1]] += 1
                toss = np.random.random_sample()
                if toss <= rho:
                    uni_random_entity_freq_dict[line.rstrip().split(',')[1]] += 1
        else:
            for line in sample_datefile:
                for item in line.rstrip().split(',')[1:]:
                    sample_entity_freq_dict[item.lower()] += 1
            for line in complete_datefile:
                for item in line.rstrip().split(',')[1:]:
                    complete_entity_freq_dict[item.lower()] += 1
                toss = np.random.random_sample()
                if toss <= rho:
                    for item in line.rstrip().split(',')[1:]:
                        uni_random_entity_freq_dict[item.lower()] += 1

        sample_datefile.close()
        complete_datefile.close()

        # compute the powerlaw fit in the complete set
        complete_freq_list = list(complete_entity_freq_dict.values())
        complete_powerlaw_fit = Fit(complete_freq_list)
        complete_alpha = complete_powerlaw_fit.power_law.alpha
        complete_xmin = complete_powerlaw_fit.power_law.xmin
        print('{0} complete set alpha {1}, xmin {2}'.format(entity, complete_alpha, complete_xmin))
        plot_ccdf(complete_freq_list, ax=axes[ax_idx], color='k', ls='-', label='complete')

        # compute the powerlaw fit in the sample set
        # infer the number of missing entities
        sample_freq_list = list(sample_entity_freq_dict.values())
        sample_freq_counter = Counter(sample_freq_list)

        # we observe the frequency of entities appearing less than 100 times
        num_interest = 100
        sample_freq_list_top100 = [0] * num_interest
        for freq in range(1, num_interest + 1):
            sample_freq_list_top100[freq - 1] = sample_freq_counter[freq]

        inferred_num_missing = infer_missing_num(sample_freq_list_top100, rho=rho, m=num_interest)
        corrected_sample_freq_list = sample_freq_list + [0] * inferred_num_missing
        sample_powerlaw_fit = Fit(corrected_sample_freq_list)
        sample_alpha = sample_powerlaw_fit.power_law.alpha
        sample_xmin = sample_powerlaw_fit.power_law.xmin
        print('{0} sample set alpha {1}, xmin {2}'.format(entity, sample_alpha, sample_xmin))
        plot_ccdf(corrected_sample_freq_list, ax=axes[ax_idx], color=blue, ls='-', label='sample')

        # compute the powerlaw fit in uniform random sample
        uni_random_num_missing = len(complete_entity_freq_dict) - len(uni_random_entity_freq_dict)
        uni_random_freq_list = list(uni_random_entity_freq_dict.values())
        uni_random_freq_list = uni_random_freq_list + [0] * uni_random_num_missing
        uni_random_powerlaw_fit = Fit(uni_random_freq_list)
        uni_random_alpha = uni_random_powerlaw_fit.power_law.alpha
        uni_random_xmin = uni_random_powerlaw_fit.power_law.xmin
        print('{0} uniform random sampling alpha {1}, xmin {2}'.format(entity, uni_random_alpha, uni_random_xmin))
        plot_ccdf(uni_random_freq_list, ax=axes[ax_idx], color='k', ls='--', label='uniform random')

        print('inferred missing', inferred_num_missing)
        print('empirical missing', len(complete_entity_freq_dict) - len(sample_entity_freq_dict))
        print('uniform random missing', uni_random_num_missing)

        print('KS test (sample, uniform)')
        print(stats.ks_2samp(corrected_sample_freq_list, uni_random_freq_list))

        print('KS test (sample, complete)')
        print(stats.ks_2samp(corrected_sample_freq_list, complete_freq_list))

        print('KS test (uniform, complete)')
        print(stats.ks_2samp(uni_random_freq_list, complete_freq_list))

        axes[ax_idx].set_xscale('symlog')
        axes[ax_idx].set_yscale('log')
        axes[ax_idx].set_xlabel('frequency', fontsize=16)
        axes[ax_idx].tick_params(axis='both', which='major', labelsize=16)

    axes[0].set_xticks([0, 1, 100, 10000])
    axes[0].set_yticks([1, 0.01, 0.0001, 0.000001])
    axes[0].set_ylabel('$P(X \geq x)$', fontsize=16)
    axes[0].legend(frameon=False, fontsize=16, ncol=1, fancybox=False, shadow=True, loc='lower left')
    axes[0].set_title('(a) user posting', fontsize=18, pad=-3*72, y=1.0001)

    axes[1].set_xticks([0, 1, 100, 10000, 1000000])
    axes[1].set_yticks([1, 0.1, 0.001, 0.00001])
    axes[1].set_title('(b) hashtag', fontsize=18, pad=-3*72, y=1.0001)

    hide_spines(axes)

    timer.stop()

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('../images/entity_freq_dist.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
