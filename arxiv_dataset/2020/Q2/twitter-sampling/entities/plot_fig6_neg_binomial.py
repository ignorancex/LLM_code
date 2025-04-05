#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot probabilities of entity occurrence.

Usage: python plot_fig6_neg_binomial.py
Input data files: ../data/[app_name]_out/complete_user_[app_name].txt, ../data/[app_name]_out/user_[app_name]_all.txt
Time: ~5M
"""

import sys, os, platform
from collections import defaultdict, Counter
import numpy as np
from scipy.special import comb
from scipy import stats

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.plot_conf import ColorPalette, hide_spines


def negative_binomial(n, k, rho):
    # n: number of trials, k: number of success (being sampled)
    return comb(n-1, k-1, exact=True) * (rho ** k) * ((1 - rho) ** (n - k))


def main():
    timer = Timer()
    timer.start()

    cc4 = ColorPalette.CC4
    blue = cc4[0]

    app_name = 'cyberbullying'
    rho = 0.5272
    entity = 'user'

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))

    print('for entity: {0}'.format(entity))
    sample_entity_freq_dict = defaultdict(int)
    with open('../data/{1}_out/{0}_{1}_all.txt'.format(entity, app_name), 'r') as sample_datefile:
        for line in sample_datefile:
            sample_entity_freq_dict[line.rstrip().split(',')[1]] += 1

    complete_entity_freq_dict = defaultdict(int)
    with open('../data/{1}_out/complete_{0}_{1}.txt'.format(entity, app_name), 'r') as complete_datefile:
        for line in complete_datefile:
            complete_entity_freq_dict[line.rstrip().split(',')[1]] += 1

    complete_to_sample_freq_dict = defaultdict(list)
    sample_to_complete_freq_dict = defaultdict(list)

    for item, complete_vol in complete_entity_freq_dict.items():
        if item in sample_entity_freq_dict:
            complete_to_sample_freq_dict[complete_vol].append(sample_entity_freq_dict[item])
        else:
            complete_to_sample_freq_dict[complete_vol].append(0)

    for item, sample_vol in sample_entity_freq_dict.items():
        sample_to_complete_freq_dict[sample_vol].append(complete_entity_freq_dict[item])

    for item in set(complete_entity_freq_dict.keys()) - set(sample_entity_freq_dict.keys()):
        sample_to_complete_freq_dict[0].append(complete_entity_freq_dict[item])

    ax1_x_axis = range(1, 101)

    ax1_y_axis = []
    empirical_mean_list = []
    expected_mean_list = []
    for num_sample in ax1_x_axis:
        # compute sample to complete
        empirical_cnt_dist = sample_to_complete_freq_dict[num_sample]
        neg_binomial_cnt_dist = []
        for x in range(num_sample, max(30, 3 * num_sample + 1)):
            neg_binomial_cnt_dist.extend([x] * int(negative_binomial(x, num_sample, rho) * len(empirical_cnt_dist)))
        ks_test = stats.ks_2samp(empirical_cnt_dist, neg_binomial_cnt_dist)
        empirical_mean = sum(empirical_cnt_dist) / len(empirical_cnt_dist)
        empirical_mean_list.append(empirical_mean)
        expected_mean = sum(neg_binomial_cnt_dist) / len(neg_binomial_cnt_dist)
        expected_mean_list.append(expected_mean)
        print('num_sample: {0}, number of Bernoulli trials: {1}, d_statistic: {2:.4f}, p: {3:.4f}, expected mean: {4:.2f}, empirical mean: {5:.2f}'
              .format(num_sample, len(empirical_cnt_dist), ks_test[0], ks_test[1], expected_mean, empirical_mean))
        ax1_y_axis.append(ks_test[0])

    axes[0].plot(ax1_x_axis, ax1_y_axis, c='k', lw=1.5, ls='-')

    axes[0].set_xlabel(r'sample frequency $n_s$', fontsize=16)
    axes[0].set_ylabel('D-statistic', fontsize=16)
    axes[0].set_xlim([-2, 102])
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[0].set_ylim([0, 0.17])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{0:.2f}'.format(x)))
    axes[0].tick_params(axis='both', which='major', labelsize=16)
    axes[0].set_title('(a)', fontsize=18, pad=-3*72, y=1.0001)

    # show an example
    num_sample = np.argmin(ax1_y_axis) + 1

    axes[0].scatter(num_sample, ax1_y_axis[num_sample - 1], s=40, c=blue, zorder=30)
    axes[0].set_yticks([0, ax1_y_axis[num_sample - 1], 0.05, 0.1, 0.15])
    axes[0].plot([axes[0].get_xlim()[0], num_sample], [ax1_y_axis[num_sample - 1], ax1_y_axis[num_sample - 1]], color=blue, ls='--', lw=1)
    axes[0].plot([num_sample, num_sample], [axes[0].get_ylim()[0], ax1_y_axis[num_sample - 1]], color=blue, ls='--', lw=1)

    # plot sample to complete
    ax2_x_axis = range(num_sample, max(30, 3 * num_sample + 1))
    num_items = len(sample_to_complete_freq_dict[num_sample])
    sample_to_complete_cnt = Counter(sample_to_complete_freq_dict[num_sample])
    ax2_y_axis = [sample_to_complete_cnt[x] / num_items for x in ax2_x_axis]
    ax2_neg_binomial_axis = [negative_binomial(x, num_sample, rho) for x in ax2_x_axis]

    axes[1].plot(ax2_x_axis, ax2_y_axis, c=blue, lw=1.5, ls='-', marker='o', zorder=20, label='empirical')
    axes[1].plot(ax2_x_axis, ax2_neg_binomial_axis, c='k', lw=1.5, ls='-', marker='x', zorder=10, label='negative binomial')

    axes[1].set_xlabel(r'complete frequency $n_c$', fontsize=16)
    axes[1].set_ylabel(r'Pr($n_c$|$n_s$={0})'.format(num_sample), fontsize=16)
    axes[1].set_xticks([num_sample, 2 * num_sample, 3 * num_sample])
    axes[1].set_ylim([-0.005, 0.15])
    axes[1].set_yticks([0, 0.05, 0.1])
    axes[1].tick_params(axis='both', which='major', labelsize=16)
    axes[1].legend(frameon=False, fontsize=16, ncol=1, fancybox=False, shadow=True, loc='upper left')
    axes[1].set_title('(b)', fontsize=18, pad=-3*72, y=1.0001)

    axes[1].plot([empirical_mean_list[num_sample - 1], empirical_mean_list[num_sample - 1]], [axes[1].get_ylim()[0], 0.1], color=blue, ls='--', lw=1)
    axes[1].plot([expected_mean_list[num_sample - 1], expected_mean_list[num_sample - 1]], [axes[1].get_ylim()[0], 0.1], color='k', ls='--', lw=1)

    hide_spines(axes)

    timer.stop()

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('../images/entity_negative_binomial.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
