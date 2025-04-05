#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot arriving of rate limit messages and #messages per second.

Usage: python plot_SI_fig1_rate_dist.py
Input data files: ./rate_limit_2015-09-08.txt
Time: ~1M
"""

import sys, os, platform, json
from datetime import datetime
from collections import defaultdict, Counter

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.plot_conf import ColorPalette, hide_spines, concise_fmt


def main():
    timer = Timer()
    timer.start()

    cc4 = ColorPalette.CC4
    blue = cc4[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))

    timestamp_list = []
    sec_count_dict = defaultdict(int)
    ms_list = []

    with open('rate_limit_2015-09-08.txt', 'r') as fin:
        for line in fin:
            rate_json = json.loads(line.rstrip())
            ms_list.append(int(rate_json['limit']['timestamp_ms'][-3:]))
            timestamp = datetime.utcfromtimestamp((int(rate_json['limit']['timestamp_ms']) - 666) // 1000)
            timestamp_list.append(timestamp)
            sec_count_dict[timestamp] += 1

    print('{0:.2f}% rate limit messages come from millisecond 700 to 1000'.format(len([x for x in ms_list if x >= 700]) / len(ms_list) * 100))

    sns.distplot(ms_list, bins=200, color=blue, ax=axes[0], kde_kws={'shade': False, 'linewidth': 1.5, 'color': 'k'})
    axes[0].set_xticks([0, 250, 500, 750, 1000])
    axes[0].set_xlim([-50, 1050])
    axes[0].set_xlabel('millisecond', fontsize=16)
    axes[0].set_ylabel('density', fontsize=16)
    axes[0].tick_params(axis='both', which='major', labelsize=16)
    axes[0].set_title('(a)', size=18, pad=-3*72, y=1.0001)

    sec_count_stats = Counter(sec_count_dict.values())
    x_axis = sorted(sec_count_stats.keys())
    axes[1].bar(x_axis, [sec_count_stats[x] for x in x_axis], facecolor=blue, edgecolor='k', width=0.7)
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_xlim([0, 5])
    axes[1].set_xlabel('#rate limit messages per second', fontsize=16)
    axes[1].set_ylabel('frequency', fontsize=16)
    axes[1].yaxis.set_major_formatter(FuncFormatter(concise_fmt))
    axes[1].tick_params(axis='both', which='major', labelsize=16)
    axes[1].set_title('(b)', size=18, pad=-3*72, y=1.0001)

    hide_spines(axes)

    timer.stop()

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('../images/SI_ratemsg_dist.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
