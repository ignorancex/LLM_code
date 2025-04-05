#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot arriving of rate limit messages and #messages per second.

Usage: python plot_SI_fig2_rate_coloring.py
Input data files: ./rate_limit_2015-09-08.txt
Time: ~1M
"""

import sys, os, platform, json
from datetime import datetime

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.plot_conf import ColorPalette, hide_spines, concise_fmt


def map_ratemsg(track_lst, ts_lst):
    split_track_lst = [[track_lst[0]]]
    split_ts_lst = [[ts_lst[0]]]
    for value, ts in zip(track_lst[1:], ts_lst[1:]):
        last_entries = [l[-1] for l in split_track_lst]
        min_value = min(last_entries)
        if value <= min_value:
            split_track_lst.append([value])
            split_ts_lst.append([ts])
        else:
            best_idx = 0
            for idx, last_entry in enumerate(last_entries):
                if last_entry < value < last_entries[best_idx]:
                    best_idx = idx
            split_track_lst[best_idx].append(value)
            split_ts_lst[best_idx].append(ts)
    print('{0} threads have been mapped'.format(len(split_track_lst)))
    return split_track_lst, split_ts_lst


def main():
    timer = Timer()
    timer.start()

    cc4 = ColorPalette.CC4

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))

    start_idx = 21000
    end_idx = 25500
    timestamp_list = []
    track_list = []

    with open('rate_limit_2015-09-08.txt', 'r') as fin:
        for line in fin:
            rate_json = json.loads(line.rstrip())
            track = rate_json['limit']['track']
            track_list.append(track)
            timestamp = datetime.utcfromtimestamp((int(rate_json['limit']['timestamp_ms'][:-3])))
            timestamp_list.append(timestamp)

    axes[0].scatter(timestamp_list[start_idx: end_idx], track_list[start_idx: end_idx], c='k', s=0.4)
    axes[0].set_xlim([timestamp_list[start_idx], timestamp_list[end_idx]])
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axes[0].set_xticks(axes[0].get_xticks()[::2])
    axes[0].set_xlabel('Sep 08, 2015', fontsize=16)
    axes[0].set_ylabel('value', fontsize=16)
    axes[0].yaxis.set_major_formatter(FuncFormatter(concise_fmt))
    axes[0].tick_params(axis='both', which='major', labelsize=16)
    axes[0].set_title('(a)', size=18, pad=-3*72, y=1.0001)

    print('start timestamp', timestamp_list[start_idx])
    print('end timestamp', timestamp_list[end_idx])
    split_track_lst, split_ts_lst = map_ratemsg(track_list[start_idx: end_idx], timestamp_list[start_idx: end_idx])
    total_miss = 0
    for track_lst, ts_lst, color in zip(split_track_lst, split_ts_lst, cc4):
        axes[1].scatter(ts_lst, track_lst, c=color, s=0.4)
        total_miss += (track_list[-1] - track_list[0])
    print('{0} tweets are missing'.format(total_miss))
    axes[1].set_xlim([timestamp_list[start_idx], timestamp_list[end_idx]])
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axes[1].set_xticks(axes[1].get_xticks()[::2])
    axes[1].set_xlabel('Sep 08, 2015', fontsize=16)
    axes[1].yaxis.set_major_formatter(FuncFormatter(concise_fmt))
    axes[1].tick_params(axis='both', which='major', labelsize=16)
    axes[1].set_title('(b)', size=18, pad=-3*72, y=1.0001)

    hide_spines(axes)

    timer.stop()

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('../images/SI_ratemsg_coloring.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
