#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot the number of tweets on different languages.

Usage: python plot_fig3_lang_volume.py
Input data files: ./youtube_out/ts_youtube_*.txt
Time: ~5M
"""

import sys, os, platform
from datetime import datetime
import numpy as np

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.metrics import mean_confidence_interval
from utils.plot_conf import ColorPalette, hide_spines, concise_fmt


def main():
    timer = Timer()
    timer.start()

    app_name = 'youtube'
    archive_dir = './{0}_out'.format(app_name)
    lang_list = ['ja+ko', 'others']

    cc4 = ColorPalette.CC4
    red = cc4[3]

    num_days = 14
    hours_in_day = 24

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))

    sample_tid_set = set()
    sample_ts_datefile = os.path.join(archive_dir, 'ts_{0}_all.txt'.format(app_name))
    with open(sample_ts_datefile, 'r') as fin:
        for line in fin:
            split_line = line.rstrip().split(',')
            if len(split_line) == 2:
                ts, tid = split_line
                sample_tid_set.add(tid)

    for idx, lang in enumerate(lang_list):
        if idx == 0:
            subcrawler_ts_datefiles = [os.path.join(archive_dir, 'ts_{0}_{1}.txt'.format(app_name, j)) for j in [2, 3, 8, 9]]
        else:
            subcrawler_ts_datefiles = [os.path.join(archive_dir, 'ts_{0}_{1}.txt'.format(app_name, j)) for j in [1, 4, 5, 6, 7, 10, 11, 12]]

        num_in = 0
        num_out = 0

        count_sample = np.zeros(shape=(hours_in_day, num_days))
        count_complete = np.zeros(shape=(hours_in_day, num_days))

        visited_tid_set = set()

        for ts_datefile in subcrawler_ts_datefiles:
            with open(ts_datefile, 'r') as fin:
                for line in fin:
                    split_line = line.rstrip().split(',')
                    if len(split_line) == 2:
                        ts, tid = split_line
                        if tid not in visited_tid_set:
                            dt_obj = datetime.utcfromtimestamp(int(ts[:-3]))
                            day_idx = dt_obj.day - 6
                            hour = dt_obj.hour
                            count_complete[hour][day_idx] += 1
                            if tid in sample_tid_set:
                                num_in += 1
                                count_sample[hour][day_idx] += 1
                            else:
                                num_out += 1
                            visited_tid_set.add(tid)

        print('collected tweets: {0}, missing tweets: {1}, sample ratio for lang {2}: {3:.2f}%'
              .format(num_in, num_out, lang, num_in / (num_in + num_out) * 100))

        # hourly tweet volume in youtube for some languages
        sample_volume_mean_list_hour = []
        sample_ub_volume_mean_list_hour = []
        sample_lb_volume_mean_list_hour = []

        complete_volume_mean_list_hour = []
        complete_ub_volume_mean_list_hour = []
        complete_lb_volume_mean_list_hour = []

        for j in range(hours_in_day):
            mean, lb, ub = mean_confidence_interval(count_sample[j, :], confidence=0.95)
            sample_volume_mean_list_hour.append(mean)
            sample_lb_volume_mean_list_hour.append(lb)
            sample_ub_volume_mean_list_hour.append(ub)

            mean, lb, ub = mean_confidence_interval(count_complete[j, :], confidence=0.95)
            complete_volume_mean_list_hour.append(mean)
            complete_lb_volume_mean_list_hour.append(lb)
            complete_ub_volume_mean_list_hour.append(ub)

        print('tweet volumes from JST-6pm to 12am: {0:.2f}%'.format(100*sum(complete_volume_mean_list_hour[9:15]) / sum(complete_volume_mean_list_hour)))
        print('sampling rates from JST-6pm to 12am: {0:.2f}%'.format(100*sum(sample_volume_mean_list_hour[9:15]) / sum(complete_volume_mean_list_hour[9:15])))

        hour_x_axis = range(hours_in_day)
        axes[idx].plot(hour_x_axis, complete_volume_mean_list_hour, c='k', lw=1.5, ls='-', zorder=20, label='complete')
        axes[idx].fill_between(hour_x_axis, complete_ub_volume_mean_list_hour, complete_volume_mean_list_hour,
                               facecolor='lightgray', lw=0, zorder=10)
        axes[idx].fill_between(hour_x_axis, complete_lb_volume_mean_list_hour, complete_volume_mean_list_hour,
                               facecolor='lightgray', lw=0, zorder=10)

        axes[idx].plot(hour_x_axis, sample_volume_mean_list_hour, c='k', lw=1.5, ls='--', zorder=20, label='sample')
        axes[idx].fill_between(hour_x_axis, sample_ub_volume_mean_list_hour, sample_volume_mean_list_hour,
                               facecolor=red, alpha=0.8, lw=0, zorder=10)
        axes[idx].fill_between(hour_x_axis, sample_lb_volume_mean_list_hour, sample_volume_mean_list_hour,
                               facecolor=red, alpha=0.8, lw=0, zorder=10)

        if idx == 0:
            axes[idx].plot([9, 9], [complete_ub_volume_mean_list_hour[9], 150000], 'k--', lw=1)
            axes[idx].plot([15, 15], [complete_ub_volume_mean_list_hour[15], 150000], 'k--', lw=1)
            axes[idx].text(8, 150000, 'JST-6pm', ha='center', va='bottom', size=16)
            axes[idx].text(16, 150000, '12am', ha='center', va='bottom', size=16)
            axes[idx].annotate('', xy=(9, 135000), xycoords='data', xytext=(15, 135000), textcoords='data',
                               arrowprops=dict(arrowstyle='<->', connectionstyle='arc3'), zorder=50)

        axes[idx].set_xlabel('hour (in UTC)', fontsize=16)
        axes[idx].set_xticks([0, 6, 12, 18, 24])
        axes[idx].set_ylim([0, 180000])
        axes[idx].set_yticks([0, 50000, 100000, 150000])
        axes[idx].yaxis.set_major_formatter(FuncFormatter(concise_fmt))
        axes[idx].tick_params(axis='both', which='major', labelsize=16)
        axes[idx].set_title('({0}) {1}'.format(['a', 'b'][idx], lang_list[idx]), size=18, pad=-3*72, y=1.0001)

    axes[0].set_ylabel('#tweets', fontsize=16)
    axes[1].legend(frameon=False, fontsize=16, ncol=1, fancybox=False, shadow=True, loc='lower right')

    hide_spines(axes)

    timer.stop()

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('../images/tweet_lang_vol.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
