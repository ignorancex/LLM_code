#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot the hourly/millisecondly sampling rates.

Usage: python plot_fig2_temporal_sampling_rates.py
Input data files: ./[app_name]_out/complete_ts_[app_name].txt, ./[app_name]_out/ts_[app_name]_all.txt
Time: ~12M
"""

import sys, os, platform
from datetime import datetime, timezone
import numpy as np

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from utils.metrics import mean_confidence_interval
from utils.plot_conf import ColorPalette, hide_spines


def main():
    timer = Timer()
    timer.start()

    cc4 = ColorPalette.CC4
    blue = cc4[0]
    red = cc4[3]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))

    num_days = 14
    hours_in_day = 24
    hour_x_axis = range(hours_in_day)
    minutes_in_hour = 60
    seconds_in_minute = 60
    ms_in_second = 1000
    ms_bins = 100
    width = ms_in_second // ms_bins
    ms_x_axis = range(ms_in_second)

    app_conf = {'cyberbullying': {'min_date': '2019-10-13',
                                  'label': 'Cyberbullying',
                                  'color': blue},
                'youtube': {'min_date': '2019-11-06',
                            'label': 'YouTube',
                            'color': red}
                }

    for app_name in app_conf.keys():
        archive_dir = './{0}_out'.format(app_name)

        min_date = datetime.strptime(app_conf[app_name]['min_date'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        min_timestamp = int(min_date.timestamp())
        min_day = min_date.day
        sample_datefile = open(os.path.join(archive_dir, 'ts_{0}_all.txt'.format(app_name)), 'r')
        complete_datefile = open(os.path.join(archive_dir, 'complete_ts_{0}.txt'.format(app_name)), 'r')

        sample_tid_set = set()

        hour_hit_mat = np.zeros(shape=(hours_in_day, num_days))
        hour_miss_mat = np.zeros(shape=(hours_in_day, num_days))

        ms_hit_mat = np.zeros(shape=(ms_in_second, num_days * hours_in_day))
        ms_miss_mat = np.zeros(shape=(ms_in_second, num_days * hours_in_day))

        confusion_hit_mat = np.zeros(shape=(hours_in_day, minutes_in_hour, seconds_in_minute, ms_bins))
        confusion_miss_mat = np.zeros(shape=(hours_in_day, minutes_in_hour, seconds_in_minute, ms_bins))

        for line in sample_datefile:
            split_line = line.rstrip().split(',')
            if len(split_line) == 2:
                sample_tid_set.add(split_line[1])

        for line in complete_datefile:
            split_line = line.rstrip().split(',')
            if len(split_line) == 2:
                timestamp_ms = int(split_line[0][:-3])
                if timestamp_ms >= min_timestamp:
                    dt_obj = datetime.utcfromtimestamp(timestamp_ms)
                    day_idx = dt_obj.day - min_day
                    hour = dt_obj.hour
                    minute = dt_obj.minute
                    second = dt_obj.second
                    millisec = int(split_line[0][-3:])
                    ms_idx = (millisec - 7) // width if millisec >= 7 else (ms_in_second + millisec - 7) // width

                    if split_line[1] in sample_tid_set:
                        hour_hit_mat[hour][day_idx] += 1
                        ms_hit_mat[millisec][hours_in_day * day_idx + hour] += 1
                        confusion_hit_mat[hour][minute][second][ms_idx] += 1
                    else:
                        hour_miss_mat[hour][day_idx] += 1
                        ms_miss_mat[millisec][hours_in_day * day_idx + hour] += 1
                        confusion_miss_mat[hour][minute][second][ms_idx] += 1

        # hourly tweet sampling rate
        rho_mean_list_hour = []
        ub_rho_mean_list_hour = []
        lb_rho_mean_list_hour = []

        for i in hour_x_axis:
            mean, lb, ub = mean_confidence_interval(hour_hit_mat[i, :] / (hour_hit_mat[i, :] + hour_miss_mat[i, :]), confidence=0.95)
            rho_mean_list_hour.append(mean)
            lb_rho_mean_list_hour.append(lb)
            ub_rho_mean_list_hour.append(ub)

        # confusion sampling rate
        confusion_sampling_rate = confusion_hit_mat / (confusion_hit_mat + confusion_miss_mat)
        confusion_sampling_rate = np.nan_to_num(confusion_sampling_rate)
        np.save(os.path.join(archive_dir, '{0}_confusion_sampling_rate.npy'.format(app_name)), confusion_sampling_rate)

        axes[0].plot(hour_x_axis, rho_mean_list_hour, c='k', lw=1.5, ls='-', zorder=20)
        axes[0].fill_between(hour_x_axis, ub_rho_mean_list_hour, rho_mean_list_hour,
                             facecolor=app_conf[app_name]['color'], alpha=0.8, lw=0, zorder=10)
        axes[0].fill_between(hour_x_axis, lb_rho_mean_list_hour, rho_mean_list_hour,
                             facecolor=app_conf[app_name]['color'], alpha=0.8, lw=0, zorder=10,
                             label='{0}'.format(app_conf[app_name]['label']))

        # msly tweet sampling rate
        rho_mean_list_ms = []
        ub_rho_mean_list_ms = []
        lb_rho_mean_list_ms = []

        for i in ms_x_axis:
            mean, lb, ub = mean_confidence_interval(ms_hit_mat[i, :] / (ms_hit_mat[i, :] + ms_miss_mat[i, :]), confidence=0.95)
            rho_mean_list_ms.append(mean)
            lb_rho_mean_list_ms.append(lb)
            ub_rho_mean_list_ms.append(ub)

        axes[1].plot(ms_x_axis, rho_mean_list_ms, c='k', lw=1.5, ls='-', zorder=20)
        axes[1].fill_between(ms_x_axis, ub_rho_mean_list_ms, rho_mean_list_ms,
                             facecolor=app_conf[app_name]['color'], alpha=0.8, lw=0, zorder=10)
        axes[1].fill_between(ms_x_axis, lb_rho_mean_list_ms, rho_mean_list_ms,
                             facecolor=app_conf[app_name]['color'], alpha=0.8, lw=0, zorder=10)

    axes[0].set_xticks([0, 6, 12, 18, 24])
    axes[0].set_xlabel('hour (in UTC)', fontsize=16)
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel(r'sampling rate $\rho_t$', fontsize=16)
    axes[0].tick_params(axis='both', which='major', labelsize=16)
    axes[0].legend(frameon=False, fontsize=16, ncol=1, fancybox=False, shadow=True)
    axes[0].set_title('(a)', size=18, pad=-3*72, y=1.0001)

    axes[1].axvline(x=657, ymin=0, ymax=0.4, c='k', ls='--')
    axes[1].text(667, 0.2, 'x=657', size=18, ha='left', va='center')
    axes[1].set_xticks([0, 250, 500, 750, 1000])
    axes[1].set_xlabel('millisecond', fontsize=16)
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[1].tick_params(axis='both', which='major', labelsize=16)
    axes[1].set_title('(b)', size=18, pad=-3*72, y=1.0001)

    hide_spines(axes)

    timer.stop()

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('../images/temporal_sampling_rates.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
