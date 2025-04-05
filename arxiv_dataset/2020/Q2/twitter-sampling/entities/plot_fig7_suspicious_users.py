#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plot ranks of top 100 users in the cyberbullying dataset

Usage: python plot_fig4_top_entities.py
Input data files: ../data/[app_name]_out/complete_user_[app_name].txt, ../data/[app_name]_out/user_[app_name]_all.txt
Time: ~8M
"""

import sys, os, platform
from collections import defaultdict
from datetime import datetime
import numpy as np
from scipy.stats import entropy

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, melt_snowflake
from utils.plot_conf import ColorPalette, hide_spines

cm = plt.cm.get_cmap('RdBu')


def write_to_file(filepath, header, datalist):
    with open(filepath, 'w') as fout:
        for user_idx, user_id in enumerate(header):
            fout.write('{0}\t{1}\t{2}\n'.format(user_id, sum(datalist[user_idx]), ','.join(map(str, datalist[user_idx]))))


def read_from_file(filepath, dtype=0):
    datalist = []
    with open(filepath, 'r') as fin:
        for line in fin:
            user_id, total, records = line.rstrip().split('\t')
            if dtype == 0:
                records = list(map(int, records.split(',')))
            else:
                records = list(map(float, records.split(',')))
            datalist.append(records)
    return datalist


def main():
    timer = Timer()
    timer.start()

    app_name = 'cyberbullying'

    hours_in_day = 24
    minutes_in_hour = 60
    seconds_in_minute = 60
    ms_in_second = 1000

    num_bins = 100
    width = ms_in_second // num_bins

    num_top = 500

    confusion_sampling_rate = np.load('../data/{0}_out/{0}_confusion_sampling_rate.npy'.format(app_name))
    confusion_sampling_rate = np.nan_to_num(confusion_sampling_rate)

    load_external_data = False
    if not load_external_data:
        sample_entity_stats = defaultdict(int)
        with open('../data/{0}_out/user_{0}_all.txt'.format(app_name), 'r') as fin:
            for line in fin:
                split_line = line.rstrip().split(',')
                sample_entity_stats[split_line[1]] += 1

        # == == == == == == Part 2: Plot entity rank == == == == == == #
        print('>>> found top {0} users in sample set...'.format(num_top))
        sample_top = [kv[0] for kv in sorted(sample_entity_stats.items(), key=lambda x: x[1], reverse=True)[:num_top]]

        # == == == == == == Part 1: Find tweets appearing in complete set == == == == == == #
        complete_post_lists_hour = [[0] * hours_in_day for _ in range(num_top)]
        complete_post_lists_min = [[0] * minutes_in_hour for _ in range(num_top)]
        complete_post_lists_sec = [[0] * seconds_in_minute for _ in range(num_top)]
        complete_post_lists_10ms = [[0] * num_bins for _ in range(num_top)]

        complete_entity_stats = defaultdict(int)
        with open('../data/{0}_out/complete_user_{0}.txt'.format(app_name), 'r') as fin:
            for line in fin:
                split_line = line.rstrip().split(',')
                user_id = split_line[1]
                if user_id in sample_top:
                    complete_entity_stats[user_id] += 1

                    user_idx = sample_top.index(user_id)
                    tweet_id = split_line[0]
                    timestamp_ms = melt_snowflake(tweet_id)[0]
                    dt_obj = datetime.utcfromtimestamp(timestamp_ms // 1000)
                    hour = dt_obj.hour
                    minute = dt_obj.minute
                    second = dt_obj.second
                    millisec = timestamp_ms % 1000
                    ms_idx = (millisec-7) // width if millisec >= 7 else (1000 + millisec-7) // width

                    complete_post_lists_hour[user_idx][hour] += 1
                    complete_post_lists_min[user_idx][minute] += 1
                    complete_post_lists_sec[user_idx][second] += 1
                    complete_post_lists_10ms[user_idx][ms_idx] += 1

        write_to_file('./complete_post_lists_hour.txt', sample_top, complete_post_lists_hour)
        write_to_file('./complete_post_lists_min.txt', sample_top, complete_post_lists_min)
        write_to_file('./complete_post_lists_sec.txt', sample_top, complete_post_lists_sec)
        write_to_file('./complete_post_lists_10ms.txt', sample_top, complete_post_lists_10ms)

        print('>>> finish dumping complete lists...')
        timer.stop()

        # == == == == == == Part 2: Find appearing tweets in sample set == == == == == == #
        sample_post_lists_hour = [[0] * hours_in_day for _ in range(num_top)]
        sample_post_lists_min = [[0] * minutes_in_hour for _ in range(num_top)]
        sample_post_lists_sec = [[0] * seconds_in_minute for _ in range(num_top)]
        sample_post_lists_10ms = [[0] * num_bins for _ in range(num_top)]

        estimated_post_lists_hour = [[0] * hours_in_day for _ in range(num_top)]
        estimated_post_lists_min = [[0] * minutes_in_hour for _ in range(num_top)]
        estimated_post_lists_sec = [[0] * seconds_in_minute for _ in range(num_top)]
        estimated_post_lists_10ms = [[0] * num_bins for _ in range(num_top)]

        hourly_conversion = np.mean(confusion_sampling_rate, axis=(1, 2, 3))
        minutey_conversion = np.mean(confusion_sampling_rate, axis=(2, 3))
        secondly_conversion = np.mean(confusion_sampling_rate, axis=(3))

        with open('../data/{0}_out/user_{0}_all.txt'.format(app_name), 'r') as fin:
            for line in fin:
                split_line = line.rstrip().split(',')
                user_id = split_line[1]
                if user_id in sample_top:
                    user_idx = sample_top.index(user_id)
                    tweet_id = split_line[0]
                    timestamp_ms = melt_snowflake(tweet_id)[0]
                    dt_obj = datetime.utcfromtimestamp(timestamp_ms // 1000)
                    hour = dt_obj.hour
                    minute = dt_obj.minute
                    second = dt_obj.second
                    millisec = timestamp_ms % 1000
                    ms_idx = (millisec-7) // width if millisec >= 7 else (1000 + millisec-7) // width

                    sample_post_lists_hour[user_idx][hour] += 1
                    sample_post_lists_min[user_idx][minute] += 1
                    sample_post_lists_sec[user_idx][second] += 1
                    sample_post_lists_10ms[user_idx][ms_idx] += 1

                    estimated_post_lists_hour[user_idx][hour] += 1 / hourly_conversion[hour]
                    estimated_post_lists_min[user_idx][minute] += 1 / minutey_conversion[hour, minute]
                    estimated_post_lists_sec[user_idx][second] += 1 / secondly_conversion[hour, minute, second]
                    estimated_post_lists_10ms[user_idx][ms_idx] += 1 / confusion_sampling_rate[hour, minute, second, ms_idx]

        write_to_file('./sample_post_lists_hour.txt', sample_top, sample_post_lists_hour)
        write_to_file('./sample_post_lists_min.txt', sample_top, sample_post_lists_min)
        write_to_file('./sample_post_lists_sec.txt', sample_top, sample_post_lists_sec)
        write_to_file('./sample_post_lists_10ms.txt', sample_top, sample_post_lists_10ms)

        write_to_file('./estimated_post_lists_hour.txt', sample_top, estimated_post_lists_hour)
        write_to_file('./estimated_post_lists_min.txt', sample_top, estimated_post_lists_min)
        write_to_file('./estimated_post_lists_sec.txt', sample_top, estimated_post_lists_sec)
        write_to_file('./estimated_post_lists_10ms.txt', sample_top, estimated_post_lists_10ms)

        print('>>> finish dumping sample and estimated lists...')
        timer.stop()
    else:
        sample_top = []
        complete_post_lists_hour = []
        with open('./complete_post_lists_hour.txt', 'r') as fin:
            for line in fin:
                user_id, total, records = line.rstrip().split('\t')
                sample_top.append(user_id)
                records = list(map(int, records.split(',')))
                complete_post_lists_hour.append(records)

        complete_post_lists_min = read_from_file('./complete_post_lists_min.txt', dtype=0)
        complete_post_lists_sec = read_from_file('./complete_post_lists_sec.txt', dtype=0)
        complete_post_lists_10ms = read_from_file('./complete_post_lists_10ms.txt', dtype=0)

        sample_post_lists_hour = read_from_file('./sample_post_lists_hour.txt', dtype=0)
        sample_post_lists_min = read_from_file('./sample_post_lists_min.txt', dtype=0)
        sample_post_lists_sec = read_from_file('./sample_post_lists_sec.txt', dtype=0)
        sample_post_lists_10ms = read_from_file('./sample_post_lists_10ms.txt', dtype=0)

        estimated_post_lists_hour = read_from_file('./estimated_post_lists_hour.txt', dtype=1)
        estimated_post_lists_min = read_from_file('./estimated_post_lists_min.txt', dtype=1)
        estimated_post_lists_sec = read_from_file('./estimated_post_lists_sec.txt', dtype=1)
        estimated_post_lists_10ms = read_from_file('./estimated_post_lists_10ms.txt', dtype=1)

    # == == == == == == Part 3: Find the best estimation by comparing JS distance == == == == == == #
    ret = {}
    num_estimate_list = []
    num_sample_list = []
    num_complete_list = []

    sample_entity_stats = {user_id: sum(sample_post_lists_hour[user_idx]) for user_idx, user_id in enumerate(sample_top)}
    complete_entity_stats = {user_id: sum(complete_post_lists_hour[user_idx]) for user_idx, user_id in enumerate(sample_top)}

    min_mat = np.array([], dtype=np.int64).reshape(0, 60)
    sec_mat = np.array([], dtype=np.int64).reshape(0, 60)

    for user_idx, user_id in enumerate(sample_top):
        num_sample = sample_entity_stats[user_id]
        num_complete = complete_entity_stats[user_id]

        hour_entropy = entropy(sample_post_lists_hour[user_idx], base=hours_in_day)
        min_entropy = entropy(sample_post_lists_min[user_idx], base=minutes_in_hour)
        sec_entropy = entropy(sample_post_lists_sec[user_idx], base=seconds_in_minute)
        ms10_entropy = entropy(sample_post_lists_10ms[user_idx], base=num_bins)

        min_mat = np.vstack((min_mat, np.array(sample_post_lists_min[user_idx]).reshape(1, -1)))
        sec_mat = np.vstack((sec_mat, np.array(sample_post_lists_sec[user_idx]).reshape(1, -1)))

        if ms10_entropy < 0.87:
            min_entropy_idx = 3
        else:
            min_entropy_idx = 0

        num_estimate = sum([estimated_post_lists_hour[user_idx], estimated_post_lists_min[user_idx],
                            estimated_post_lists_sec[user_idx], estimated_post_lists_10ms[user_idx]][min_entropy_idx])
        num_estimate_list.append(num_estimate)

        num_sample_list.append(num_sample)
        num_complete_list.append(num_complete)

        ret[user_id] = (num_sample, num_complete, num_estimate, min_entropy_idx)

    # == == == == == == Part 3: Plot case users == == == == == == #
    case_user_ids = ['1033778124968865793', '1182605743335211009']
    case_user_screennames = ['WeltRadio', 'bensonbersk']

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))

    cc4 = ColorPalette.CC4
    blue = cc4[0]
    red = cc4[3]
    filled_colors = [blue, red]
    labels = ['(c)', '(d)']
    for ax_idx, user_id in enumerate(case_user_ids):
        user_idx = sample_top.index(user_id)
        min_entropy_idx = ret[user_id][-1]

        if min_entropy_idx == 0:
            axes[ax_idx].bar(range(hours_in_day), complete_post_lists_hour[user_idx], color='lightgray', width=1)
            axes[ax_idx].bar(range(hours_in_day), sample_post_lists_hour[user_idx], color=filled_colors[ax_idx], alpha=0.8, width=1)
            axes[ax_idx].plot(range(hours_in_day), estimated_post_lists_hour[user_idx], 'k-', lw=1.5)
            axes[ax_idx].set_xlabel('hour', fontsize=12)
            axes[ax_idx].set_xlim([-1, hours_in_day+1])
            axes[ax_idx].set_xticks([0, 6, 12, 18, 24])
            axes[ax_idx].set_title('{0} {1}'.format(labels[ax_idx], case_user_screennames[ax_idx]), fontsize=13)
        elif min_entropy_idx == 1:
            axes[ax_idx].bar(range(minutes_in_hour), complete_post_lists_min[user_idx], color='lightgray', width=1)
            axes[ax_idx].bar(range(minutes_in_hour), sample_post_lists_min[user_idx], color=filled_colors[ax_idx], alpha=0.8, width=1)
            axes[ax_idx].plot(range(minutes_in_hour), estimated_post_lists_min[user_idx], 'k-', lw=1.5)
            axes[ax_idx].set_xlabel('minute', fontsize=12)
            axes[ax_idx].set_xlim([-1, minutes_in_hour+1])
            axes[ax_idx].set_xticks([0, 15, 30, 45, 60])
        elif min_entropy_idx == 2:
            axes[ax_idx].bar(range(seconds_in_minute), complete_post_lists_sec[user_idx], color='lightgray', width=1)
            axes[ax_idx].bar(range(seconds_in_minute), sample_post_lists_sec[user_idx], color=filled_colors[ax_idx], alpha=0.8, width=1)
            axes[ax_idx].plot(range(seconds_in_minute), estimated_post_lists_sec[user_idx], 'k-', lw=1.5)
            axes[ax_idx].set_xlabel('second', fontsize=12)
            axes[ax_idx].set_xlim([-1, seconds_in_minute+1])
            axes[ax_idx].set_xticks([0, 15, 30, 45, 60])
        elif min_entropy_idx == 3:
            axes[ax_idx].bar(range(num_bins), complete_post_lists_10ms[user_idx], color='lightgray', width=1)
            axes[ax_idx].bar(range(num_bins), sample_post_lists_10ms[user_idx], color=filled_colors[ax_idx], alpha=0.8, width=1)
            axes[ax_idx].plot(range(num_bins), estimated_post_lists_10ms[user_idx], 'k-', lw=1.5)
            axes[ax_idx].set_xlabel('millisecond', fontsize=12)
            axes[ax_idx].set_xlim([-3, num_bins+3])
            axes[ax_idx].set_xticks([0, 25, 50, 75, 100])
            axes[ax_idx].xaxis.set_major_formatter(FuncFormatter(lambda x, _: 10*x))

        axes[ax_idx].tick_params(axis='both', which='major', labelsize=11)
        axes[ax_idx].set_title('{0} {1}'.format(labels[ax_idx], case_user_screennames[ax_idx]), fontsize=13)

    axes[0].set_ylabel('volume', fontsize=12)

    hide_spines(axes)

    timer.stop()

    plt.tight_layout()
    plt.savefig('../images/suspicious_users.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
