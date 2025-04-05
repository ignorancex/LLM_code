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
from scipy.stats import entropy, kendalltau

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, melt_snowflake
from utils.metrics import mean_absolute_percentage_error as mape

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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), gridspec_kw={'width_ratios': [2.75, 3]})
    axes = axes.ravel()

    confusion_sampling_rate = np.load('../data/{0}_out/{0}_confusion_sampling_rate.npy'.format(app_name))
    confusion_sampling_rate = np.nan_to_num(confusion_sampling_rate)

    load_external_data = True
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

        min_entropy, min_entropy_idx = min((min_entropy, min_entropy_idx) for (min_entropy_idx, min_entropy) in enumerate([hour_entropy, min_entropy, sec_entropy]))
        if ms10_entropy < 0.87:
            min_entropy_idx = 3
        else:
            min_entropy_idx = 2
        # # if they are all very large
        # if min_entropy >= msly_entropy_benchmark:
        #     min_entropy_idx = 2

        num_estimate = sum([estimated_post_lists_hour[user_idx], estimated_post_lists_min[user_idx],
                            estimated_post_lists_sec[user_idx], estimated_post_lists_10ms[user_idx]][min_entropy_idx])
        num_estimate_list.append(num_estimate)

        num_sample_list.append(num_sample)
        num_complete_list.append(num_complete)

        ret[user_id] = (num_sample, num_complete, num_estimate, min_entropy_idx)

    np.savetxt('min_sample.npy', min_mat, delimiter=',')
    np.savetxt('sec_sample.npy', sec_mat, delimiter=',')

    rank_by_sample = [k for k, v in sorted(ret.items(), key=lambda item: item[1][0], reverse=True)]
    rank_by_complete = [k for k, v in sorted(ret.items(), key=lambda item: item[1][1], reverse=True)]
    rank_by_estimated = [k for k, v in sorted(ret.items(), key=lambda item: item[1][2], reverse=True)]

    for user_idx, user_id in enumerate(sample_top):
        print(user_id, ret[user_id][:-1], (rank_by_sample.index(user_id)+1, rank_by_complete.index(user_id)+1, rank_by_estimated.index(user_id)+1))
        print(ret[user_id][0]/ret[user_id][1], mape(ret[user_id][1], ret[user_id][2])[0], rank_by_sample.index(user_id)-rank_by_complete.index(user_id), rank_by_estimated.index(user_id)-rank_by_complete.index(user_id))
        print(np.sum(np.array(sample_post_lists_min[user_idx]) > 0), np.sum(np.array(sample_post_lists_sec[user_idx]) > 0), np.sum(np.array(sample_post_lists_10ms[user_idx]) > 0))

    observed_top100 = rank_by_sample[:100]
    complete_rank_for_observed_top100 = [rank_by_complete.index(uid) + 1 for uid in observed_top100]
    user_sampling_rates_for_observed_top100 = [sample_entity_stats[uid] / complete_entity_stats[uid] for uid in observed_top100]
    print('kendall tau for observed', kendalltau(range(1, 101), complete_rank_for_observed_top100))

    estimated_top100 = rank_by_estimated[:100]
    complete_rank_for_estimated_top100 = [rank_by_complete.index(uid) + 1 for uid in estimated_top100]
    user_sampling_rates_for_estimated_top100 = [sample_entity_stats[uid] / complete_entity_stats[uid] for uid in estimated_top100]
    print('kendall tau for estimated', kendalltau(range(1, 101), complete_rank_for_estimated_top100))

    axes[0].scatter(range(1, 101), complete_rank_for_observed_top100, s=30,
                    c=user_sampling_rates_for_observed_top100,
                    edgecolors='gray',
                    vmin=0.2, vmax=0.9, cmap=cm, zorder=50)
    axes[0].set_xlabel('observed rank in sample set', fontsize=13)
    axes[0].set_ylabel('rank in complete set', fontsize=13)
    axes[0].text(0.04, 0.9, r"kendall's $\tau$: {0:.4f}".format(kendalltau(range(1, 101), complete_rank_for_observed_top100)[0]),
                 ha='left', va='top', size=12, transform=axes[0].transAxes)
    axes[0].plot([0, 100], [100, 100], color='gray', ls='--', lw=1)
    axes[0].plot([100, 100], [0, 100], color='gray', ls='--', lw=1)
    axes[0].plot([0, 100], [0, 100], color='gray', ls='--', lw=1)
    axes[0].set_title('(a)', fontsize=13)

    sc = axes[1].scatter(range(1, 101), complete_rank_for_estimated_top100, s=30,
                         c=user_sampling_rates_for_estimated_top100,
                         edgecolors='gray',
                         vmin=0.2, vmax=0.9, cmap=cm, zorder=50)
    axes[1].set_xlabel('estimated rank in sample set', fontsize=13)
    axes[1].plot([0, 100], [100, 100], color='gray', ls='--', lw=1)
    axes[1].plot([100, 100], [0, 100], color='gray', ls='--', lw=1)
    axes[1].plot([0, 100], [0, 100], color='gray', ls='--', lw=1)
    axes[1].text(0.04, 0.9, r"kendall's $\tau$: {0:.4f}".format(kendalltau(range(1, 101), complete_rank_for_estimated_top100)[0]),
                 ha='left', va='top', size=12, transform=axes[1].transAxes)
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_title('(b)', fontsize=13)

    cb = plt.colorbar(sc, fraction=0.055)
    cb.set_label(label='user sampling rate', size=13)
    cb.ax.tick_params(labelsize=11)

    for ax in axes[:2]:
        ax.set_xlim([-4, 104])
        ax.set_ylim(bottom=-4)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 50, 100])
        ax.tick_params(axis='both', which='major', labelsize=11)

    timer.stop()

    plt.tight_layout()
    plt.savefig('../images/top_entity_rank.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
