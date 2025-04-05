#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Validate rate limit message by selecting unsampled segments, i.e., no ratemsg in complete dataset.

Usage: python plot_fig1_missing_in_segment.py
Input data files: ./[app_name]_out/ts_[app_name]_all.txt, ./[app_name]_out/complete_ts_[app_name].txt
Time: ~8M
"""

import sys, os, platform, copy
from datetime import datetime
import numpy as np

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, count_track, melt_snowflake
from utils.metrics import mean_absolute_percentage_error as mape
from utils.vars import ColorPalette


def to_datetime(x, _):
    return datetime.utcfromtimestamp(float(x) / 1000).strftime('%H:%M:%S')


def main():
    timer = Timer()
    timer.start()

    app_names = ['cyberbullying', 'youtube']
    # data for plot subfig (a)
    showcase_segment_idx = 0
    showcase_complete_tid_list = []
    showcase_retrieved_tid_list = []
    showcase_ratemsg_list = []
    showcase_track_list = []

    # data for plot subfig (b)
    mape_dict = {app_name: [] for app_name in app_names}

    rate_silence_length = 10000
    disconnect_silence_length = 180000
    print('>>> We silence {0} seconds around rate limit messages'.format(rate_silence_length // 1000))
    print('>>> We silence {0} seconds proceeding disconnect messages\n'.format(disconnect_silence_length // 1000))

    for app_name in app_names:
        print('>>> Computing on app {0}'.format(app_name))
        archive_dir = './{0}_out'.format(app_name)

        sample_input_path = os.path.join(archive_dir, 'ts_{0}_all.txt'.format(app_name))
        complete_input_path = os.path.join(archive_dir, 'complete_ts_{0}.txt'.format(app_name))

        # == == == == == == Part 1: Initially select segments in the complete set == == == == == == #
        # segments that silence 10s around rate limit messages and 180s proceeding disconnect messages in complete set
        init_segment_list = []
        init_start_ts = 0
        with open(complete_input_path, 'r') as fin:
            for line in fin:
                split_line = line.rstrip().split(',')
                # if it is a disconnect msg
                if 'disconnect' in split_line[1]:
                    disconnect_ts = int(split_line[0])
                    # disconnect message, remove the proceeding [disconnect_silence_length]
                    init_end_ts = disconnect_ts - disconnect_silence_length
                    if init_end_ts > init_start_ts:
                        init_segment_list.append((init_start_ts, init_end_ts, init_end_ts - init_start_ts))
                    init_start_ts = disconnect_ts
                # elif it is a rate limit msg
                elif 'ratemsg' in split_line[1]:
                    ratemsg_ts = int(split_line[0])
                    # rate limit message, remove the surrounding [rate_silence_length]
                    init_end_ts = ratemsg_ts - rate_silence_length // 2
                    if init_end_ts > init_start_ts:
                        init_segment_list.append((init_start_ts, init_end_ts, init_end_ts - init_start_ts))
                    init_start_ts = ratemsg_ts + rate_silence_length // 2
        print('>>> Initially, we identify {0} segments in complete set without rate limit message'.format(len(init_segment_list)))
        # print(init_segment_list[: 10])

        # == == == == == == Part 2: Segments are bounded by 2 rate limit messages in the sample set == == == == == == #
        bounded_segment_list = []
        current_segment_idx = 0
        current_start_ts = 0
        current_ratemsg_list = []
        current_track_list = []
        last_ratemsg_ts = 0

        look_for_end = False
        found_showcase = False

        with open(sample_input_path, 'r') as fin:
            for line in fin:
                split_line = line.rstrip().split(',')
                if 'ratemsg' in split_line[1]:
                    ratemsg_ts = int(split_line[0])
                    track = int(split_line[2])
                    if not look_for_end or (look_for_end and current_start_ts == last_ratemsg_ts and init_segment_list[current_segment_idx][1] < ratemsg_ts):
                        # fast forward, skip some really short segments
                        while ratemsg_ts >= init_segment_list[current_segment_idx][1]:
                            current_segment_idx += 1
                        if current_segment_idx == len(init_segment_list):
                            break
                        if ratemsg_ts >= init_segment_list[current_segment_idx][0]:
                            current_start_ts = ratemsg_ts
                            current_ratemsg_list = [ratemsg_ts]
                            current_track_list = [track]
                            look_for_end = True
                        else:
                            look_for_end = False
                    elif look_for_end:
                        if current_start_ts < last_ratemsg_ts <= init_segment_list[current_segment_idx][1] < ratemsg_ts:
                            current_num_miss = count_track(current_track_list, start_with_rate=True, subcrawler=False)
                            bounded_segment_list.append((current_start_ts, last_ratemsg_ts, last_ratemsg_ts - current_start_ts, current_num_miss))

                            # find the first example segment that is around 11 sec long
                            if app_name == 'cyberbullying' and not found_showcase and 10000 <= last_ratemsg_ts - current_start_ts <= 12000:
                                showcase_segment_idx = len(bounded_segment_list) - 1
                                showcase_ratemsg_list = copy.deepcopy(current_ratemsg_list)
                                showcase_track_list = copy.deepcopy(current_track_list)
                                found_showcase = True

                            current_segment_idx += 1
                            if current_segment_idx == len(init_segment_list):
                                break
                            if ratemsg_ts >= init_segment_list[current_segment_idx][0]:
                                current_start_ts = ratemsg_ts
                                current_ratemsg_list = [ratemsg_ts]
                                current_track_list = [track]
                                look_for_end = True
                            else:
                                look_for_end = False
                        else:
                            current_ratemsg_list.append(ratemsg_ts)
                            current_track_list.append(track)

                    last_ratemsg_ts = ratemsg_ts
                    if current_segment_idx == len(init_segment_list):
                        break
        print('>>> We further bound {0} segments with 2 rate limit messages'.format(len(bounded_segment_list)))
        # print(bounded_segment_list[-10:])

        # == == == == == == Part 3: Add sample and complete volume in each segment == == == == == == #
        for input_path, tid_list in zip([sample_input_path, complete_input_path], [showcase_retrieved_tid_list, showcase_complete_tid_list]):
            current_segment_idx = 0
            current_segment_cnt = 0
            with open(input_path, 'r') as fin:
                for line in fin:
                    split_line = line.rstrip().split(',')
                    if len(split_line) == 2:
                        msg_ts = int(split_line[0])
                        if bounded_segment_list[current_segment_idx][0] < msg_ts <= bounded_segment_list[current_segment_idx][1]:
                            current_segment_cnt += 1

                            if app_name == 'cyberbullying' and current_segment_idx == showcase_segment_idx:
                                tweet_id = split_line[1]
                                tid_list.append(tweet_id)
                        elif msg_ts > bounded_segment_list[current_segment_idx][1]:
                            bounded_segment_list[current_segment_idx] = (*bounded_segment_list[current_segment_idx], current_segment_cnt)
                            current_segment_idx += 1
                            current_segment_cnt = 0
                            if current_segment_idx == len(bounded_segment_list):
                                break
            # print(bounded_segment_list[-10:])

        length_tracker = 0
        mape_list = []
        for segment in bounded_segment_list:
            length_tracker += segment[2]
            mape_list.append(mape(segment[-1], segment[-2] + segment[-3]))
        mape_dict[app_name] = copy.deepcopy(mape_list)
        print('MAPE: {0:.5f} +- {1:.5f}, median: {2:.5f}'.format(np.mean(mape_list), np.std(mape_list), np.median(mape_list)))
        print('total tracked days bounded: {0:.2f} out of 14'.format(length_tracker / 1000 / 60 / 60 / 24))

        if app_name == 'cyberbullying':
            print('complete tweets: {0}, retrieved tweets: {1}, estimated missing: {2}'
                  .format(len(showcase_complete_tid_list),
                          len(showcase_retrieved_tid_list),
                          count_track(showcase_track_list, start_with_rate=True, subcrawler=False)))
            print('ratemsg timestamp', showcase_ratemsg_list)
            print('ratemsg track', showcase_track_list)
        print()

    timer.stop()

    # == == == == == == Part 5: Plot a showcase segment that is roughly 10s == == == == == == #
    cc4 = ColorPalette.CC4
    blue = cc4[0]
    green = cc4[1]
    red = cc4[3]
    fig, axes = plt.subplots(1, 4, figsize=(12, 1.6))
    ax2 = axes[-1]
    gs = axes[1].get_gridspec()
    for ax in axes[:-1]:
        ax.remove()
    ax1 = fig.add_subplot(gs[:-1])

    # add a timeline
    ax1.axhline(0, linewidth=2, color='k')

    observed_tweet_ts_list = sorted([melt_snowflake(tid)[0] for tid in showcase_retrieved_tid_list])
    showcase_missing_tid_set = set(showcase_complete_tid_list).difference(set(showcase_retrieved_tid_list))
    missing_tweet_ts_list = sorted([melt_snowflake(tid)[0] for tid in showcase_missing_tid_set])
    ax1.scatter(observed_tweet_ts_list, [1] * len(observed_tweet_ts_list), marker='o', facecolors='none', edgecolors=blue, lw=1, s=20)
    ax1.scatter(missing_tweet_ts_list, [0.5] * len(missing_tweet_ts_list), marker='x', c='k', lw=1, s=20)
    # stats for missing tweets, cut by rate limit msg timestamp_ms
    complete_track_list = []
    i, j, curr_cnt = 0, 1, 0
    while i < len(missing_tweet_ts_list) and j < len(showcase_ratemsg_list):
        if missing_tweet_ts_list[i] <= showcase_ratemsg_list[j]:
            curr_cnt += 1
            i += 1
        else:
            complete_track_list.append(curr_cnt)
            curr_cnt = 0
            j += 1
    complete_track_list.append(curr_cnt)
    # print(complete_track_list)

    for idx, ts in enumerate(showcase_ratemsg_list):
        ax1.axvline(ts, ymin=0, ymax=1.1, linewidth=1, color='k')

    for idx, ts in enumerate(showcase_ratemsg_list[1:]):
        ax1.text(ts - 50, 0.42, '/{0:>3}'.format(complete_track_list[idx]),
                 color='k', ha='right', va='top', size=10)
        ax1.text(ts - 470, 0.42, str(showcase_track_list[idx+1] - showcase_track_list[idx]),
                 color=green, ha='right', va='top', size=10)

    ax1.xaxis.set_major_formatter(FuncFormatter(to_datetime))
    ax1.set_xlim(left=showcase_ratemsg_list[0]-200, right=showcase_ratemsg_list[-1]+200)
    ax1.set_yticks([0.5, 1.0])
    ax1.set_ylim(top=1.2, bottom=0)
    num_missing_by_counting = len(showcase_complete_tid_list) - len(showcase_retrieved_tid_list)
    num_missing_by_estimating = count_track(showcase_track_list, start_with_rate=True, subcrawler=False)
    num_observed_tweets = len(showcase_retrieved_tid_list)
    ax1.tick_params(axis='x', which='major', labelsize=10)
    ax1.tick_params(axis='y', which='both', length=0)
    ax1.set_yticklabels(['missing tweets\n{0}/{1}'.format(num_missing_by_estimating, num_missing_by_counting),
                         'collected tweets\n{0}'.format(num_observed_tweets)], fontsize=10)

    # remove borders
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_title('(a)', fontsize=11, pad=-1.35*72, y=1.0001)

    bplot = ax2.boxplot([mape_dict['cyberbullying'], mape_dict['youtube']], labels=['Cyberbullying', 'YouTube'],
                        widths=0.5, showfliers=False, showmeans=False,
                        patch_artist=True)

    for patch, color in zip(bplot['boxes'], [blue, red]):
        patch.set_facecolor(color)

    for median in bplot['medians']:
        median.set(color='k', linewidth=1)

    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_ylabel('MAPE', fontsize=10)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title('(b)', fontsize=11, pad=-1.35*72, y=1.0001)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('../images/validate_ratemsg.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
