#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" .

Usage: python plot_SI_fig3_rate_firehose.py
Input data files: ./firehose_stream.txt, ./filtered_stream.txt
Time: ~1M
"""

import sys, os, platform, re
from datetime import datetime, timedelta
from collections import defaultdict
from urllib.parse import urlparse

import matplotlib as mpl
if platform.system() == 'Linux':
    mpl.use('Agg')  # no UI backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, melt_snowflake, make_snowflake
from utils.vars import ColorPalette


def to_datetime(x, _):
    return x.strftime('%H:%M:%S')


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

    start_timestamp = '1483660780000'  # January 5, 2017 23:59:40
    end_timestamp = '1483662210000'  # January 6, 2017 0:23:30
    start_time_obj = datetime.utcfromtimestamp(int(start_timestamp[:-3]))
    end_time_obj = datetime.utcfromtimestamp(int(end_timestamp[:-3]))
    total_duration = (end_time_obj - start_time_obj).seconds
    x_axis = [start_time_obj + timedelta(seconds=x) for x in range(total_duration)]
    start_tid = str(make_snowflake(start_timestamp, 0, 0, 0))
    end_tid = str(make_snowflake(end_timestamp, 0, 0, 0))

    firehose_stats = defaultdict(int)
    firehose_cnt = 0
    firehose_tid = set()
    to_check = False
    with open('./firehose_stream.txt', 'r') as fin:
        for line in fin:
            line = line.rstrip()
            if line != '':
                if line.startswith('tweet_id'):
                    tweet_id = line.split(':')[-1]
                    to_check = True
                if to_check:
                    if line.startswith('tweet_url'):
                        parsed_url = urlparse(line.split(':')[-1])
                        if 'youtube.com' in parsed_url.netloc:
                            timestamp_ms = melt_snowflake(tweet_id)[0]
                            timestamp = datetime.utcfromtimestamp(timestamp_ms // 1000)
                            firehose_stats[timestamp] += 1
                            firehose_cnt += 1
                            firehose_tid.add(tweet_id)
                            to_check = False
                    if line.startswith('text:'):
                        if 'youtube' in line.lower().split(':', 1)[-1].split():
                            timestamp_ms = melt_snowflake(tweet_id)[0]
                            timestamp = datetime.utcfromtimestamp(timestamp_ms // 1000)
                            firehose_stats[timestamp] += 1
                            firehose_cnt += 1
                            firehose_tid.add(tweet_id)
                            to_check = False
    print('{0}/{1} tweets were collected via Firehose'.format(firehose_cnt, len(firehose_tid)))

    filtered_stats = defaultdict(int)
    filtered_cnt = 0
    ratemsg_ts_list = []
    ratemsg_track_list = []
    filtered_tid = set()
    with open('./filtered_stream.txt', 'r') as fin:
        for line in fin:
            if len(line.rstrip().split(',')) > 3:
                tweet_id = line.rstrip().split(',')[0]
                if start_tid <= tweet_id <= end_tid:
                    timestamp_ms = melt_snowflake(tweet_id)[0]
                    timestamp = datetime.utcfromtimestamp(timestamp_ms // 1000)
                    filtered_stats[timestamp] += 1
                    filtered_cnt += 1
                    filtered_tid.add(tweet_id)
            else:
                _, timestamp_ms, track = line.rstrip().split(',')
                track = int(track)
                ratemsg_ts_list.append(timestamp_ms)
                ratemsg_track_list.append(track)
    print('{0}/{1} tweets were collected via filtered stream'.format(filtered_cnt, len(filtered_tid)))

    split_track_lst, split_ts_lst = map_ratemsg(ratemsg_track_list, ratemsg_ts_list)
    missing_stats = {x: 0 for x in x_axis}
    last_track = 0
    for track_lst, ts_lst in zip(split_track_lst, split_ts_lst):
        for track, ts in zip(track_lst, ts_lst):
            timestamp = datetime.utcfromtimestamp(int(ts[:-3]))
            if timestamp in x_axis:
                missing_stats[timestamp] += (track - last_track)
            last_track = track

    # == == == == == == Part 5: Plot a showcase segment that is roughly 10s == == == == == == #
    cc4 = ColorPalette.CC4
    blue = cc4[0]
    red = cc4[3]
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 2))

    firehose_y_axis = [firehose_stats[x] if x in firehose_stats else 0 for x in x_axis]
    filtered_y_axis = [filtered_stats[x] if x in filtered_stats else 0 for x in x_axis]
    missing_y_axis = [missing_stats[x] if x in missing_stats else 0 for x in x_axis]
    estimated_y_axis = [filtered_y_axis[x] + missing_y_axis[x] for x in range(len(x_axis))]

    ax1.plot_date(x_axis, firehose_y_axis, '-', c='lightgrey', lw=1, label='firehose: {0:,}'.format(sum(firehose_y_axis)))
    ax1.plot_date(x_axis, filtered_y_axis, '-', c=blue, lw=1, label='filtered: {0:,}'.format(sum(filtered_y_axis)))
    ax1.plot_date(x_axis, estimated_y_axis, '-', c=red, lw=1, label='estimated: {0:,}'.format(sum(estimated_y_axis)))

    print('MAPE against Firehose volume: {0:.3f}%'.format(abs(sum(estimated_y_axis) - sum(firehose_y_axis)) / sum(firehose_y_axis) * 100))

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.set_xlabel('Jan 06, 2017', fontsize=12)
    ax1.set_ylabel('#tweets', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(frameon=False, loc='upper left', fontsize=12, ncol=3)

    # remove borders
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig('../images/SI_ratemsg_firehose.pdf', bbox_inches='tight')
    if not platform.system() == 'Linux':
        plt.show()


if __name__ == '__main__':
    main()
