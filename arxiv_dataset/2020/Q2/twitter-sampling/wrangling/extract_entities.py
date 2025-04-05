#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract timestamp_ms, user posting, hashtag, and user mentioned.
In the process, correct timestamp_ms in rate limit message, remove duplicate tweets, and sort tweets chronologically.

Usage: python extract_entities.py
Input data files: ../data/[app_name]_out/*.txt.bz
Output data files: ../data/[app_name]_out/[ts|user|vid|mention|hashtag|retweet]_*.txt
Time: ~4H
"""

import sys, os, bz2
from datetime import datetime, timezone
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer, make_snowflake, melt_snowflake


def main():
    timer = Timer()
    timer.start()

    app_name = 'cyberbullying'
    if app_name == 'cyberbullying':
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', 'all']
    elif app_name == 'youtube':
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'all']
    else:
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', 'all']

    os.makedirs('../data/{0}_out'.format(app_name), exist_ok=True)

    # load disconnect msg
    disconnect_dict = {k: [] for k in target_suffix}
    if os.path.exists('../log/{0}_crawl.log'.format(app_name)):
        with open('../log/{0}_crawl.log'.format(app_name), 'r') as fin:
            for line in fin:
                split_line = line.rstrip().split()
                timestamp_ms = int(datetime.strptime('{0} {1}'.format(split_line[0], split_line[1][:-4]), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()) * 1000 + int(split_line[1][-3:])
                disconnect_dict[split_line[4].split('_')[1]].append(timestamp_ms)

    # ratemsg_offset = []
    # for suffix_idx, suffix in enumerate(target_suffix):
    #     suffix_dir = '{0}_{1}'.format(app_name, suffix)
    #     input_path = '../data/{0}_out/{1}.txt.bz2'.format(app_name, suffix_dir)
    #
    #     with bz2.BZ2File(input_path, mode='r') as fin:
    #         ratemsg_ts_streaming_dict = {}
    #
    #         last_timestamp_ms = 0
    #         last_ratemsg_timestamp_ms = 0
    #         has_ratemsg = False
    #
    #         for line in fin:
    #             split_line = line.decode('utf8').rstrip().split(',')
    #             if len(split_line) == 3:
    #                 last_ratemsg_timestamp_ms = int(split_line[1])
    #                 ratemsg_ts_streaming_dict[last_ratemsg_timestamp_ms] = 'ratemsg{0}-{1}'.format(suffix, split_line[2])
    #                 has_ratemsg = True
    #             else:
    #                 current_timestamp_ms = int(split_line[2])
    #
    #                 if has_ratemsg:
    #                     corrected_ratemsg_ts = (last_timestamp_ms + current_timestamp_ms) // 2
    #                     ratemsg_offset.append(last_ratemsg_timestamp_ms - corrected_ratemsg_ts)
    #                     has_ratemsg = False
    #
    #                 last_timestamp_ms = current_timestamp_ms
    #         print('>>> Loaded rate limit message timestamp offset for {0}...'.format(input_path))
    #
    # if len(ratemsg_offset) > 0:
    #     print('>>> rate limit message timestamp_ms offset')
    #     print('|  min  |  10th  |  25th  | median |  75th  |  90th  |  max  |  mean  |  std  |')
    #     print('|{0: ^7.0f}|{1: ^8.0f}|{2: ^8.0f}|{3: ^8.0f}|{4: ^8.0f}|{5: ^8.0f}|{6: ^7.0f}|{7: ^8.0f}|{8: ^7.0f}|'
    #           .format(np.min(ratemsg_offset), np.percentile(ratemsg_offset, 10), np.percentile(ratemsg_offset, 25),
    #                   np.median(ratemsg_offset), np.percentile(ratemsg_offset, 75), np.percentile(ratemsg_offset, 90),
    #                   np.max(ratemsg_offset), np.mean(ratemsg_offset), np.std(ratemsg_offset)))
    #     best_offset = int(np.mean(ratemsg_offset))
    # else:
    best_offset = 5000
    print('best rate limit message timestamp_ms offset is {0}'.format(best_offset))
    # timer.stop()

    for suffix_idx, suffix in enumerate(target_suffix):
        timer = Timer()
        timer.start()

        suffix_dir = '{0}_{1}'.format(app_name, suffix)
        input_path = '../data/{0}_out/{1}.txt.bz2'.format(app_name, suffix_dir)
        ts_output_path = '../data/{0}_out/ts_{1}.txt'.format(app_name, suffix_dir)
        user_output_path = '../data/{0}_out/user_{1}.txt'.format(app_name, suffix_dir)
        vid_output_path = '../data/{0}_out/vid_{1}.txt'.format(app_name, suffix_dir)
        mention_output_path = '../data/{0}_out/mention_{1}.txt'.format(app_name, suffix_dir)
        hashtag_output_path = '../data/{0}_out/hashtag_{1}.txt'.format(app_name, suffix_dir)
        retweet_output_path = '../data/{0}_out/retweet_{1}.txt'.format(app_name, suffix_dir)
        follower_output_path = '../data/{0}_out/follower_{1}.txt'.format(app_name, suffix_dir)

        disconnect_list = disconnect_dict[suffix]

        with bz2.BZ2File(input_path, mode='r') as fin:
            min_tweet_id = None
            visited_tid = set()
            ts_streaming_dict = {}
            user_streaming_dict = {}
            vid_streaming_dict = {}
            mention_streaming_dict = {}
            hashtag_streaming_dict = {}
            tid_retweet_dict = defaultdict(set)
            root_tweet_follower_dict = {}

            for disconnect_ts in disconnect_list:
                # make a snowflake id for disconnect message
                ts_streaming_dict[str(make_snowflake(disconnect_ts, 31, 31, suffix_idx + 1))] = 'disconnect'

            for line in fin:
                split_line = line.decode('utf8').rstrip().split(',')
                if len(split_line) == 3:
                    # make a snowflake id for rate limit message
                    ts_streaming_dict[str(make_snowflake(int(split_line[1]) - best_offset, 31, 31, suffix_idx))] = 'ratemsg{0}-{1}'.format(suffix, split_line[2])
                else:
                    tweet_id = split_line[0]
                    if min_tweet_id is None:
                        min_tweet_id = tweet_id
                    else:
                        if tweet_id < min_tweet_id:
                            min_tweet_id = tweet_id
                    if tweet_id in visited_tid:
                        continue

                    ts_streaming_dict[tweet_id] = split_line[2]

                    # root_user_id_str, reply_user_id_str, retweeted_user_id_str, quoted_user_id_str
                    user_streaming_dict[tweet_id] = '{0},{1},{2},{3}'.format(split_line[3], split_line[35], split_line[36], split_line[37])

                    to_write_vid = set()
                    for vids in [split_line[5], split_line[6], split_line[7]]:
                        if vids != 'N':
                            if ';' in vids:
                                to_write_vid.update(set(vids.split(';')))
                            else:
                                to_write_vid.add(vids)
                    if len(to_write_vid) > 0:
                        vid_streaming_dict[tweet_id] = ','.join(to_write_vid)

                    to_write_mention = set()
                    for mentions in [split_line[8], split_line[9], split_line[10]]:
                        if mentions != 'N':
                            if ';' in mentions:
                                to_write_mention.update(set(mentions.split(';')))
                            else:
                                to_write_mention.add(mentions)
                    if len(to_write_mention) > 0:
                        mention_streaming_dict[tweet_id] = ','.join(to_write_mention)

                    to_write_hashtag = set()
                    for hashtags in [split_line[11], split_line[12], split_line[13]]:
                        if hashtags != 'N':
                            if ';' in hashtags:
                                to_write_hashtag.update(set(hashtags.split(';')))
                            else:
                                to_write_hashtag.add(hashtags)
                    if len(to_write_hashtag) > 0:
                        hashtag_streaming_dict[tweet_id] = ','.join(to_write_hashtag)

                    if split_line[33] != 'N' and split_line[33] >= min_tweet_id:
                        tid_retweet_dict[split_line[33]].add('{0}-{1}'.format(tweet_id, split_line[20]))

                    if split_line[32] == 'N' and split_line[33] == 'N' and split_line[34] == 'N':
                        root_tweet_follower_dict[tweet_id] = split_line[20]

                    visited_tid.add(split_line[0])

            print('>>> Loaded all data, ready to sort and dump {0}...'.format(input_path))

            with open(ts_output_path, 'w') as fout1:
                for tid in sorted(ts_streaming_dict.keys()):
                    if ts_streaming_dict[tid].startswith('ratemsg'):
                        ts = melt_snowflake(tid)[0]
                        ratesuffix, track = ts_streaming_dict[tid].split('-')
                        fout1.write('{0},{1},{2}\n'.format(ts, ratesuffix, track))
                    elif ts_streaming_dict[tid].startswith('disconnect'):
                        ts = melt_snowflake(tid)[0]
                        fout1.write('{0},{1},{2}\n'.format(ts, 'disconnect', suffix))
                    else:
                        fout1.write('{0},{1}\n'.format(ts_streaming_dict[tid], tid))

            with open(user_output_path, 'w') as fout2:
                for tid in sorted(user_streaming_dict.keys()):
                    fout2.write('{0},{1}\n'.format(tid, user_streaming_dict[tid]))

            with open(vid_output_path, 'w') as fout3:
                for tid in sorted(vid_streaming_dict.keys()):
                    fout3.write('{0},{1}\n'.format(tid, vid_streaming_dict[tid]))

            with open(mention_output_path, 'w') as fout4:
                for tid in sorted(mention_streaming_dict.keys()):
                    fout4.write('{0},{1}\n'.format(tid, mention_streaming_dict[tid]))

            with open(hashtag_output_path, 'w', encoding='utf-8') as fout5:
                for tid in sorted(hashtag_streaming_dict.keys()):
                    fout5.write('{0},{1}\n'.format(tid, hashtag_streaming_dict[tid]))

            with open(retweet_output_path, 'w') as fout6:
                for root_tweet_id in sorted(tid_retweet_dict.keys()):
                    fout6.write('{0}:{1}\n'.format(root_tweet_id, ','.join(sorted(list(tid_retweet_dict[root_tweet_id])))))

            with open(follower_output_path, 'w') as fout7:
                for root_tweet_id in sorted(root_tweet_follower_dict.keys()):
                    fout7.write('{0},{1}\n'.format(root_tweet_id, root_tweet_follower_dict[root_tweet_id]))

        timer.stop()


if __name__ == '__main__':
    main()
