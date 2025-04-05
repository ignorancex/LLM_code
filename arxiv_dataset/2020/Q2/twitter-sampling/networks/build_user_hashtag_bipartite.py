#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract timestamp_ms, user posting, hashtag, and user mentioned.
In the process, correct timestamp_ms in rate limit message, remove duplicate tweets, and sort tweets chronologically.

Usage: python build_user_hashtag_bipartite.py
Input data files: ../data/[app_name]_out/complete_[user|hashtag]_[app_name].txt, ../data/[app_name]_out/[user|hashtag]_[app_name]_all.txt
Output data files: ../data/[sample|complete]_hashtag_user_stats.p, ../data/[sample|complete]_user_hashtag_stats.p
Time: ~30M
"""

import sys, os, pickle
from collections import defaultdict, Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer


def main():
    timer = Timer()
    timer.start()

    app_name = 'cyberbullying'

    complete_uid_embed_dict = {}
    with open('../networks/{0}_embed_user.txt'.format(app_name), 'r') as fin:
        for line in fin:
            embed, user = line.rstrip().split(',')
            complete_uid_embed_dict[user] = embed

    complete_hashtag_embed_dict = {}
    with open('../networks/{0}_embed_hashtag.txt'.format(app_name), 'r', encoding='utf-8') as fin:
        for line in fin:
            embed, hashtag = line.rstrip().split(',')
            complete_hashtag_embed_dict[hashtag] = embed

    for date_type in ['sample', 'complete']:
        if date_type == 'sample':
            user_datefile = open('../data/{0}_out/user_{0}_all.txt'.format(app_name), 'r')
            hashtag_datefile = open('../data/{0}_out/hashtag_{0}_all.txt'.format(app_name), 'r', encoding='utf-8')
        else:
            user_datefile = open('../data/{0}_out/complete_user_{0}.txt'.format(app_name), 'r')
            hashtag_datefile = open('../data/{0}_out/complete_hashtag_{0}.txt'.format(app_name), 'r', encoding='utf-8')

        tid_set = set()
        tid_uid_dict = {}
        for line in user_datefile:
            tid, user, _ = line.rstrip().split(',', 2)
            tid_set.add(tid)
            tid_uid_dict[tid] = complete_uid_embed_dict[user]

        uid_hid_dict = defaultdict(list)
        hid_uid_dict = defaultdict(list)
        uid_tweet_dict = defaultdict(int)
        hid_tweet_dict = defaultdict(int)
        for line in hashtag_datefile:
            tid, hashtags = line.rstrip().lower().split(',', 1)
            hashtags = hashtags.split(',')
            hids = [complete_hashtag_embed_dict[hashtag] for hashtag in hashtags]
            uid_hid_dict[tid_uid_dict[tid]].extend(hids)
            uid_tweet_dict[tid_uid_dict[tid]] += 1
            for hid in hids:
                hid_uid_dict[hid].append(tid_uid_dict[tid])
                hid_tweet_dict[hid] += 1

        print('in {0} set, {1} hashtags mentioned by {2} users'.format(date_type, len(hid_uid_dict), len(uid_hid_dict)))
        uid_hid_stats = {uid: [uid_tweet_dict[uid]] + list(Counter(uid_hid_dict[uid]).items()) for uid in uid_hid_dict}
        hid_uid_stats = {hid: [hid_tweet_dict[hid]] + list(Counter(hid_uid_dict[hid]).items()) for hid in hid_uid_dict}

        test = sorted(uid_hid_stats.keys(), key=lambda x: len(uid_hid_dict[x]), reverse=True)
        print(test[0])
        print(uid_hid_stats[test[0]])
        print(test[1])
        print(uid_hid_stats[test[1]])

        pickle.dump(uid_hid_stats, open('../networks/{0}_uid_hid_stats.p'.format(date_type), 'wb'))
        pickle.dump(hid_uid_stats, open('../networks/{0}_hid_uid_stats.p'.format(date_type), 'wb'))

    timer.stop()


if __name__ == '__main__':
    main()
