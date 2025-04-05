#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Merge all subcrawlers into one stream.
For the ground-truth stream, the lower bound is the merged tweets (as they are what we observed/what happened).
The upper bound is the number of merged tweets plus the sum of all rate limit messages,
assuming all missing tweets in each sub-crawler are disjointed, representing a upper bound of total missing tweets.

Usage: python merge_subcrawlers.py
Input data files: ../data/[app_name]_out/[ts|user|vid|hashtag|mention|retweet]_*.txt, ../log/[app_name]_crawl.log
Output data files: ../data/[app_name]_out/complete_[ts|user|vid|hashtag|mention|retweet]_[app].txt
Time: ~1H
"""

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer


def find_next_item(nextline_list):
    end_flag = all(v.rstrip() == '' for v in nextline_list)
    if end_flag:
        return None, None, True
    else:
        lst = []
        for item in nextline_list:
            if item.rstrip() == '':
                lst.append('END')
            else:
                lst.append(item)
        index_min = min(range(len(lst)), key=lst.__getitem__)
        return index_min, lst[index_min], False


def main():
    app_name = 'cyberbullying'
    if app_name == 'cyberbullying':
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', 'all']
    elif app_name == 'youtube':
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'all']
    else:
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', 'all']

    archive_dir = '../data/{0}_out'.format(app_name)

    # extract retweet cascade
    timer = Timer()
    timer.start()

    print('>>> Merging entity retweet cascade')

    # get sample cascade
    root_tweet_follower_dict = {}
    with open(os.path.join(archive_dir, 'follower_{0}_all.txt'.format(app_name)), 'r') as fin:
        for line in fin:
            root_tweet_id, root_user_follower = line.rstrip().split(',')
            root_user_follower = int(root_user_follower)
            root_tweet_follower_dict[root_tweet_id] = root_user_follower

    with open(os.path.join(archive_dir, 'sample_retweet_{0}.txt'.format(app_name)), 'w') as fout:
        with open(os.path.join(archive_dir, 'retweet_{0}_all.txt'.format(app_name)), 'r') as fin:
            for line in fin:
                root_tweet_id, cascade = line.rstrip().split(':')
                if root_tweet_id in root_tweet_follower_dict:
                    fout.write('{0}-{1}:{2}\n'.format(root_tweet_id, root_tweet_follower_dict[root_tweet_id], cascade))

    # get complete cascade
    root_tweet_follower_dict = {}
    follower_file_list = ['follower_{0}_{1}.txt'.format(app_name, suffix) for suffix in target_suffix]
    for follower_file in follower_file_list:
        with open(os.path.join(archive_dir, follower_file), 'r') as fin:
            for line in fin:
                root_tweet_id, root_user_follower = line.rstrip().split(',')
                root_user_follower = int(root_user_follower)
                root_tweet_follower_dict[root_tweet_id] = root_user_follower

    retweet_file_list = ['retweet_{0}_{1}.txt'.format(app_name, suffix) for suffix in target_suffix]
    retweet_file_handles = [open(os.path.join(archive_dir, retweet_file), 'r') for retweet_file in retweet_file_list]

    with open(os.path.join(archive_dir, 'complete_retweet_{0}.txt'.format(app_name)), 'w') as fout:
        nextline_list = [retweet_file.readline() for retweet_file in retweet_file_handles]

        while True:
            end_flag = all(v.rstrip() == '' for v in nextline_list)
            if end_flag:
                break
            else:
                root_tid_list = []
                children_tid_list = []

                for item in nextline_list:
                    if item.rstrip() == '':
                        root_tid_list.append('END')
                        children_tid_list.append('END')
                    else:
                        root_tid_list.append(item.rstrip().split(':')[0])
                        children_tid_list.append(set(item.rstrip().split(':')[1].split(',')))

                minimum_root_tid = min(root_tid_list)
                min_indices = [i for i, v in enumerate(root_tid_list) if v == minimum_root_tid]

                if minimum_root_tid in root_tweet_follower_dict:
                    min_children_tid_set = set()
                    for idx in min_indices:
                        min_children_tid_set.update(children_tid_list[idx])
                    fout.write('{0}-{1}:{2}\n'.format(minimum_root_tid,
                                                      root_tweet_follower_dict[minimum_root_tid],
                                                      ','.join(sorted(list(min_children_tid_set)))))

                for idx in min_indices:
                    nextline_list[idx] = retweet_file_handles[idx].readline()

    for retweet_file in retweet_file_handles:
        retweet_file.close()

    timer.stop()

    # merge other entities
    entities = ['ts', 'user', 'vid', 'hashtag', 'mention']
    for entity in entities:
        timer = Timer()
        timer.start()

        print('>>> Merging entity {0}'.format(entity))

        inputfile_list = ['{0}_{1}_{2}.txt'.format(entity, app_name, suffix) for suffix in target_suffix]
        inputfile_handles = [open(os.path.join(archive_dir, inputfile), 'r', encoding='utf-8') for inputfile in inputfile_list]
        visited_item_set = set()

        with open(os.path.join(archive_dir, 'complete_{0}_{1}.txt'.format(entity, app_name)), 'w', encoding='utf-8') as fout:
            nextline_list = [inputfile.readline() for inputfile in inputfile_handles]

            while True:
                next_idx, next_item, end_flag = find_next_item(nextline_list)
                if end_flag:
                    break
                # omit rate limit messages in the all crawler
                if 'ratemsgall' not in next_item:
                    if 'ratemsg' in next_item or 'disconnect' in next_item:
                        fout.write(next_item)
                    elif next_item not in visited_item_set:
                        fout.write(next_item)
                        visited_item_set.add(next_item)
                nextline_list[next_idx] = inputfile_handles[next_idx].readline()

        for inputfile in inputfile_handles:
            inputfile.close()

        timer.stop()


if __name__ == '__main__':
    main()
