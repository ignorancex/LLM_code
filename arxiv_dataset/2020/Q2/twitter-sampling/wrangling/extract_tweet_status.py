#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Entry program to extract tweet status and user status.
1. tweet_id_str, created_at, timestamp_ms, user_id_str,
   original_lang, retweeted_lang, quoted_lang,
   original_vids, retweeted_vids, quoted_vids,
   original_mentions, retweeted_mentions, quoted_mentions,
   original_hashtags, retweeted_hashtags, quoted_hashtags,
   original_geo, retweeted_geo, quoted_geo,
   original_coordinates, retweeted_coordinates, quoted_coordinates,
   original_place, retweeted_place, quoted_place,
   original_sensitive, retweeted_sensitive, quoted_sensitive,
   original_filter, retweeted_filter, quoted_filter,
   original_retweet_count, retweeted_retweet_count, quoted_retweet_count,
   original_favorite_count, retweeted_favorite_count, quoted_favorite_count,
   original_user_followers_count, retweeted_user_followers_count, quoted_user_followers_count,
   original_user_friends_count, retweeted_user_friends_count, quoted_user_friends_count,
   original_user_statuses_count, retweeted_user_statuses_count, quoted_user_statuses_count,
   original_user_favourites_count, retweeted_user_favourites_count, quoted_user_favourites_count,
   reply_tweet_id_str, retweeted_tweet_id_str, quoted_tweet_id_str,
   reply_user_id_str, retweeted_user_id_str, quoted_user_id_str,
   original_text, retweeted_text, quoted_text
2. user_id_str, screen_name, created_at, verified, location, followers_count, friends_count, listed_count, statuses_count, description
3. ratemsg, timestamp_ms, track

Usage: python extract_tweet_status.py
Input data files: ../data/[app_name]/*/*.bz2
Output data files: ../data/[app_name]_out/*.txt.bz
Time: ~4H
"""

import sys, os, bz2

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer
from wrangling.tweet_extractor import TweetExtractor


def extract_status(input_dir, output_dir, proc_num):
    """Extract tweet status from given folder, output in output_dir."""
    extractor = TweetExtractor(input_dir, output_dir)
    extractor.set_proc_num(proc_num)
    extractor.extract()


if __name__ == '__main__':
    app_name = 'cyberbullying'
    if app_name == 'cyberbullying':
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', 'all']
    elif app_name == 'youtube':
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'all']
    else:
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', 'all']

    os.makedirs('../data/{0}_out'.format(app_name), exist_ok=True)

    for suffix in target_suffix:
        timer = Timer()
        timer.start()

        suffix_dir = '{0}_{1}'.format(app_name, suffix)
        input_dir = '/mnt/siqi/data/{0}/{1}'.format(app_name, suffix_dir)
        output_dir = '../data/{0}/{1}'.format(app_name, suffix_dir)
        bz2_output_path = '../data/{0}_out/{1}.txt.bz2'.format(app_name, suffix_dir)
        bz2_user_output_path = '../data/{0}_out/{1}_user.txt.bz2'.format(app_name, suffix_dir)

        proc_num = 24
        extract_status(input_dir, output_dir, proc_num)
        print('>>> Completed extracting tweet status for {0}.'.format(suffix_dir))

        print('>>> Start to bz2 the texts...')
        # merge all files into one bz2 file
        with bz2.open(bz2_output_path, 'at') as fout:
            for subdir, _, files in os.walk(os.path.join(output_dir, 'tweet_stats')):
                for f in sorted(files):
                    with bz2.BZ2File(os.path.join(subdir, f), mode='r') as fin:
                        for line in fin:
                            fout.write(line)
        print('>>> Completed bz2 text for {0}.'.format(suffix_dir))

        print('>>> Start to bz2 the users...')
        # merge all files into one bz2 file
        visited_user_id_str = set()
        with bz2.open(bz2_user_output_path, 'at') as fout:
            for subdir, _, files in os.walk(os.path.join(output_dir, 'user_stats')):
                for f in sorted(files):
                    with bz2.BZ2File(os.path.join(subdir, f), mode='r') as fin:
                        for line in fin:
                            user_id_str, _ = line.decode('utf8').rstrip().split(',', 1)
                            if user_id_str not in visited_user_id_str:
                                fout.write(line)
                                visited_user_id_str.add(user_id_str)
        print('>>> Completed bz2 user for {0}.'.format(suffix_dir))

        timer.stop()
