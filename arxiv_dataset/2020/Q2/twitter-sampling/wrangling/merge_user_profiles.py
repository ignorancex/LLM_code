#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Merge all user profiles into one file.

Usage: python merge_user_profiles.py
Input data files: ../data/[app_name]_out/[app_name]_*_user.txt.bz2
Output data files: ../data/[app_name]_out/complete_user_profiles.txt
Time: ~1H
"""

import sys, os, bz2

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer


def main():
    app_name = 'cyberbullying'
    if app_name == 'cyberbullying':
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', 'all']
    elif app_name == 'youtube':
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'all']
    else:
        target_suffix = ['1', '2', '3', '4', '5', '6', '7', '8', 'all']

    archive_dir = '../data/{0}_out'.format(app_name)

    timer = Timer()
    timer.start()

    print('>>> Merging user profile')

    fout = open(os.path.join(archive_dir, 'complete_user_profile_{0}.txt'.format(app_name)), 'w')
    visited_user_id_str = set()
    num_users = 0
    for suffix in target_suffix:
        with bz2.BZ2File(os.path.join(archive_dir, '{0}_{1}_user.txt.bz2'.format(app_name, suffix)), mode='r') as fin:
            for line in fin:
                line = line.decode('utf8')
                user_id_str, _ = line.rstrip().split(',', 1)
                if user_id_str not in visited_user_id_str:
                    fout.write(line)
                    visited_user_id_str.add(user_id_str)
                    num_users += 1
    print('>>> We retrieve profiles for {0} users'.format(num_users))
    fout.close()

    timer.stop()


if __name__ == '__main__':
    main()
