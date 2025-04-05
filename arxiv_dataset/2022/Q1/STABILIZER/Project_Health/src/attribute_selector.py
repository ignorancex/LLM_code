import pandas as pd
import numpy as np

import utils

import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path
import sys
import os
import copy
import traceback
import timeit
import random

def calculate_median(repo_pool, path, goal):
    feature_selected = []
    repos = []
    for repo in sorted(repo_pool):
        try:
            data = utils.data_goal_arrange(repo, path, goal)
            data = data.iloc[:,:-1]
            data_median = data.median().values.tolist()
            feature_selected.append(data_median)
            repos.append(repo)
        except Exception as e:
            print(e)
            continue
    return feature_selected, repos


if __name__ == '__main__':
    repo_pool = []
    path = 'data/data_use/'
    for filename in os.listdir(path):
        if not filename.startswith('.'):
            repo_pool.append(os.path.join(filename))

    for _goal in range(7):
        print(_goal)
        goal = utils.get_goal(_goal)
        df_cols = utils.data_goal_arrange(repo_pool[0], path, _goal).columns[:-1]
        feature_selected, repos = calculate_median(repo_pool, path, _goal)
        median_df = pd.DataFrame(feature_selected, columns = df_cols, index = repos)
        median_df.to_csv('results/attribute/data_attribute_' + goal + '.csv')
