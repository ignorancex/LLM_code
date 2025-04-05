import pandas as pd
import numpy as np
from statistics import stdev, median
from scipy.io.arff import loadarff
import os


repo_pool = []
path = r'../data/data_use/'
# repo = "cantata_monthly.csv"
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(data_github_monthly("abp_monthly.csv", path, 4))


def month_counter():
    i = 0
    for filename in os.listdir(path):
        repo_pool.append(os.path.join(filename))
    for repo in repo_pool:
        df_raw = pd.read_csv(path + repo, sep=',')
        i += len(df_raw.index)

    return i


def data_stats(column):
    stats_list = []
    for filename in os.listdir(path):
        repo_pool.append(os.path.join(filename))

    print(pd.read_csv(path + repo_pool[0], sep=',').columns.values[column])

    for repo in repo_pool:
        df_raw = pd.read_csv(path + repo, sep=',')
        for i in range(len(df_raw.index)):
            stats_list.append(df_raw.iloc[i, column])

    print("min", min(stats_list))
    print("max", max(stats_list))
    print("mean", np.mean(stats_list))
    print("median", median(stats_list))
    print("std", stdev(stats_list))
    print("iqr", np.subtract(*np.percentile(stats_list, [75, 25])))


if __name__ == '__main__':
    data_stats(12)
