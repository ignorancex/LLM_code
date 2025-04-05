import pandas as pd
from datetime import datetime
import numpy as np
from scipy.io.arff import loadarff
import os


def data_remove_zero(repo_name, directory):
    df_raw = pd.read_csv(directory + repo_name, sep=',')
    df_raw = df_raw.drop(columns=['monthly_commit_comments'])
    for index, row in df_raw.iterrows():
        if int(row['dates'][0:4]) < 2015:
            df_raw.drop(index, inplace=True)
    for index, row in df_raw.iterrows():
        if int(row['monthly_closed_issues']) == 0:
            df_raw.drop(index, inplace=True)
        else:
            break
    df_raw.to_csv(repo_name, index=False, encoding='utf-8')
    return df_raw


if __name__ == '__main__':

    repo_pool = []
    path = r'../data/data_use/'

    for filename in os.listdir(path):
        repo_pool.append(os.path.join(filename))

    for repo in repo_pool:
        data_remove_zero(repo, path)
