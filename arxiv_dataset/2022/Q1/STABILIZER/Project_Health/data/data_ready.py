import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
import os


def data_github_monthly(repo_name, directory, goal):
    df_raw = pd.read_csv(directory + repo_name, sep=',')
    df_raw = df_raw.drop(columns=['dates'])
    last_col = ''
    if goal == 0:
        last_col = 'monthly_commits'
    elif goal == 1:
        last_col = 'monthly_contributors'
    elif goal == 2:
        last_col = 'monthly_stargazer'
    elif goal == 3:
        last_col = 'monthly_open_PRs'
    elif goal == 4:
        last_col = 'monthly_closed_PRs'
    elif goal == 5:
        last_col = 'monthly_open_issues'
    elif goal == 6:
        last_col = 'monthly_closed_issues'

    cols = list(df_raw.columns.values)
    cols.pop(cols.index(last_col))
    df_adjust = df_raw[cols+[last_col]]

    return df_adjust


if __name__ == '__main__':
    repo_pool = []
    path = r'../data/data_use/'
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(data_github_monthly("abp_monthly.csv", path, 4))

    for filename in os.listdir(path):
        repo_pool.append(os.path.join(filename))
    for repo in repo_pool:
        print(data_github_monthly(repo, path, 4).iloc[-1,-1])

