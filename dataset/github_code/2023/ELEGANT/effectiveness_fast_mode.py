import os
import fnmatch
import pandas as pd
import numpy as np
from certify_K import certify_K
from operator import itemgetter
import glob



results = {}

dataframes_set = {}

def read_files(directory_path, names):

    file_names = os.listdir(directory_path)
    dataframes = []

    for file_name in file_names:
        if all(sub in file_name for sub in names):
            df = pd.read_csv(os.path.join(directory_path, file_name), sep='\t')
            dataframes.append(df)

    return dataframes


for gnn in ['sage', 'gcn', 'jk']:
    for dataset in ['german', 'credit', 'bail']:
        dataframes_set[(dataset, gnn)] = read_files('./exps_results/', [gnn, dataset])


for key, value in dataframes_set.items():

    correct_ratio_list = []
    for one_df in value:

        to_be_used = one_df.loc[:, "correct"].to_numpy()
        to_be_used_index = np.array(np.where(to_be_used==1))
        correct = one_df.loc[:, "cor_acc_value"].to_numpy()[to_be_used_index]
        correct_ratio_list.append(correct.sum() / to_be_used_index.shape[1])

    results[key] = [[eval('%.3g'%(np.array(correct_ratio_list).mean() * 100)), eval('%.3g'%(np.array(correct_ratio_list).std() * 100))]]


for key, value in dataframes_set.items():

    correct_ratio_list = []
    for one_df in value:
        to_be_used = one_df.loc[:, "correct"].to_numpy()
        to_be_used_index = np.array(np.where(to_be_used==1))
        correct = one_df.loc[:, "best_fair_value"].to_numpy()[to_be_used_index]
        # print(correct)
        correct_ratio_list.append(correct.sum() / to_be_used_index.shape[1])

    results[key].append([eval('%.3g'%(np.array(correct_ratio_list).mean() * 100)), eval('%.3g'%(np.array(correct_ratio_list).std() * 100))])




for key, value in dataframes_set.items():

    correct_ratio_list = []
    for one_df in value:
        correct = one_df.loc[:, "correct"].to_numpy()
        if correct.shape[0] == 100:
            correct_ratio_list.append(correct.sum() / 100.0)
        else:
            correct_ratio_list.append(-1e3)
        
    results[key].append([eval('%.3g'%(np.array(correct_ratio_list).mean() * 100)), eval('%.3g'%(np.array(correct_ratio_list).std() * 100))])


for key, value in results.items():
    print(key, value)




