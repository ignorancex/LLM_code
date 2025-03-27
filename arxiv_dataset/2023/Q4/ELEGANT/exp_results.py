import os
import fnmatch
import pandas as pd
import numpy as np
# from certify_K import certify_K
from certify_K_fast import certify_K
from operator import itemgetter
import glob



results = {}

dataframes_set = {}

directory_path = './exps_results/'
txt_files = glob.glob(os.path.join(directory_path, '*.txt'))
latest_file = max(txt_files, key=os.path.getmtime)
df = pd.read_csv(latest_file, sep='\t')

gnn = os.path.basename(latest_file).split("_")[1]
dataset = os.path.basename(latest_file).split("_")[2]



dataframes_set[(dataset, gnn)] = [df]


print((dataset, gnn))




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
        correct_ratio_list.append(correct.sum() / to_be_used_index.shape[1])

    results[key].append([eval('%.3g'%(np.array(correct_ratio_list).mean() * 100)), eval('%.3g'%(np.array(correct_ratio_list).std() * 100))])




for key, value in dataframes_set.items():

    correct_ratio_list = []
    for one_df in value:
        correct = one_df.loc[:, "correct"].to_numpy()
        print(key)
        print(correct.sum() / 100.0)
        if correct.shape[0] == 100:
            correct_ratio_list.append(correct.sum() / 100.0)
        else:
            correct_ratio_list.append(-1e3)
        
    results[key].append([eval('%.3g'%(np.array(correct_ratio_list).mean() * 100)), eval('%.3g'%(np.array(correct_ratio_list).std() * 100))])


for key, value in results.items():
    print(key, value)
