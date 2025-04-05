import pandas as pd
import numpy as np
import csv
import platform
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

def get_code_data(Files, degree):
    actual_file = 0
    usable_files = 0
    size_s = 0
    size_m = 0
    size_l = 0
    result = []
    for file in Files:
        try:
            _df = []
            actual_file += 1
            df = pd.read_pickle(file)
            buggy_commit_data_df = df.loc[0,'buggy_commit_data_df']
            commit_data_df = df.loc[0,'commit_data_df']
            commit_count = df.loc[0,'commit_count']
            sg_data_df = df.loc[0,'sg_data_df']
            cg_data_df = df.loc[0,'cg_data_df']
            if commit_data_df.shape[0] > 30:
                size_l += 1
            elif commit_data_df.shape[0] > 15:
                size_m += 1
            elif commit_data_df.shape[0] > 8:
                size_s += 1
            if commit_data_df.shape[0] < 8:
                continue
            for i in range(cg_data_df.shape[0]):
                buggy_commit_count = buggy_commit_data_df[buggy_commit_data_df['committer'] == cg_data_df.loc[i,'committer']]['count']
                if len(buggy_commit_count) == 0:
                    continue
                commit_count = commit_data_df[commit_data_df['committer'] == cg_data_df.loc[i,'committer']]['count']
                if len(commit_count) == 0:
                    continue
                node_degree = cg_data_df[cg_data_df['committer'] == cg_data_df.loc[i,'committer']]['count']
                #print([buggy_commit_count.values[0],commit_count.values[0],node_degree.values[0]])
                _df.append([buggy_commit_count.values[0]/commit_count.values[0],node_degree.values[0]])
            df = pd.DataFrame(_df, columns = ['per','degree'])
            if(df.shape[0] <= 0):
                continue
            usable_files += 1
            degree_d = np.array(df['degree'].values.tolist())
            forth = np.int32(np.percentile(degree_d,degree))
            first_l = []
            second_l = []
            third_l = []
            forth_l = []
            for i in range(df.shape[0]):
                if df.loc[i,'degree'] < forth:
                    forth_l.append(df.loc[i,'per'])
                else:
                    first_l.append(df.loc[i,'per'])
            #print(round(np.median(first_l),2),round(np.median(second_l),2),round(np.median(third_l),2),round(np.median(forth_l),2))
            result.append([file,commit_data_df.shape[0],round(np.median(first_l),2),round(np.median(forth_l),2)])
        except:
            continue
    print(actual_file,usable_files)
    print(size_s,size_m,size_l)
    c_result_df = pd.DataFrame(result, columns = ['project','c_size','c_experienced','c_inexperienced'])
    c_result_df_n = c_result_df.set_index('project')
    c_result_df.to_csv('results/Result_data_team_c_' + str(degree) + '.csv')
    return c_result_df_n


def get_social_data(Files,degree):
    actual_file = 0
    usable_files = 0
    size_s = 0
    size_m = 0
    size_l = 0
    result = []
    for file in Files:
        try:
            _df = []
            actual_file += 1
            df = pd.read_pickle(file)
            buggy_commit_data_df = df.loc[0,'buggy_commit_data_df']
            commit_data_df = df.loc[0,'commit_data_df']
            commit_count = df.loc[0,'commit_count']
            sg_data_df = df.loc[0,'sg_data_df']
            cg_data_df = df.loc[0,'cg_data_df']
            if commit_data_df.shape[0] > 30:
                size_l += 1
            elif commit_data_df.shape[0] > 15:
                size_m += 1
            elif commit_data_df.shape[0] > 8:
                size_s += 1
            if commit_data_df.shape[0] < 8:
                continue
            for i in range(sg_data_df.shape[0]):
                buggy_commit_count = buggy_commit_data_df[buggy_commit_data_df['committer'] == sg_data_df.loc[i,'committer']]['count']
                if len(buggy_commit_count) == 0:
                    continue
                commit_count = commit_data_df[commit_data_df['committer'] == sg_data_df.loc[i,'committer']]['count']
                if len(commit_count) == 0:
                    continue
                node_degree = sg_data_df[sg_data_df['committer'] == sg_data_df.loc[i,'committer']]['count']
                #print([buggy_commit_count.values[0],commit_count.values[0],node_degree.values[0]])
                _df.append([buggy_commit_count.values[0]/commit_count.values[0],node_degree.values[0]])
            df = pd.DataFrame(_df, columns = ['per','degree'])
            if(df.shape[0] <= 0):
                continue
            usable_files += 1
            degree_d = np.array(df['degree'].values.tolist())
            forth = np.int32(np.percentile(degree_d,degree))
            first_l = []
            second_l = []
            third_l = []
            forth_l = []
            for i in range(df.shape[0]):
                if df.loc[i,'degree'] < forth:
                    forth_l.append(df.loc[i,'per'])
                else:
                    first_l.append(df.loc[i,'per'])
            #print(round(np.median(first_l),2),round(np.median(second_l),2),round(np.median(third_l),2),round(np.median(forth_l),2))
            result.append([file,commit_data_df.shape[0],round(np.median(first_l),2),round(np.median(forth_l),2)])
        except Exception as e:
            print(e)
            continue
    print(actual_file,usable_files)
    print(size_s,size_m,size_l)
    s_result_df = pd.DataFrame(result, columns = ['project','s_size','s_experienced','s_inexperienced'])
    s_result_df_n = s_result_df.set_index('project')
    s_result_df_n.to_csv('results/Result_data_team_s_' + str(degree) + '.csv')
    return s_result_df_n

if __name__ == "__main__":
    if platform.system() == 'Darwin' or platform.system() == 'Linux':
        _dir = 'Processed_data/'
    else:
        _dir = 'Processed Data\\'
    Files = [join(_dir, f) for f in listdir(_dir) if isfile(join(_dir, f))]

    degrees = [80,85,90,95]

    joined_df = pd.DataFrame()

    for degree in degrees:
        c_result_df_n = get_code_data(Files, degree)
        s_result_df_n = get_social_data(Files, degree)
        joined_df = pd.concat([c_result_df_n,s_result_df_n],join='inner',axis = 1)

        joined_df.to_csv('results/Result_data_team_joined_' + str(degree) + '.csv')