import pandas as pd
import CFS
import numpy as np
import math
import pickle

import sys
import traceback
import warnings
import os
import copy
import platform
from os import listdir
from os.path import isfile, join
from glob import glob
from pathlib import Path

import threading
from threading import Thread
from threading import Barrier
from multiprocessing import Queue
from multiprocessing import Pool, cpu_count

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def prepare_data(project, metric):
    data_path = '../data/700/merged_data/' + project + '.csv'
    data_df = pd.read_csv(data_path)
    # data_df.rename(columns = {'Unnamed: 0':'id'},inplace = True)
    # data_df = data_df.drop(labels = ['id'],axis=1)

    for col in ['Unnamed: 0', 'commit_hash', 'release']:
        if col in data_df.columns:
            data_df = data_df.drop([col], axis = 1)
    y = data_df.Bugs
    X = data_df.drop(['Bugs'],axis = 1)
    
    if metric == 'process':
            X = X[['file_la', 'file_ld', 'file_lt', 'file_age', 'file_ddev',
        'file_nuc', 'own', 'minor', 'file_ndev', 'file_ncomm', 'file_adev',
        'file_nadev', 'file_avg_nddev', 'file_avg_nadev', 'file_avg_ncomm',
        'file_ns', 'file_exp', 'file_sexp', 'file_rexp', 'file_nd', 'file_sctr']]
    elif metric == 'product':
            X = X.drop(['file_la', 'file_ld', 'file_lt', 'file_age', 'file_ddev',
        'file_nuc', 'own', 'minor', 'file_ndev', 'file_ncomm', 'file_adev',
        'file_nadev', 'file_avg_nddev', 'file_avg_nadev', 'file_avg_ncomm',
        'file_ns', 'file_exp', 'file_sexp', 'file_rexp', 'file_nd', 'file_sctr'],axis = 1)
    else:
        X = X
    df = X
    df['Bugs'] = y
    return df

def apply_cfs(df):
    _cols = df.columns
    y = df.Bugs.values
    X = df.drop(labels = ['Bugs'],axis = 1)
    X = X.values
    selected_cols = CFS.cfs(X,y)
    fss = []
    cols = df.columns[[selected_cols]].tolist()
    cols.append('Bugs')
    for col in _cols:
        if col in cols:
            fss.append(1)
        else:
            fss.append(0)
    return df[cols],cols,fss

def run(projects, metric):
    count = 0
    project_selection = {}
    for project in projects:
        try:
            project_attr = []
            df = prepare_data(project, metric)
            if df.shape[0] < 50:
                continue
            else:
                count+=1
            for repeat in range(1):
                _df,cols,fss = apply_cfs(df)
                project_attr.append(fss)
            project_attr = np.array(list(map(sum,zip(*project_attr))))/len(project_attr)
            print(project_attr)
            project_attr = [round(x) for x in project_attr]
            project_selection[project] = project_attr      
        except Exception as e:
            print(e)
            continue
    return project_selection



if __name__ == "__main__":
    proj_df = pd.read_csv('projects.csv')
    metric = 'all'
    project_list = proj_df.repo_name.tolist()[151:]
    cores = 50
    threads = []
    final_results = {}
    projects = np.array_split(project_list, cores)
    for i in range(len(projects)):
        t = ThreadWithReturnValue(target = run, args = [projects[i], metric])
        threads.append(t)
        print("Starting thread: ",i)
    for th in threads:
        th.start()
    for th in threads:
        response = th.join()
        final_results = {**response, **final_results}
    print(final_results)

    with open('results/attributes/projects_attributes_all.pkl', 'wb') as handle:
        pickle.dump(final_results, handle, protocol=pickle.HIGHEST_PROTOCOL)