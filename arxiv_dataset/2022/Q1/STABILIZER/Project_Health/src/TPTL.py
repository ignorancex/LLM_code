import os
from os import listdir
from os.path import isfile, join
import sys
import pandas as pd
import random
import numpy as np
import copy


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue

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
    data_path = '../data/700/merged_data_original/' + project + '.csv'
    data_df = pd.read_csv(data_path)
    data_df.rename(columns = {'Unnamed: 0':'id'},inplace = True)

    for col in ['id', 'commit_hash', 'release']:
        if col in data_df.columns:
            data_df = data_df.drop([col], axis = 1)
    data_df = data_df.dropna()
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



def create_models(metric, fold):
    meta_data = pd.read_pickle('results/TCA/TCA_all/tca.pkl')
    source_projects = meta_data.source.unique()
    pseudo_source_projects = meta_data.target.unique()

    cluster_data_loc = 'results/median_data/level_2/fold_' + str(fold)
    test_data = pd.read_pickle(cluster_data_loc + '/test_data.pkl')
    target_projects = test_data.index.values.tolist()

    selected_src_projects = copy.deepcopy(pseudo_source_projects)
    selected_src_projects = set(selected_src_projects)
    selected_src_projects = list(selected_src_projects - set(target_projects))

    data_vectors = {}
    for src in pseudo_source_projects:
        print(src)
        src_data = prepare_data(src, metric)
        src_data_vector = src_data.median().values.tolist()[:-1]
        data_vectors[src] = src_data_vector

    train_X = []
    for i in range(meta_data.shape[0]):
        if i%1000 == 0:
            print(i)
        src = meta_data.iloc[i,0]
        trg = meta_data.iloc[i,1]
        f = meta_data.iloc[i,5]
        pci_20 = meta_data.iloc[i,6]
        src_data_vector = data_vectors[src]
        trg_data_vector = data_vectors[trg]
        train_X.append(trg_data_vector + src_data_vector)
        

    train_y_f = meta_data.f.values.tolist()
    train_y_p = meta_data.pci_20.values.tolist()


    clf_f = SVR()
    clf_p = SVR()

    clf_f.fit(train_X,train_y_f)
    clf_p.fit(train_X,train_y_p)


    test_X = []
    test_map = []
    for sp in pseudo_source_projects:
        src_data = prepare_data(sp, metric)
        src_data_vector = src_data.median().values.tolist()[:-1]
        for tp in target_projects:
            trg_data = prepare_data(tp, metric)
            trg_data_vector = trg_data.median().values.tolist()[:-1]
            test_X.append(trg_data_vector + src_data_vector)
            test_map.append([sp, tp])
    
    predicted_f = clf_f.predict(test_X)
    predicted_p = clf_p.predict(test_X)


    for i in range(len(predicted_f)):
        test_map[i].append(round(predicted_f[i],2))
        test_map[i].append(round(predicted_p[i],2))

    final_result = test_map
    final_result_df = pd.DataFrame(final_result, columns=['src','trg', 'f', 'pci_20'])
    final_result_df.to_csv('results/mixed_data/level_2/fold_' + str(fold) + '/predicted_source_process.csv')
    return None


if __name__ == "__main__":
    threads = []
    for i in range(10):
        metric = 'process'
        t = ThreadWithReturnValue(target = create_models, args = [metric, i])
        threads.append(t)
    for th in threads:
        th.start()
    for th in threads:
        response = th.join()