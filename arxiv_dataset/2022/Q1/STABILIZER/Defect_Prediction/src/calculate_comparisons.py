import pandas as pd
import numpy as np
import math
import pickle
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

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



import matplotlib.pyplot as plt

import SMOTE
import CFS
import birch


from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue

import metrics

import sys
import traceback
import warnings
warnings.filterwarnings("ignore")

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



class Bellwether_Method(object):

    def __init__(self, data_path, attr_df):
        # print(attr_df.shape)
        self.attr_df = attr_df
        self.cores = 50
        
    def prepare_data(self, project, metric):
        data_path = 'data/700/merged_data/' + project + '.csv'
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


    def apply_cfs(self,df):
        y = df.Bugs.values
        X = df.drop(labels = ['Bugs'],axis = 1)
        X = X.values
        selected_cols = CFS.cfs(X,y)
        cols = df.columns[[selected_cols]].tolist()
        cols.append('Bugs')
        return df[cols],cols
        
    def apply_smote(self,df):
        cols = df.columns
        smt = SMOTE.smote(df)
        df = smt.run()
        df.columns = cols
        return df


    # Cluster Driver
    def cluster_driver(self,df,print_tree = False):
        X = df.apply(pd.to_numeric)
        cluster = birch.birch(branching_factor=20)
        #X.set_index('Project Name',inplace=True)
        # print(X)
        cluster.fit(X)
        cluster_tree,max_depth = cluster.get_cluster_tree()
        #cluster_tree = cluster.model_adder(cluster_tree)
        if print_tree:
            cluster.show_clutser_tree()
        return cluster,cluster_tree,max_depth

    def build_BIRCH(self):
        cluster,cluster_tree,max_depth = self.cluster_driver(self.attr_df)
        return cluster,cluster_tree,max_depth


    def bellwether(self, selected_projects, all_projects, metric):
        # print('starting bellwether')
        with open('src/results/attributes/projects_attributes_all.pkl', 'rb') as handle:
            cfs_data = pickle.load(handle)
        cfs_data_df = pd.DataFrame.from_dict(cfs_data, orient= 'index')
        final_score = {}
        count = 0
        for s_project in selected_projects:
            try:
                df = self.prepare_data(s_project, metric)
                cols = df.columns
                if df.shape[0] < 50:
                    continue
                else:
                    count+=1
                df.reset_index(drop=True,inplace=True)
                s_cols = []
                for i  in range(cfs_data_df.loc[s_project].shape[0]):
                    col = cfs_data_df.loc[s_project].values.tolist()[i]
                    if col == 1:
                        s_cols.append(cols[i])
                # print(s_cols)
                df = df[s_cols]
                # df, s_cols = self.apply_cfs(df)
                #s_cols = df.columns.tolist()
                df = self.apply_smote(df)
                y = df.Bugs
                X = df.drop(labels = ['Bugs'],axis = 1)
                # kf = StratifiedKFold(n_splits = 5)
                score = {}
                F = {}
                for i in range(1):
                    clf = RandomForestClassifier()
                    clf.fit(X,y)
                    destination_projects = copy.deepcopy(all_projects)
                    for d_project in destination_projects:
                        # print(d_project)
                        try:
                            _test_df = self.prepare_data(d_project, metric)
                            _df_test_loc = _test_df['file_la'] + _test_df['file_lt']
                            test_df = _test_df[s_cols]
                            # print(test_df.shape)
                            if test_df.shape[0] < 50:
                                continue
                            test_df.reset_index(drop=True,inplace=True)
                            test_y = test_df.Bugs
                            test_X = test_df.drop(labels = ['Bugs'],axis = 1)
                            predicted = clf.predict(test_X)
                            abcd = metrics.measures(test_y,predicted,_df_test_loc)
                            F['f1'] = [abcd.calculate_f1_score()]
                            F['precision'] = [abcd.calculate_precision()]
                            F['recall'] = [abcd.calculate_recall()]
                            F['g-score'] = [abcd.get_g_score()]
                            F['d2h'] = [abcd.calculate_d2h()]
                            F['pci_20'] = [abcd.get_pci_20()]
                            F['ifa'] = [abcd.get_ifa()]
                            F['pd'] = [abcd.get_pd()]
                            F['pf'] = [abcd.get_pf()]
                            _F = copy.deepcopy(F)
                            if 'f1' not in score.keys():
                                score[d_project] = _F
                            else:
                                score[d_project]['f1'].append(F['f1'][0])
                                score[d_project]['precision'].append(F['precision'][0])
                                score[d_project]['recall'].append(F['recall'][0])
                                score[d_project]['g-score'].append(F['g-score'][0])
                                score[d_project]['d2h'].append(F['d2h'][0])
                                score[d_project]['pci_20'].append(F['pci_20'][0])
                                score[d_project]['ifa'].append(F['ifa'][0])
                                score[d_project]['pd'].append(F['pd'][0])
                                score[d_project]['pf'].append(F['pf'][0])
                        except Exception as e:
                            # print("dest",d_project,e)
                            continue
                    final_score[s_project] = score 
            except Exception as e:
                # print("src",s_project,e)
                continue
        return final_score

    # def run_bellwether(self,projects):
    #     threads = []
    #     results = {}
    #     _projects = projects
    #     split_projects = np.array_split(_projects, self.cores)
    #     for i in range(self.cores):
    #         print("starting thread ",i)
    #         t = ThreadWithReturnValue(target = self.bellwether, args = [split_projects[i],projects])
    #         threads.append(t)
    #     for th in threads:
    #         th.start()
    #     for th in threads:
    #         response = th.join()
    #         results.update(response)
    #     return results

    def run(self, selected_projects, cluster_id, data_store_path, metric):
        # print(cluster_id,'running bellwether')
        final_score = self.bellwether(selected_projects, selected_projects, metric)
        data_path = Path(data_store_path + str(cluster_id))
        if not data_path.is_dir():
            os.makedirs(data_path)
        with open(data_store_path + str(cluster_id)  + '/700_RF_default_bellwether.pkl', 'wb') as handle:
            pickle.dump(final_score, handle, protocol=pickle.HIGHEST_PROTOCOL)
        df = pd.read_pickle(data_store_path + str(cluster_id)  + '/700_RF_default_bellwether.pkl')
        results_f1 = {}
        results_precision = {}
        results_recall = {}
        results_g = {}
        results_d2h = {}
        results_pci_20 = {}
        results_ifa = {}
        results_pd = {}
        results_pf = {}
        for s_project in df.keys():
            if s_project not in results_f1.keys():
                results_f1[s_project] = {}
                results_precision[s_project] = {}
                results_recall[s_project] = {}
                results_g[s_project] = {}
                results_d2h[s_project] = {}
                results_pci_20[s_project] = {}
                results_ifa[s_project] = {}
                results_pd[s_project] = {}
                results_pf[s_project] = {}
            for d_projects in df[s_project].keys():
                results_f1[s_project][d_projects] = np.median(df[s_project][d_projects]['f1'])
                results_precision[s_project][d_projects] = np.median(df[s_project][d_projects]['precision'])
                results_recall[s_project][d_projects] = np.median(df[s_project][d_projects]['recall'])
                results_g[s_project][d_projects] = np.median(df[s_project][d_projects]['g-score'])
                results_d2h[s_project][d_projects] = np.median(df[s_project][d_projects]['d2h'])
                results_pci_20[s_project][d_projects] = np.median(df[s_project][d_projects]['pci_20'])
                results_ifa[s_project][d_projects] = np.median(df[s_project][d_projects]['ifa'])
                results_pd[s_project][d_projects] = np.median(df[s_project][d_projects]['pd'])
                results_pf[s_project][d_projects] = np.median(df[s_project][d_projects]['pf'])
        results_f1_df = pd.DataFrame.from_dict(results_f1, orient='index')
        results_precision_df = pd.DataFrame.from_dict(results_precision, orient='index')
        results_recall_df = pd.DataFrame.from_dict(results_recall, orient='index')
        results_g_df = pd.DataFrame.from_dict(results_g, orient='index')
        results_d2h_df = pd.DataFrame.from_dict(results_d2h, orient='index')
        results_pci_20_df = pd.DataFrame.from_dict(results_pci_20, orient='index')
        results_ifa_df = pd.DataFrame.from_dict(results_ifa, orient='index')
        results_pd_df = pd.DataFrame.from_dict(results_pd, orient='index')
        results_pf_df = pd.DataFrame.from_dict(results_pf, orient='index')
        
        results_f1_df.to_csv(data_store_path + str(cluster_id)  + '/700_RF_bellwether_f1.csv')
        results_precision_df.to_csv(data_store_path + str(cluster_id)  + '/700_RF_bellwether_precision.csv')
        results_recall_df.to_csv(data_store_path + str(cluster_id)  + '/700_RF_bellwether_recall.csv')
        results_g_df.to_csv(data_store_path + str(cluster_id)  + '/700_RF_bellwether_g.csv')
        results_d2h_df.to_csv(data_store_path + str(cluster_id)  + '/700_RF_bellwether_d2h.csv')
        results_pci_20_df.to_csv(data_store_path + str(cluster_id)  + '/700_RF_bellwether_pci_20.csv')
        results_ifa_df.to_csv(data_store_path + str(cluster_id)  + '/700_RF_bellwether_ifa.csv')
        results_pd_df.to_csv(data_store_path+ str(cluster_id)  + '/700_RF_bellwether_pd.csv')
        results_pf_df.to_csv(data_store_path + str(cluster_id)  + '/700_RF_bellwether_pf.csv')

        return None


if __name__ == "__main__":
    start = timeit.default_timer()
    limit = sys.argv[1]
    path = 'data/merged_data'
    metric = 'all'
    meta_path = 'src/results/attributes/projects_median.pkl'
    _data_store_path = 'src/results/median_data/level_2/'
    attr_dict = pd.read_pickle(meta_path)
    attr_df = attr_dict
    # attr_df = pd.DataFrame.from_dict(attr_dict,orient='index')
    attr_df.dropna(inplace = True)
    attr_df_index = list(attr_df.index)
    kf = KFold(n_splits=10)
    i = 0
    total_coms = []
    num_clusters = []
    for train_index, test_index in kf.split(attr_df):
        data_store_path = _data_store_path
        _train_index = []
        _test_index = []
        for index in train_index:
            _train_index.append(attr_df_index[index])
        for index in test_index:
            _test_index.append(attr_df_index[index])
        data_store_path = data_store_path + 'fold_' + str(i) + '/'
        i += 1
        _attr_df_train = attr_df.loc[_train_index]

        # print(_attr_df_train.shape)
        # _sample_projects = random.sample(list(_attr_df_train.index), 720)

        # _attr_df_train = _attr_df_train.loc[_sample_projects]

        _attr_df_test = attr_df.loc[_test_index]

        data_path = Path(data_store_path)
        if not data_path.is_dir():
            os.makedirs(data_path)

        _attr_df_train = _attr_df_train[0:int(limit)]

        bell = Bellwether_Method(path,_attr_df_train)
        cluster,cluster_tree,max_depth = bell.build_BIRCH()

        total_com = 0
        clusters = {}
        for dep in range(max_depth+1):
            cluster_ids = []
            if dep-1 not in clusters.keys():
                clusters[dep-1] = {}
            for key in cluster_tree:
                if cluster_tree[key].depth == dep:
                    cluster_ids.append(key)
            if dep == max_depth:
                # print(len(cluster_ids))
                for ids in cluster_ids:
                    # print("ids",ids)
                    selected_projects = list(_attr_df_train.loc[cluster_tree[ids].data_points].index)
                    total_com += len(selected_projects)**2
                    if cluster_tree[ids].parent_id not in clusters[dep-1].keys():
                        clusters[dep-1][cluster_tree[ids].parent_id] = []
                    clusters[dep-1][cluster_tree[ids].parent_id].append(ids)
                num_clusters.append(len(cluster_ids))
            else:
                for ids in cluster_ids:
                    if cluster_tree[ids].parent_id not in clusters[dep-1].keys():
                        clusters[dep-1][cluster_tree[ids].parent_id] = []
                    clusters[dep-1][cluster_tree[ids].parent_id].append(ids)

        for dep in clusters.keys():
            if dep == -1:
                continue
            for ids in clusters[dep].keys():
                total_com += len(clusters[dep][ids])**2
        # print(total_com)
        total_coms.append(total_com)
    print("Average Num of Comparisons:", np.median(total_coms))
    print("Average Num of Clusters:", np.median(num_clusters))
    
