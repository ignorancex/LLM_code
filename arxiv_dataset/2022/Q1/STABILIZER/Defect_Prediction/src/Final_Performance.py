import pandas as pd
import numpy as np
import math
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import random

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

def prepare_data(project, metric):
    data_path = '../data/700/merged_data/' + project + '.csv'
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

def apply_cfs(df):
    y = df.Bugs.values
    X = df.drop(labels = ['Bugs'],axis = 1)
    X = X.values
    selected_cols = CFS.cfs(X,y)
    cols = df.columns[[selected_cols]].tolist()
    cols.append('Bugs')
    return df[cols],cols

    
def apply_smote(df):
    cols = df.columns
    smt = SMOTE.smote(df)
    df = smt.run()
    df.columns = cols
    return df

def cluster_driver(df,print_tree = True):
    X = df.apply(pd.to_numeric)
    cluster = birch.birch(branching_factor=20)
    #X.set_index('Project Name',inplace=True)
    cluster.fit(X)
    cluster_tree,max_depth = cluster.get_cluster_tree()
    #cluster_tree = cluster.model_adder(cluster_tree)
    if print_tree:
        cluster.show_clutser_tree()
    return cluster,cluster_tree,max_depth


def get_predicted_1(cluster_data_loc,fold,data_location,default_bellwether_loc,depth, metric):
    with open('results/attributes/projects_attributes.pkl', 'rb') as handle:
        cfs_data = pickle.load(handle)
    cfs_data_df = pd.DataFrame.from_dict(cfs_data, orient= 'index')
    train_data = pd.read_pickle(cluster_data_loc + '/train_data.pkl')
    cluster,cluster_tree,max_depth = cluster_driver(train_data)
    test_data = pd.read_pickle(cluster_data_loc + '/test_data.pkl')
    test_projects = test_data.index.values.tolist()
    goals = ['recall','precision','pf','pci_20','ifa']
    results = {}
    bellwether_models = {}
    bellwether0_models = {}
    bellwether0_s_cols = {}
    bellwether_s_cols = {}
    self_model = {}
    self_model_test = {}
    test_data = test_data
    predicted_cluster = cluster.predict(test_data,depth)
    s_project_df = pd.read_csv(cluster_data_loc + '/bellwether_cdom_' + str(depth) + '.csv')
    if depth != 2:
        s_project_df.rename(columns = {'Unnamed: 0':'id'},inplace = True)
    for i in range(len(predicted_cluster)):
        F = {}
        F_bell = {}
        c_id = predicted_cluster[i]
        if depth != 0:
            cluster_bellwether = s_project_df[s_project_df['id'] == predicted_cluster[i]].bellwether.values[0]
        else:
            cluster_bellwether = s_project_df.bellwether.values
        d_project = test_projects[i]
        kf = StratifiedKFold(n_splits = 3)
        test_df = prepare_data(d_project, metric)
        test_df.reset_index(drop=True,inplace=True)
        test_y = test_df.Bugs
        test_X = test_df.drop(labels = ['Bugs'],axis = 1)
        _df_test_loc = test_df['file_la'] + test_df['file_lt']
        s_projects = s_project_df.bellwether.values.tolist()
        if depth != 0:
            s_projects.remove(cluster_bellwether)
            s_project_lists = [s_projects[random.randint(0,len(s_projects)-1)]]
            s_project_lists.append(cluster_bellwether)
        else:
            s_project_lists = cluster_bellwether
        for s_project in s_project_lists:
            if s_project not in bellwether_models.keys():
                df = prepare_data(s_project, metric)
                cols = df.columns
                df.reset_index(drop=True,inplace=True)
                s_cols = []
                for i  in range(cfs_data_df.loc[s_project].shape[0]):
                    col = cfs_data_df.loc[s_project].values.tolist()[i]
                    if col == 1:
                        s_cols.append(cols[i])
                df = df[s_cols]
                bellwether_s_cols[s_project] = s_cols
                df = apply_smote(df)
                y = df.Bugs
                X = df.drop(labels = ['Bugs'],axis = 1)
                clf_bellwether = RandomForestClassifier()
                clf_bellwether.fit(X,y)
                bellwether_models[s_project] = clf_bellwether
            else:
                clf_bellwether = bellwether_models[s_project]
                s_cols = bellwether_s_cols[s_project]

            for train_index, test_index in kf.split(test_X,test_y):
                X_train, X_test = test_X.iloc[train_index], test_X.iloc[test_index]
                y_train, y_test = test_y[train_index], test_y[test_index]
                if s_project !=  cluster_bellwether:
                    try:
                        df_test_bell = copy.deepcopy(test_df[s_cols])
                        y_test = df_test_bell.Bugs
                        X_test = df_test_bell.drop(labels = ['Bugs'],axis = 1)
                        predicted_self = clf_bellwether.predict(X_test) 
                        abcd = metrics.measures(y_test,predicted_self,_df_test_loc)
                        if 'f1' not in F.keys():
                            F['f1'] = []
                            F['precision'] = []
                            F['recall'] = []
                            F['g-score'] = []
                            F['d2h'] = []
                            F['pci_20'] = []
                            F['ifa'] = []
                            F['pd'] = []
                            F['pf'] = []
                        F['f1'].append(abcd.calculate_f1_score())
                        F['precision'].append(abcd.calculate_precision())
                        F['recall'].append(abcd.calculate_recall())
                        F['g-score'].append(abcd.get_g_score())
                        F['d2h'].append(abcd.calculate_d2h())
                        F['pci_20'].append(abcd.get_pci_20())
                        try:
                            F['ifa'].append(abcd.get_ifa_roc())
                        except:
                            F['ifa'].append(0)
                        F['pd'].append(abcd.get_pd())
                        F['pf'].append(abcd.get_pf())
                    except:
                        F['f1'].append(0)
                        F['precision'].append(0)
                        F['recall'].append(0)
                        F['g-score'].append(0)
                        F['d2h'].append(0)
                        F['pci_20'].append(0)
                        F['ifa'].append(0)
                        F['pd'].append(0)
                        F['pf'].append(0)
                else:
                    try:
                        df_test_bell = copy.deepcopy(test_df[s_cols])
                        y_test = df_test_bell.Bugs
                        X_test = df_test_bell.drop(labels = ['Bugs'],axis = 1)
                        predicted_self = clf_bellwether.predict(X_test) 
                        abcd = metrics.measures(y_test,predicted_self,_df_test_loc)
                        if 'f1' not in F_bell.keys():
                            F_bell['f1'] = []
                            F_bell['precision'] = []
                            F_bell['recall'] = []
                            F_bell['g-score'] = []
                            F_bell['d2h'] = []
                            F_bell['pci_20'] = []
                            F_bell['ifa'] = []
                            F_bell['pd'] = []
                            F_bell['pf'] = []
                        F_bell['f1'].append(abcd.calculate_f1_score())
                        F_bell['precision'].append(abcd.calculate_precision())
                        F_bell['recall'].append(abcd.calculate_recall())
                        F_bell['g-score'].append(abcd.get_g_score())
                        F_bell['d2h'].append(abcd.calculate_d2h())
                        F_bell['pci_20'].append(abcd.get_pci_20())
                        try:
                            F_bell['ifa'].append(abcd.get_ifa_roc())
                        except:
                            F_bell['ifa'].append(0)
                        F_bell['pd'].append(abcd.get_pd())
                        F_bell['pf'].append(abcd.get_pf())
                    except Exception as e:
                        print(e)
                        F_bell['f1'].append(0)
                        F_bell['precision'].append(0)
                        F_bell['recall'].append(0)
                        F_bell['g-score'].append(0)
                        F_bell['d2h'].append(0)
                        F_bell['pci_20'].append(0)
                        F_bell['ifa'].append(0)
                        F_bell['pd'].append(0)
                        F_bell['pf'].append(0)
        for goal in goals:
            if goal == 'g':
                _goal = 'g-score'
            else:
                _goal = goal
            if _goal not in results.keys():
                results[_goal] = {}
            if d_project not in results[_goal].keys():
                results[_goal][d_project] = []
            results[_goal][d_project].append(np.median(F[_goal]))
            results[_goal][d_project].append(np.median(F_bell[_goal]))
    return results
            
def get_predicted_all(cluster_data_loc,fold,data_location,default_bellwether_loc, metric):
    with open('results/attributes/projects_attributes.pkl', 'rb') as handle:
        cfs_data = pickle.load(handle)
    cfs_data_df = pd.DataFrame.from_dict(cfs_data, orient= 'index')
    train_data = pd.read_pickle(cluster_data_loc + '/train_data.pkl')
    cluster,cluster_tree,max_depth = cluster_driver(train_data)
    test_data = pd.read_pickle(cluster_data_loc + '/test_data.pkl')
    test_projects = test_data.index.values.tolist()
    goals = ['recall','precision','pf','pci_20','ifa','f1','g']
    results = {}
    GENERAL_0_models = {}
    GENERAL_1_models = {}
    GENERAL_2_models = {}
    bellwether_models = {}
    
    GENERAL_0_s_cols = {}
    GENERAL_1_s_cols = {}
    GENERAL_2_s_cols = {}
    bellwether_s_cols = {}
    self_model = {}
    self_model_test = {}
    
    # t_df = pd.DataFrame()
    # for project in train_data.index.values.tolist():
    #     s_df = prepare_data(project,metric)
    #     t_df = pd.concat([t_df,s_df])

    # t_df = pd.read_csv(cluster_data_loc + '/global_data.csv', index_col = 0)
    # t_df, global_cols = apply_cfs(t_df)  

    # t_df = apply_smote(t_df)

    t_df = pd.read_csv('results/median_data/level_2/fold_0/global_data_cfs_SMOTE.csv', index_col = 0)
    global_cols = t_df.columns.values.tolist()

    train_y = t_df.Bugs
    train_X = t_df.drop(labels = ['Bugs'],axis = 1)
    clf_global = RandomForestClassifier()
    clf_global.fit(train_X,train_y)
    
    
    for t_project in test_data.index: 
        test_project = test_data.loc[[t_project,t_project]]
        predicted_cluster_0 = cluster.predict(test_project,0)
        predicted_cluster_1 = cluster.predict(test_project,1)
        predicted_cluster_2 = cluster.predict(test_project,2)
        
        s_project_df_0 = pd.read_csv(cluster_data_loc + '/bellwether_cdom_' + str(0) + '.csv')
        s_project_df_1 = pd.read_csv(cluster_data_loc + '/bellwether_cdom_' + str(1) + '.csv')
        s_project_df_2 = pd.read_csv(cluster_data_loc + '/bellwether_cdom_' + str(2) + '.csv')
        bellwether_df = pd.read_csv(default_bellwether_loc + '/cdom_latest.csv', index_col = 0)
        
        
        s_project_df_0.rename(columns = {'Unnamed: 0':'id'},inplace = True)
        s_project_df_1.rename(columns = {'Unnamed: 0':'id'},inplace = True)
        
        self = {}
        general_0 = {}
        general_1 = {}
        general_2 = {}
        _global = {}
        random = {}
        bellwether = {}
        
        s_project_0 = s_project_df_0.bellwether.values[0]
        s_project_1 = s_project_df_1[s_project_df_1['id'] == predicted_cluster_1[0]].bellwether.values[0]
        s_project_2 = s_project_df_2[s_project_df_2['id'] == predicted_cluster_2[0]].bellwether.values[0]
        s_project_bell = bellwether_df.wins.idxmax()
        
        # Level 0
        if s_project_0 not in GENERAL_0_models.keys():
            df = prepare_data(s_project_0, metric)
            cols = df.columns
            df.reset_index(drop=True,inplace=True)
            s_cols = []
            for i  in range(cfs_data_df.loc[s_project_0].shape[0]):
                col = cfs_data_df.loc[s_project_0].values.tolist()[i]
                if col == 1:
                    s_cols.append(cols[i])
            GENERAL_0_cols = s_cols
            df = df[s_cols]
            GENERAL_0_s_cols[s_project_0] = s_cols
            df = apply_smote(df)
            y = df.Bugs
            X = df.drop(labels = ['Bugs'],axis = 1)
            clf_GENERAL_0 = RandomForestClassifier()
            clf_GENERAL_0.fit(X,y)
            GENERAL_0_models[s_project_0] = clf_GENERAL_0
        else:
            clf_GENERAL_0 = GENERAL_0_models[s_project_0]
            GENERAL_0_cols = GENERAL_0_s_cols[s_project_0]
        
        # Level 1
        if s_project_1 not in GENERAL_1_models.keys():
            df = prepare_data(s_project_1, metric)
            cols = df.columns
            df.reset_index(drop=True,inplace=True)
            s_cols = []
            for i  in range(cfs_data_df.loc[s_project_1].shape[0]):
                col = cfs_data_df.loc[s_project_1].values.tolist()[i]
                if col == 1:
                    s_cols.append(cols[i])
            GENERAL_1_cols = s_cols
            df = df[s_cols]
            GENERAL_1_s_cols[s_project_1] = s_cols
            df = apply_smote(df)
            y = df.Bugs
            X = df.drop(labels = ['Bugs'],axis = 1)
            clf_GENERAL_1 = RandomForestClassifier()
            clf_GENERAL_1.fit(X,y)
            GENERAL_1_models[s_project_1] = clf_GENERAL_1
        else:
            clf_GENERAL_1 = GENERAL_1_models[s_project_1]
            GENERAL_1_cols = GENERAL_1_s_cols[s_project_1]
        
        # Level 2
        if s_project_2 not in GENERAL_2_models.keys():
            df = prepare_data(s_project_2, metric)
            cols = df.columns
            df.reset_index(drop=True,inplace=True)
            s_cols = []
            for i  in range(cfs_data_df.loc[s_project_2].shape[0]):
                col = cfs_data_df.loc[s_project_2].values.tolist()[i]
                if col == 1:
                    s_cols.append(cols[i])
            GENERAL_2_cols = s_cols
            df = df[s_cols]
            GENERAL_2_s_cols[s_project_2] = s_cols
            df = apply_smote(df)
            y = df.Bugs
            X = df.drop(labels = ['Bugs'],axis = 1)
            clf_GENERAL_2 = RandomForestClassifier()
            clf_GENERAL_2.fit(X,y)
            GENERAL_2_models[s_project_2] = clf_GENERAL_2
        else:
            clf_GENERAL_2 = GENERAL_2_models[s_project_2]
            GENERAL_2_cols = GENERAL_2_s_cols[s_project_2]
            
        # Bellwether
        if s_project_bell not in bellwether_models.keys():
            df = prepare_data(s_project_bell, metric)
            cols = df.columns
            df.reset_index(drop=True,inplace=True)
            s_cols = []
            for i  in range(cfs_data_df.loc[s_project_bell].shape[0]):
                col = cfs_data_df.loc[s_project_bell].values.tolist()[i]
                if col == 1:
                    s_cols.append(cols[i])
            bellwether_cols = s_cols
            df = df[s_cols]
            bellwether_s_cols[s_project_bell] = s_cols
            df = apply_smote(df)
            y = df.Bugs
            X = df.drop(labels = ['Bugs'],axis = 1)
            clf_bellwether = RandomForestClassifier()
            clf_bellwether.fit(X,y)
            bellwether_models[s_project_bell] = clf_bellwether
        else:
            clf_bellwether = bellwether_models[s_project_bell]
            bellwether_cols = bellwether_s_cols[s_project_bell]
        
        
        kf = StratifiedKFold(n_splits = 3)
        test_df = prepare_data(t_project, metric)
        test_df.reset_index(drop=True,inplace=True)
        test_y = test_df.Bugs
        test_X = test_df.drop(labels = ['Bugs'],axis = 1)
        
        all_model_results = {'GENERAL_0':{},
                         'GENERAL_1':{},
                         'GENERAL_2':{},
                         'bellwether':{},
                         'global':{},
                         'self':{},
                         'random':{}}
        for train_index, test_index in kf.split(test_X,test_y):
            X_train, X_test = test_X.iloc[train_index], test_X.iloc[test_index]
            y_train, y_test = test_y[train_index], test_y[test_index]
            target_train_df = pd.concat([X_train,y_train], axis = 1)
            target_test_df = pd.concat([X_test,y_test], axis = 1)
            _count_major = Counter(y_train)
            
            
            try:
                s_cols = []
                for i  in range(cfs_data_df.loc[t_project].shape[0]):
                    col = cfs_data_df.loc[t_project].values.tolist()[i]
                    if col == 1:
                        s_cols.append(cols[i])
            except:
                print('in else')
                try:
                    target_train_df, s_cols = apply_cfs(target_train_df)
                except:
                    s_cols = target_test_df.columns.tolist()
                    
            self_cols = s_cols
                    
            target_train_df = target_train_df[s_cols]
            target_train_df = apply_smote(target_train_df)
            
            y_train = target_train_df.Bugs
            X_train = target_train_df.drop(labels = ['Bugs'],axis = 1)
            clf_self = RandomForestClassifier()
            clf_self.fit(X_train,y_train)
            
            target_test_df_loc = target_test_df['file_la'] + target_test_df['file_lt']
            
            
            target_test_df_GENERAL_0 = copy.deepcopy(target_test_df[GENERAL_0_cols])
            target_test_df_GENERAL_1 = copy.deepcopy(target_test_df[GENERAL_1_cols])
            target_test_df_GENERAL_2 = copy.deepcopy(target_test_df[GENERAL_2_cols])
            target_test_df_bellwether = copy.deepcopy(target_test_df[bellwether_cols])
            target_test_df_global = copy.deepcopy(target_test_df[global_cols])
            target_test_df_self = copy.deepcopy(target_test_df[self_cols])
            target_test_df_random = copy.deepcopy(target_test_df)
            
            
            y_test = target_test_df_GENERAL_0.Bugs
            X_test = target_test_df_GENERAL_0.drop(labels = ['Bugs'],axis = 1)
            predicted = clf_GENERAL_0.predict(X_test)
            GENERAL_0_abcd = metrics.measures(y_test,predicted,target_test_df_loc)
            
            y_test = target_test_df_GENERAL_1.Bugs
            X_test = target_test_df_GENERAL_1.drop(labels = ['Bugs'],axis = 1)
            predicted = clf_GENERAL_1.predict(X_test)
            GENERAL_1_abcd = metrics.measures(y_test,predicted,target_test_df_loc)
            
            y_test = target_test_df_GENERAL_2.Bugs
            X_test = target_test_df_GENERAL_2.drop(labels = ['Bugs'],axis = 1)
            predicted = clf_GENERAL_2.predict(X_test)
            GENERAL_2_abcd = metrics.measures(y_test,predicted,target_test_df_loc)
            
            y_test = target_test_df_bellwether.Bugs
            X_test = target_test_df_bellwether.drop(labels = ['Bugs'],axis = 1)
            predicted = clf_bellwether.predict(X_test)
            bellwether_abcd = metrics.measures(y_test,predicted,target_test_df_loc)
            
            y_test = target_test_df_global.Bugs
            X_test = target_test_df_global.drop(labels = ['Bugs'],axis = 1)
            predicted = clf_global.predict(X_test)
            global_abcd = metrics.measures(y_test,predicted,target_test_df_loc)
            
            y_test = target_test_df_self.Bugs
            X_test = target_test_df_self.drop(labels = ['Bugs'],axis = 1)
            predicted = clf_self.predict(X_test)
            self_abcd = metrics.measures(y_test,predicted,target_test_df_loc)
            
            y_test = target_test_df_random.Bugs
            X_test = target_test_df_random.drop(labels = ['Bugs'],axis = 1)
            # _count_major = Counter(y_train)
            _count_major = _count_major.most_common(1)[0][0]
            predicted = [_count_major]*X_test.shape[0]
            random_abcd = metrics.measures(y_test,predicted,target_test_df_loc)
            
            all_abcds = {'GENERAL_0':GENERAL_0_abcd,
                         'GENERAL_1':GENERAL_1_abcd,
                         'GENERAL_2':GENERAL_2_abcd,
                         'bellwether':bellwether_abcd,
                         'global':global_abcd,
                         'self':self_abcd,
                         'random':random_abcd}
            
            for key in all_abcds.keys():
                abcd = all_abcds[key]
                project_result = {}
                if 'f1' not in all_model_results[key].keys():
                    all_model_results[key]['f1'] = []
                    all_model_results[key]['precision'] = []
                    all_model_results[key]['recall'] = []
                    all_model_results[key]['g-score'] = []
                    all_model_results[key]['d2h'] = []
                    all_model_results[key]['pci_20'] = []
                    all_model_results[key]['ifa'] = []
                    all_model_results[key]['pd'] = []
                    all_model_results[key]['pf'] = []
                all_model_results[key]['f1'].append(abcd.calculate_f1_score())
                all_model_results[key]['precision'].append(abcd.calculate_precision())
                all_model_results[key]['recall'].append(abcd.calculate_recall())
                all_model_results[key]['g-score'].append(abcd.get_g_score())
                all_model_results[key]['d2h'].append(abcd.calculate_d2h())
                all_model_results[key]['pci_20'].append(abcd.get_pci_20())
                try:
                    all_model_results[key]['ifa'].append(abcd.get_ifa_roc())
                except:
                    all_model_results[key]['ifa'].append(0)
                all_model_results[key]['pd'].append(abcd.get_pd())
                all_model_results[key]['pf'].append(abcd.get_pf())
                
        for goal in goals:
            if goal == 'g':
                _goal = 'g-score'
            else:
                _goal = goal
            if _goal not in results.keys():
                results[_goal] = {}
            if t_project not in results[_goal].keys():
                results[_goal][t_project] = []
            for key in all_model_results.keys():
                results[_goal][t_project].append(np.median(all_model_results[key][_goal]))
    for key in results:
        df = pd.DataFrame.from_dict(results[key],orient='index',columns = list(all_model_results.keys()))
        if not Path(data_location).is_dir():
            os.makedirs(Path(data_location))
        df.to_csv(data_location + '/bellwether_' + key + '.csv')
    return results      
    
    
if __name__ == "__main__":
    metric = 'process'
    for i in range(10):
        fold = str(i)
        data_location = 'src/results/median_data_1/level_2/fold_' + fold
        cluster_data_loc = 'src/results/median_data_1/level_2/fold_' + fold
        default_bellwether_loc = 'src/results/median_data_1/default_bell/fold_' + fold
        results = get_predicted_all(cluster_data_loc,fold,data_location,default_bellwether_loc,metric)
        with open(data_location + '/level_' + str(depth) + '.pkl', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)