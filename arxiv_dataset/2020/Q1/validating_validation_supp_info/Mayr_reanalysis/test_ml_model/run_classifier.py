'''
@author: Matthew C. Robinson
@email: matthew67robinson@gmail.com

The code below gives an example of running a standard Scikit-Learn Classifier
on the Mayr et al. data. Our code uses the same folds and nested cluster CV scheme
used by the authors. In this case, the classifier is a SVM, which 
can be quite computationally intensive. 

In our case, the main quantities of interest from this analysis were not the 
performances, but rather the train/test sizes and imbalances.
'''

import math
import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse

import pickle
import imp
import os
import sys
import time
import gc
import rdkit
import rdkit.Chem
from rdkit.Chem import AllChem

with open('../data_for_replication/sparse_target_matrix.pckl', 'rb') as f:
    target_mat = pickle.load(f)

with open('../data_for_replication/folds.pckl', 'rb') as f:
    folds = pickle.load(f)

with open('../data_for_replication/sparse_fps_arr.pckl', 'rb') as f:
    sparse_fp_arr = pickle.load(f)

import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from itertools import product

def get_train_test_sets(train_idx, test_idx, target_idx):
    train_target_compounds = target_mat[train_idx, target_idx].copy().toarray().reshape(-1,)
    test_target_compounds = target_mat[test_idx, target_idx].copy().toarray().reshape(-1,)

    y_train = train_target_compounds[np.nonzero(train_target_compounds)]
    y_test = test_target_compounds[np.nonzero(test_target_compounds)]

    train_fp_arr = sparse_fp_arr[train_idx]
    test_fp_arr = sparse_fp_arr[test_idx]

    X_train_fps = train_fp_arr[np.nonzero(train_target_compounds)].toarray()
    X_train = X_train_fps[(X_train_fps>=0).all(axis=1)] # b/c failed fp rows are all -1
    y_train = y_train[(X_train_fps>=0).all(axis=1)]

    X_test_fps = test_fp_arr[np.nonzero(test_target_compounds)].toarray()
    X_test = X_test_fps[(X_test_fps>=0).all(axis=1)]
    y_test = y_test[(X_test_fps>=0).all(axis=1)]

    return X_train, X_test, y_train, y_test

def clustered_cv(fold1_idx, fold2_idx, target_idx, param_grid):
    items = sorted(param_grid.items())
    keys, values = zip(*items)
    param_dicts_list = []
    for v in product(*values):
        param_dict = dict(zip(keys,v))
        param_dicts_list.append(param_dict)
 
    scoring_dict = dict.fromkeys(list(range(len(param_dicts_list))), 0)
    # first fold
    X_train, X_test, y_train, y_test = get_train_test_sets(fold1_idx, fold2_idx, target_idx)
    for idx, param_dict in enumerate(param_dicts_list):
        svm = SVC(kernel='rbf', max_iter=10000000, cache_size=500)
        svm.set_params(**param_dict)
        svm.fit(X_train, y_train)
        try:
            y_pred = svm.decision_function(X_test)
        except:
            continue
        metric = metrics.roc_auc_score(y_test, y_pred)
        scoring_dict[idx] = scoring_dict[idx] + metric

    # second fold 
    X_train, X_test, y_train, y_test = get_train_test_sets(fold2_idx, fold1_idx, target_idx)
    for idx, param_dict in enumerate(param_dicts_list):
        svm = SVC(kernel='rbf', max_iter=10000000, cache_size=500)
        svm.set_params(**param_dict)
        svm.fit(X_train, y_train)
        try:
            y_pred = svm.decision_function(X_test)
        except:
            continue
        metric = metrics.roc_auc_score(y_test, y_pred)
        scoring_dict[idx] = scoring_dict[idx] + metric

    best_scoring_idx = [k for k,v in scoring_dict.items() if v==max(scoring_dict.values())][0]
    best_param_dict = param_dicts_list[best_scoring_idx]
    return best_param_dict

all_samples = folds[0] + folds[1] + folds[2]

def run_single_assay(target_idx):

    metrics_list = []
    train_size_list = []
    test_size_list = []
    train_imbalance_list = []
    test_imbalance_list = []
    best_scoring_list = []
    for fold_idx in [0,1,2]:

        # train_samples_idx = list(set(all_samples)-set(folds[fold_idx]))
        cv_folds  = [0,1,2]
        cv_folds.remove(fold_idx)
        test_samples_idx = folds[fold_idx]

        C_range = 10. ** np.arange(-3, 4, 2)
        gamma_range = 10. ** np.arange(-5, 4, 2)
        param_grid = dict(gamma=gamma_range, C=C_range)
        best_param_dict = clustered_cv(folds[cv_folds[0]], folds[cv_folds[1]], target_idx, param_grid)
        
        train_samples_idx = list(set(all_samples)-set(folds[fold_idx]))
        X_train, X_test, y_train, y_test = get_train_test_sets(train_samples_idx, test_samples_idx, target_idx)
        
        svm = SVC(kernel='rbf', max_iter=10000000, cache_size=500)
        svm.set_params(**best_param_dict)
        svm.fit(X_train, y_train)
        y_pred = svm.decision_function(X_test)
        metric = metrics.roc_auc_score(y_test, y_pred)
        metrics_list.append(metric)
        
        train_size = X_train.shape[0]
        test_size = X_test.shape[0]
        train_size_list.append(train_size)
        test_size_list.append(test_size )

        train_imbalance_list.append( np.mean(y_train==1) )
        test_imbalance_list.append( np.mean(y_test==1) )

        best_scoring_list.append(list(best_param_dict.values()))
        
    results  = [metrics_list[0], metrics_list[1], metrics_list[2],
                                train_size_list[0], train_size_list[1], train_size_list[2],
                                test_size_list[0], test_size_list[1], test_size_list[2],
                                train_imbalance_list[0], train_imbalance_list[1], train_imbalance_list[2],
                                test_imbalance_list[0], test_imbalance_list[1], test_imbalance_list[2],
                                best_scoring_list[0], best_scoring_list[1], best_scoring_list[2]]

    return target_idx, results

from joblib import Parallel, delayed

target_dict={}
for start_idx in range(0, target_mat.shape[1], 8):
    if target_mat.shape[1] - start_idx > 8:
        assay_results = Parallel(n_jobs=-1)(
                        delayed(run_single_assay)(i) for i in range(start_idx,start_idx+8))
    else:
        assay_results = Parallel(n_jobs=-1)(
                        delayed(run_single_assay)(i) for i in range(start_idx,target_mat.shape[1]))
    
    target_idxs, list_of_results = zip(*assay_results)
    # print(idxs)
    # print(list_of_results)
    for target_idx in target_idxs:
        target_dict[target_idx] = list_of_results[target_idx-target_idxs[0]]

    print(start_idx)
    with open('svm_target_dict.pckl','wb') as f:
        pickle.dump(target_dict,f)


with open('svm_target_dict_final.pckl','wb') as f:
            pickle.dump(target_dict,f)
