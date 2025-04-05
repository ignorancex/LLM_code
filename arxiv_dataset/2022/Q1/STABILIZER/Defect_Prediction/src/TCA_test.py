from __future__ import print_function, division

import os
from os import listdir
from os.path import isfile, join
import sys
import pandas as pd
import random
import numpy as np
import copy
from operator import add 

# root = os.path.join(os.getcwd().split('src')[0], 'src/defects')
# if root not in sys.path:
#     sys.path.append(root)

import warnings


from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import *
from mklaren.projection.icd import ICD
from pdb import set_trace
from scipy.spatial.distance import pdist, squareform


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from multiprocessing import Pool, cpu_count
from threading import Thread
from multiprocessing import Queue
# import tl_algs
# from tl_algs import tca_plus

import SMOTE
from utils import *
import metrics


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


def apply_smote(df):
        cols = df.columns
        smt = SMOTE.smote(df)
        df = smt.run()
        df.columns = cols
        return df

def _replaceitem(x):
    if x >= 0.5:
        return 0.5
    else:
        return 0.0

def _replaceitem_logistic(x):
    if x >= 0.5:
        return 1
    else:
        return 0


def prepare_data(project, metric):
    data_path = '../data/merged_data_original/' + project + '.csv'
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


def get_kernel_matrix(dframe, n_dim=15):
    """
    This returns a Kernel Transformation Matrix $\Theta$

    It uses kernel approximation offered by the MKlaren package
    For the sake of completeness (and for my peace of mind, I use the best possible approx.)

    :param dframe: input data as a pandas dataframe.
    :param n_dim: Number of dimensions for the kernel matrix (default=15)
    :return: $\Theta$ matrix
    """
    ker = Kinterface(data=dframe.values, kernel=linear_kernel)
    model = ICD(rank=n_dim)
    model.fit(ker)
    g_nystrom = model.G
    return g_nystrom



def map_transform(src, tgt, n_components=5):
    """
    Run a map and transform x and y onto a new space using TCA
    :param src: IID samples
    :param tgt: IID samples
    :return: Mapped x and y
    """
    s_col = [col for col in src.columns[:-1] if '?' not in col]
    t_col = [col for col in tgt.columns[:-1] if '?' not in col]
    S = src[s_col]
    T = tgt[t_col]


    col_name = ["Col_" + str(i) for i in range(n_components)]
    x0 = pd.DataFrame(get_kernel_matrix(S, n_components), columns=col_name)
    y0 = pd.DataFrame(get_kernel_matrix(T, n_components), columns=col_name)
    # set_trace()
    x0.loc[:, src.columns[-1]] = pd.Series(src[src.columns[-1]], index=x0.index)
    y0.loc[:, tgt.columns[-1]] = pd.Series(tgt[tgt.columns[-1]], index=y0.index)

    return x0, y0


def create_model(train):
    """
    :param train:
    :type train:
    :param test:
    :type test:
    :param weka:
    :type weka:
    :return:
    """
    train = apply_smote(train)

    train_y = train.Bugs
    train_X = train.drop(labels = ['Bugs'],axis = 1)

    clf = LogisticRegression()
    clf.fit(train_X, train_y)

    return clf 

def predict_defects(clf, test):
    test_y = test.Bugs
    test_X = test.drop(labels = ['Bugs'],axis = 1)

    predicted = clf.predict(test_X)
    predicted_proba = clf.predict_proba(test_X)

    return test_y, predicted, predicted_proba



def get_dcv(src, tgt):
    """Get dataset characteristic vector."""
    s_col = [col for col in src.columns[:-1] if '?' not in col]
    t_col = [col for col in tgt.columns[:-1] if '?' not in col]
    S = src[s_col]
    T = tgt[t_col]

    def self_dist_mtx(arr):
        dist_arr = pdist(arr)
        return squareform(dist_arr)

    dist_src = self_dist_mtx(S.values)
    dist_tgt = self_dist_mtx(T.values)

    dcv_src = [np.mean(dist_src), np.median(dist_src), np.min(dist_src), np.max(dist_src), np.std(dist_src),
               len(S.values)]
    dcv_tgt = [np.mean(dist_tgt), np.median(dist_tgt), np.min(dist_tgt), np.max(dist_tgt), np.std(dist_tgt),
               len(T.values)]
    return dcv_src, dcv_tgt


def sim(c_s, c_t, e=0):
    if c_s[e] * 1.6 < c_t[e]:
        return "VH"  # Very High
    if c_s[e] * 1.3 < c_t[e] <= c_s[e] * 1.6:
        return "H"  # High
    if c_s[e] * 1.1 < c_t[e] <= c_s[e] * 1.3:
        return "SH"  # Slightly High
    if c_s[e] * 0.9 <= c_t[e] <= c_s[e] * 1.1:
        return "S"  # Same
    if c_s[e] * 0.7 <= c_t[e] < c_s[e] * 0.9:
        return "SL"  # Slightly Low
    if c_s[e] * 0.4 <= c_t[e] < c_s[e] * 0.7:
        return "L"  # Low
    if c_t[e] < c_s[e] * 0.4:
        return "VL"  # Very Low


def smart_norm(src, tgt, c_s, c_t):
    """
    ARE THESE NORMS CORRECT?? OPEN AN ISSUE REPORT TO VERIFY
    :param src:
    :param tgt:
    :param c_s:
    :param c_t:
    :return:
    """
    try:  # !!GUARD: PLEASE REMOVE AFTER DEBUGGING!!
        # Rule 1
        
        if sim(c_s, c_t, e=0) == "S" and sim(c_s, c_t, e=-2) == "S":
            # print("Rule 1")
            return src, tgt

        # Rule 2
        elif sim(c_s, c_t, e=2) == "VL" or "VH" \
                and sim(c_s, c_t, e=3) == "VL" or "VH" \
                and sim(c_s, c_t, e=-1) == "VL" or "VH":
            # print("Rule 2")
            return df_norm(src), df_norm(tgt)

        # Rule 3.1
        elif sim(c_s, c_t, e=-2) == "VH" and c_s[-1] > c_t[-1] or \
                                sim(c_s, c_t, e=-2) == "VL" and c_s[-1] < c_t[-1]:
            # print("Rule 3")
            return df_norm(src, type="normal"), df_norm(tgt)

        # Rule 4
        elif sim(c_s, c_t, e=-2) == "VH" and c_s[-1] < c_t[-1] or \
                                sim(c_s, c_t, e=-2) == "VL" and c_s[-1] > c_t[-1]:
            # print("Rule 4")
            return df_norm(src), df_norm(tgt, type="normal")
        else:
            # print("Rule 5")
            return df_norm(src, type="normal"), df_norm(tgt, type="normal")
    except:
        set_trace()
        return src, tgt

def tca_plus_test(source, target):
    """
    TCA: Transfer Component Analysis
    :param source:
    :param target:
    :param n_rep: number of repeats
    :return: result
    """
    result = dict()
    metric = 'process'
    predicted_probability = []
    for src_name in source:
        stats = []
        val = []
        src = prepare_data(src_name, metric)
        for tgt_name in target:
            tgt = prepare_data(tgt_name, metric)
            loc = tgt['file_la'] + tgt['file_lt']
            dcv_src, dcv_tgt = get_dcv(src, tgt)

            norm_src, norm_tgt = smart_norm(src, tgt, dcv_src, dcv_tgt)
            _train, _test = map_transform(norm_src, norm_tgt)

            clf = create_model(_train)

            actual, predicted, predicted_proba = predict_defects(clf=clf, test=_test)

            predicted_proba = np.array(predicted_proba)
            predicted_proba = predicted_proba[:,1]

            predicted_probability.append(predicted_proba)

    predicted_probability_f = list(map(_replaceitem, predicted_probability[0]))
    predicted_probability_p = [x / 2 for x in predicted_probability[1]]

    final_predicted_proba = list(map(add, predicted_probability_f, predicted_probability_p)) 

    predicted = list(map(_replaceitem_logistic, final_predicted_proba))

    abcd = metrics.measures(actual,predicted,loc)

    recall = abcd.calculate_recall()
    pf = abcd.get_pf()
    g = abcd.get_g_score()
    f = abcd.calculate_f1_score()
    pci_20 = abcd.get_pci_20()
    precision = abcd.calculate_precision()
    ifa = abcd.get_ifa()
    return tgt_name, recall, pf, g, f, pci_20, precision, ifa

    


if __name__ == "__main__":
    path = 'data/merged_data_original/'
    for fold in range(10):
        prediction_df = pd.read_csv('src/results/median_data_1/level_2/fold_' + str(fold) + '/predicted_source_process.csv')
        prediction_df = prediction_df.drop(['Unnamed: 0'], axis = 1)
        prediction_df = prediction_df.sort_values(['trg'])
        src_projects = []
        source_p = set()
        for trg in prediction_df.trg.unique():
            sub_df = prediction_df[prediction_df['trg'] == trg]
            sub_df.reset_index(inplace = True, drop = True)
            print(trg,sub_df.iloc[sub_df.f.idxmax()].src, sub_df.f.max())
            src_projects.append([trg,sub_df.iloc[sub_df.f.idxmax()].src,sub_df.iloc[sub_df.pci_20.idxmax()].src])
            source_p.add(sub_df.iloc[sub_df.pci_20.idxmax()].src)
        results = []
        for project_pairs in src_projects:
            try:    
                tgt_name, recall, pf, g, f, pci_20,precision, ifa = tca_plus_test(project_pairs[1:],[project_pairs[0]])
                print(tgt_name, recall, pf, g, f, pci_20, precision, ifa)
                results.append([tgt_name, recall, pf, g, f, pci_20, precision, ifa])
            except:
                continue
        results_df = pd.DataFrame(results, columns = ['tgt_name', 'recall', 'pf', 'g', 'f1', 'pci_20', 'precision', 'ifa'])
        results_df.to_csv('results/median_data_1/level_2/fold_' + str(fold) + '/2PTL_results.csv')
    

