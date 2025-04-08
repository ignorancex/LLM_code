#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:13:26 2021

@author: trduong
"""

import pandas as pd
import numpy as np
import yaml
import random
import torch

from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from geomloss import SamplesLoss


def load_data(path, name):
    print('loading dataset: ', name)
    if name == 'law':
        csv_data = pd.read_csv(path)

        races = ['White', 'Black', 'Asian', 'Hispanic', 'Mexican', 'Other', 'Puertorican', 'Amerindian']
        sexes = [1, 2]

        # index selection
        selected_races = ['White', 'Black', 'Asian']
        print("select races: ", selected_races)
        select_index = np.array(csv_data[(csv_data['race'] == selected_races[0]) | (csv_data['race'] == selected_races[1]) |
                                         (csv_data['race'] == selected_races[2])].index, dtype=int)
        # shuffle
        np.random.shuffle(select_index)

        x = csv_data[['LSAT','UGPA']].to_numpy()[select_index]  # n x d
        y = csv_data[['ZFYA']].to_numpy()[select_index]  # n x 1

        n = x.shape[0]
        env_race = csv_data['race'][select_index].to_list()  # n, string list
        env_race_onehot = np.zeros((n, len(selected_races)))
        #env_sex = csv_data['sex'][select_index].to_list()  # n, int list
        #env_sex_onehot = np.zeros((n, len(sexes)))
        for i in range(n):
            env_race_onehot[i][selected_races.index(env_race[i])] = 1.   # n x len(selected_races)
            #env_sex_onehot[i][sexes.index(env_sex[i])] = 1.
        #env = np.concatenate([env_race_onehot, env_sex_onehot], axis=1)  # n x (No. races + No. sex)
        env = env_race_onehot

    return x, y, env

def run_full(x, y, env, trn_idx, tst_idx):
    clf = LinearRegression()  # linear ? logistic ?

    features_full = np.concatenate([x, env], axis=1)
    features_full_trn = features_full[trn_idx]
    features_full_tst = features_full[tst_idx]

    clf.fit(features_full_trn, y[trn_idx])  # train

    # test
    y_pred_tst = clf.predict(features_full_tst)
    return y_pred_tst, clf

def run_unaware(x, y, trn_idx, tst_idx):
    clf = LinearRegression()  # linear ? logistic ?
    clf.fit(x[trn_idx], y[trn_idx])  # train

    # test
    y_pred_tst = clf.predict(x[tst_idx])
    return y_pred_tst, clf

def split_trn_tst_random(trn_rate, tst_rate, n):
    trn_id_list = random.sample(range(n), int(n * trn_rate))
    not_trn = list(set(range(n)) - set(trn_id_list))
    tst_id_list = random.sample(not_trn, int(n * tst_rate))
    val_id_list = list(set(not_trn) - set(tst_id_list))
    trn_id_list.sort()
    val_id_list.sort()
    tst_id_list.sort()
    return trn_id_list, val_id_list, tst_id_list

def evaluate_fairness(y_cf1, y_cf2):  # y_cf1: n x samplenum
    mmd = mmd2_lin(y_cf1, y_cf2, 0.3)
    return mmd

if __name__ == "__main__":
    with open("/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml", 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    """Load data"""
    data_path = conf['data_law']
    data_name = "law"
    x, y, env = load_data(data_path, data_name)
    x_dim = x.shape[1]
    n = len(x)

    """Preprocess data"""
    standardize = True
    if standardize:
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 0.000001)
    
    """Split train/test"""
    trn_rate = 0.6
    tst_rate = 0.2
    
    trn_idx, val_idx, tst_idx = split_trn_tst_random(trn_rate, tst_rate, n)
    
    """Run model"""
    full_y_prediction , full_model = run_full(x, y, env, trn_idx, tst_idx)
    unaware_y_prediction , unaware_model = run_unaware(x, y, env, trn_idx, tst_idx)
    
