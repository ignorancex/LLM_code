#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:55:24 2021
@author: trduong
"""

import pandas as pd
import numpy as np
import logging
import yaml
import pyro
import torch 
import pyro.distributions as dist
import sys

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from torch import nn
from pyro.nn import PyroModule
from tqdm import tqdm 


def GroundTruthModel():
    """
    

    Returns
    -------
    None.

    """
    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Nknowledge': dist.Normal(torch.tensor(0.), torch.tensor(1.))
    }
    
    R = pyro.sample("Race", exo_dist['Nrace'])
    S = pyro.sample("Sex", exo_dist['Nsex'])
    K = pyro.sample("Knowledge", exo_dist['Nknowledge'])
    
    # PsuedoDelta 
    G = pyro.sample("UGPA", dist.Normal(K + R + S, 0.1))
    L = pyro.sample("LSAT", dist.Normal(K + R + S, 0.1))
    F = pyro.sample("ZFYA", dist.Normal(K + R + S, 0.1))


def infer_knowledge(df):
    """
    
    :param df: DESCRIPTION
    :type df: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """
    
    knowledge = []
    for i in tqdm(range(len(df))):
        conditioned = pyro.condition(GroundTruthModel, data={"UGPA": df["UGPA"][i], "LSAT": df["LSAT"][i]})
        posterior = pyro.infer.Importance(conditioned, num_samples=10).run()
        post_marginal = pyro.infer.EmpiricalMarginal(posterior, "Knowledge")
        post_samples = [post_marginal().item() for _ in range(10)]
        post_unique, post_counts = np.unique(post_samples, return_counts=True)
        mean = np.mean(post_samples)
        knowledge.append(mean)
    return knowledge
    

if __name__ == "__main__":
    
    """Load configuration"""
    with open("/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml", 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    """Set up logging"""
    logger = logging.getLogger('genetic')
    file_handler = logging.FileHandler(filename=conf['log_law'])
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)
    

    """Load data"""
    data_path = conf['data_law']
    df = pd.read_csv(data_path)
    
    """Setup features"""
    sensitive_feature = ['race', 'sex']
    normal_feature = ['LSAT', 'UGPA']
    categorical_feature = ['race', 'sex']
    full_feature = sensitive_feature + normal_feature
    target = 'ZFYA'
    selected_race = ['White', 'Black']
    df = df[df['race'].isin(selected_race)]
    
    df = df.reset_index(drop = True)
    
    """Preprocess data"""
    df['LSAT'] = (df['LSAT']-df['LSAT'].mean())/df['LSAT'].std()
    df['UGPA'] = (df['UGPA']-df['UGPA'].mean())/df['UGPA'].std()
    df['ZFYA'] = (df['ZFYA']-df['ZFYA'].mean())/df['ZFYA'].std()
    
    le = preprocessing.LabelEncoder()
    df['race'] = le.fit_transform(df['race'])
    df['sex'] = le.fit_transform(df['sex'])
    
    """Full model"""
    logger.debug('Full model')
    
    reg = LinearRegression().fit(df[full_feature], df['ZFYA'])
    y_pred = reg.predict(df[full_feature].values)
    df['full_prediction'] = y_pred.reshape(-1)

    """Unaware model"""
    logger.debug('Unware model')
    
    reg = LinearRegression().fit(df[normal_feature], df['ZFYA'])
    y_pred = reg.predict(df[normal_feature].values)
    df['unaware_prediction'] = y_pred.reshape(-1)

    """Counterfactual fairness model"""
    for i in ['LSAT', 'UGPA', 'ZFYA']:
        df[i] = [torch.tensor(x) for x in df[i].values]
        
        
    
    logger.debug('Counterfactual fairness model')
    knowledged = infer_knowledge(df)
    knowledged = np.array(knowledged).reshape(-1,1)
    reg = LinearRegression().fit(knowledged, df['ZFYA'])
    y_pred = reg.predict(knowledged)
    df['cf_prediction'] = y_pred.reshape(-1)
    
    df['ZFYA'] = [x.detach().numpy() for x in df['ZFYA']]
    df['LSAT'] = [x.detach().numpy() for x in df['LSAT']]
    df['UGPA'] = [x.detach().numpy() for x in df['UGPA']]
    
    """Output the result"""
    df = df[['race', 'sex', 'LSAT', 'UGPA', 'ZFYA', 'full_prediction', 'unaware_prediction', 'cf_prediction']]
    df.to_csv(conf['result_law'], index = False)
    
    
    sys.modules[__name__].__dict__.clear()