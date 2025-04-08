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

import repackage
repackage.up()

from src.utils.helpers import load_adult_income_dataset
# from src.utils.dataloader import DataLoader

def GroundTruthModel():
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
    

    """Load data and dataloader and normalize data"""
    # dataset = load_adult_income_dataset()
    # params= {'dataframe':dataset.copy(),
    #           'continuous_features':['age','hours_per_week'],
    #           'outcome_name':'income'}
    # d = DataLoader(params)
    # df = d.data_df
    # df = d.normalize_data(df)
    # print(df)
