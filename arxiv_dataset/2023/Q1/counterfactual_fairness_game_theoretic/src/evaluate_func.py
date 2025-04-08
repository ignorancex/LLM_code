#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:01:51 2021

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

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from geomloss import SamplesLoss


def evaluate_pred(y_pred, y_true):
    """
    
    :param y_pred: DESCRIPTION
    :type y_pred: TYPE
    :param y_true: DESCRIPTION
    :type y_true: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    evaluations = {}
    evaluations['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    evaluations['MAE'] = mean_absolute_error(y_true, y_pred)
    # evaluations['R2score'] = r2_score(y_true, y_pred)


    return evaluations

def evaluate_distribution(ys, ys_hat):
    """
    :param ys: DESCRIPTION
    :type ys: TYPE
    :param ys_hat: DESCRIPTION
    :type ys_hat: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """
    
    evaluation = {}
    
    Loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
    evaluation['sinkhorn'] = Loss(ys, ys_hat).cpu().detach().numpy() 
    
    Loss = SamplesLoss("energy", p=2, blur=0.05, scaling=0.8)
    evaluation["energy"] = Loss(ys, ys_hat).cpu().detach().numpy() 
    
    Loss = SamplesLoss("gaussian", p=2, blur=0.05, scaling=0.8)
    evaluation["gaussian"] = Loss(ys, ys_hat).cpu().detach().numpy() 
    
    return evaluation 


def evaluate_fairness(sensitive_att, df, target):
    df = df.sample(frac=0.35, replace=True, random_state=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sinkhorn, energy, gaussian = 0,0,0
    
    for s in sensitive_att:
        ys = df[df[s] == 1][target].values
        ys_hat = df[df[s] == 0][target].values

        ys = torch.Tensor(ys).to(device).reshape(-1,1)
        ys_hat = torch.Tensor(ys_hat).to(device).reshape(-1,1)
        
        eval_performance = evaluate_distribution(ys, ys_hat)
        sinkhorn += eval_performance['sinkhorn']
        energy += eval_performance['energy']
        gaussian += eval_performance['gaussian']
        
        print("Sensitive ", s, sinkhorn)
    
    eval_performance = {}
    eval_performance['sinkhorn'] = sinkhorn/len(sensitive_att)
    eval_performance['energy'] = energy/len(sensitive_att)
    eval_performance['gaussian'] = gaussian/len(sensitive_att)
    
    return eval_performance
    
    
if __name__ == "__main__":    
    """Load configuration"""
    with open("/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml", 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    """Set up logging"""
    logger = logging.getLogger('genetic')
    file_handler = logging.FileHandler(filename=conf['evaluate_law'])
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)
    

    """Load data"""
    data_path = conf['result_law']
    df = pd.read_csv(data_path)
    
    df = df.sample(frac=0.5, replace=True, random_state=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ys = df[df['race'] == 0].values
    ys_hat = df[df['race'] == 1].values
        
    ys = torch.Tensor(ys).to(device).reshape(-1,1)
    ys_hat = torch.Tensor(ys_hat).to(device).reshape(-1,1)
        
    #evaluate_distribution(ys, ys_hat)
    Loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
    Loss(ys, ys_hat)
