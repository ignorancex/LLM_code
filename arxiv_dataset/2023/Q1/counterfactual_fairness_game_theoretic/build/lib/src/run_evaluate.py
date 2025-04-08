#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:17:22 2021

@author: trduong
"""

import os, sys; 
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import pandas as pd
import numpy as np
import logging
import yaml
import pyro
import torch 
import pyro.distributions as dist
import sys
from evaluate_func import evaluate_pred, evaluate_distribution, evaluate_fairness



def evaluate_law(df, df_result):
    sensitive_att = ['race', 'sex']
    target = 'ZFYA'
    # for m in ['full_prediction', 'unaware_prediction', 'cf_prediction']:
    for m in ['full_prediction', 'unaware_prediction', 'cf_prediction', 'inv_prediction']:
        performance_reg = evaluate_pred(df[m].values, df[target].values)        
        performance_fairness = evaluate_fairness(sensitive_att, df, m)
        
        print(m, performance_fairness)
        performance_reg.update(performance_fairness)
        performance_reg['method'] = m
        df_result = df_result.append(performance_reg, ignore_index=True)
    return df_result

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
    

    logger.debug(df)

    
    df_result = pd.DataFrame()
    df_result['method'] = ''
    df_result['RMSE'] = ''
    df_result['MAE'] = ''
    # df_result['R2score'] = ''
    df_result['sinkhorn'] = ''
    df_result['energy'] = ''
    df_result['gaussian'] = ''
    

    """Evaluate performance"""
    df_result = evaluate_law(df, df_result)
    df_result.to_csv(conf['result_evaluate_law'], index = False)
    
    logger.debug(df_result)
    sys.modules[__name__].__dict__.clear()

    
    