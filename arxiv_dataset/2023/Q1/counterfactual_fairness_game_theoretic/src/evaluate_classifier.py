#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:17:22 2021

@author: trduong
"""

# import os, sys;
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import pandas as pd
import logging
import argparse
import sys

from utils.evaluate_func import evaluate_distribution, evaluate_fairness, evaluate_classifier, classification_performance
from utils.helpers import load_config
from utils.helpers import features_setting
if __name__ == "__main__":
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, default='both')
    parser.add_argument('--data_name', type=str, default='compas')
    # parser.add_argument('--data_name', type=str, default='adult')
    # parser.add_argument('--data_name', type=str, default='bank')

    args = parser.parse_args()
    data_name = args.data_name

    """Load configuration"""
    config_path = "/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml"
    conf = load_config(config_path)

    """Set up logging"""
    logger = logging.getLogger('genetic')
    file_handler = logging.FileHandler(filename=conf['evaluate_compas_log'])
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)

    """Setup features"""
    dict_ = features_setting(data_name)
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]

    baseline_path = conf['{}'.format(data_name)]
    ivr_path = conf['ivr_{}'.format(data_name)]
    evaluate_path = conf['evaluate_{}'.format(data_name)]

    logger.debug("Baseline path: {}".format(baseline_path))
    logger.debug("Evaluate path: {}".format(evaluate_path))
    logger.debug("Invariant path: {}".format(ivr_path))

    # baseline_columns = ['full', 'unaware', 'cf1', 'cf2']
    # method_columns = ['AL_prediction', 'GL_prediction', 'GD_prediction']
    # prediction_columns = baseline_columns + method_columns



    """Load data"""
    df1 = pd.read_csv(baseline_path)
    df2 = pd.read_csv(ivr_path)
    df2 = df2.drop(columns=full_features+[target])
    df = pd.concat([df1, df2], axis=1)

    if data_name == 'compas':
        col = ['full_log',
               'full_gbr',
               'unaware_log',
               'unaware_gbr',
               'cf1',
               'cf2',
               'AL_prediction',
               'GL_prediction',
               'GD_prediction'
               ]
    else:
        col = ['full_log',
               'full_gbr',
               'unaware_log',
               'unaware_gbr',
               'cf1',
               'cf2',
               'AL_prediction',
               'GL_prediction',
               'GD_prediction'
               ]

    df_result = pd.DataFrame()

    """Evaluate performance"""

    print(df)
    df_result = classification_performance(df,
                                        df_result,
                                        sensitive_features,
                                        target,
                                        col,
                                        data_name)

    cols = list(df_result)
    cols.insert(0, cols.pop(cols.index('method')))
    df_result = df_result[cols]
    print("Output to {}".format(evaluate_path))
    df_result.to_csv(evaluate_path, index = False)
    
    logger.debug(df_result)
    sys.modules[__name__].__dict__.clear()

    
    