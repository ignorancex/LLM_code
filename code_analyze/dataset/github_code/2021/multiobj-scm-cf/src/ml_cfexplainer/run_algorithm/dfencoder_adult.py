#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 10:58:19 2020

@author: trduong
"""

import sys
import pandas as pd
import argparse
import torch
import yaml 
import logging 

from dfencoder.autoencoder import AutoEncoder


if __name__ == "__main__":

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='adult')
    parser.add_argument('--version', type=str, default='full')
    parser.add_argument('--emb_size', type=int, default=512)

    args = parser.parse_args()
    name = args.name
    version = args.version
    emb_size = args.emb_size

    """Load configuration"""
    with open('/home/trduong/Data/multiobj-scm-cf/src/config.yml') as file:
        conf = yaml.safe_load(file)

    """Set up logging"""
    logger = logging.getLogger('genetic')
    file_handler = logging.FileHandler(filename=conf['credit_log_path'])
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)

    """Load data"""
    data = pd.read_feather(conf['processed_data_adult'], columns=None, use_threads=True)

    feature_col = ['age', 'hours_per_week', 'workclass', 'education', 'marital_status', 'occupation', 'race',
                   'gender']
    outcome = 'income'
    category = ['workclass', 'education', 'marital_status', 'occupation', 'race', 'gender']

    """Convert data to category"""
    for v in category:
        data[v] = pd.Categorical(data[v].values)


    """Select feature for training"""
    features = data[feature_col]
    train = features.sample(frac=.8, random_state=42)
    test = features.loc[~data.index.isin(train.index)]

    X_train = features
    X_val = test

    """Model architecture"""
    ae_model = AutoEncoder(
        encoder_layers=[512, 512, emb_size],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        activation='relu',
        swap_p=0.2,  # noise parameter
        lr=0.01,
        lr_decay=.99,
        batch_size=512,  # 512
        verbose=False,
        optimizer='sgd',
        scaler='gauss_rank',  # gauss rank scaling forces your numeric features into standard normal distributions
    )

    """Train model"""
    ae_model.to(device)
    ae_model.generator_fit(X_train, epochs=100, val=X_val)

    """Save model"""
    torch.save(ae_model, conf['autoencoder_model_credit'].format(version, emb_size))


