#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 10:58:19 2020

@author: trduong

Build auto-encoder model for different dataset

"""

import pandas as pd
import torch
import argparse

from dfencoder.autoencoder import AutoEncoder
from utils.helpers import preprocess_dataset
from utils.helpers import load_config
from utils.helpers import features_setting
from torchsummary import summary

if __name__ == "__main__":
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='law')
    args = parser.parse_args()
    data_name = args.data_name

    """Device"""
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    """Load configuration"""
    config_path = "/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml"
    conf = load_config(config_path)

    """Load data"""
    if data_name == "law":
        data_path = conf['data_law']
        dict_ = features_setting("law")
        save_path = conf['law_encoder']
    elif data_name == "adult":
        data_path = conf['data_adult']
        dict_ = features_setting("adult")
        save_path = conf['adult_encoder']
    elif data_name == "compas":
        data_path = conf['data_compas']
        dict_ = features_setting("compas")
        save_path = conf['compas_encoder']

    df = pd.read_csv(data_path)

    """Setup features"""
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    discrete_features = dict_["discrete_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]
    standard_features = continuous_features + discrete_features

    """Preprocess data"""
    if data_name == "law":
        selected_race = ['White', 'Black']
        df = df[df['race'].isin(selected_race)]
        df = df.reset_index(drop=True)

    print(df.head())
    df = preprocess_dataset(df, [], categorical_features)
    print("Full features: ", full_features)
    print("Categorical features :", categorical_features)
    print("Continuous_features features: ", continuous_features)
    print("Standard features: ", standard_features)
    df = df[full_features]
    print(df.head())

    emb_size = 128
    """Model architecture"""
    ae_model = AutoEncoder(
        input_shape=df.shape[1],
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
    ae_model.fit(df, epochs=10000)
    # print(ae_model)
    # race_feature = ae_model.categorical_fts['race']
    # cats = race_feature['cats']
    # emb = race_feature['embedding']
    # print(emb)

    """Save model"""
    torch.save(ae_model.state_dict(), save_path)



