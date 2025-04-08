import torch
import pandas as pd
import sys
import numpy as np

from tqdm import tqdm
from model_arch.discriminator import DiscriminatorLaw
from dfencoder.autoencoder import AutoEncoder
from dfencoder.dataframe import EncoderDataFrame
from utils.evaluate_func import evaluate_pred, evaluate_distribution, evaluate_fairness
from utils.helpers import preprocess_dataset
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

import argparse

# def load_aemodel(model, path, df):
#     print("Path {}".format(path))
#     print("Model ", model)
#     print(df)
#     model.build_model(df.copy())
#     model.load_state_dict(path)
#     model.eval()
#     return model

def get_predict(ae_model, generator, discriminator, df, normal_features, full_features, l = ''):

    GD_prediction = 'GD_prediction' + l

    df_generator = df[normal_features].copy()
    df_autoencoder = df[full_features].copy()

    Z = ae_model.get_representation(df_autoencoder)
    Z = Z.cpu().detach().numpy()
    reg = LinearRegression()
    reg.fit(Z, df['ZFYA'].values)
    y_pred = reg.predict(Z)
    df["AL_prediction"] = y_pred

    """Generator + Linear regression"""
    Z = generator.custom_forward(df_generator)
    Z = Z.cpu().detach().numpy()
    reg = LinearRegression()
    reg.fit(Z, df['ZFYA'].values)
    y_pred = reg.predict(Z)
    df["GL_prediction"] = y_pred

    """Generator + Discriminator"""
    Z = generator.custom_forward(df_generator)
    predictor_agnostic = discriminator(Z)
    y_pred = predictor_agnostic.cpu().detach().numpy().reshape(-1)
    df[GD_prediction] = y_pred

    return df



if __name__ == "__main__":
    """Parsing argument"""
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--lambda_weight', type=str, default="0.1 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8 8.5 9 9.5 10 20 30 40 50")
    # parser.add_argument('--run_lambda', action='store_true')

    # args = parser.parse_args()
    # run_lambda = args.run_lambda
    # lambda_weight = args.lambda_weight
    # print(lambda_weight)
    # print(lambda_weight.split(" "))
    # lambda_weight = [float(x) for x in lambda_weight.split(' ')]
    # lambda_weight = [str(x) for x in lambda_weight]


    # if run_lambda:
    #     print("Run lambda with lambda ", lambda_weight)
    # else:
    #     print("Run normal flow")

    """Device"""
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    """Load configuration"""
    config_path = "/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml"
    conf = load_config(config_path)

    """Set up logging"""
    logger = setup_logging(conf['log_train_law'])

    """Load data"""
    data_path = conf['data_law']
    df = pd.read_csv(data_path)

    """Setup features"""
    data_name = "law"
    dict_ = features_setting("law")
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]

    selected_race = ['White', 'Black']
    df = df[df['race'].isin(selected_race)]
    df = df.reset_index(drop = True)

    """Preprocess data"""
    df = preprocess_dataset(df, [], categorical_features)
    # df['ZFYA'] = (df['ZFYA']-df['ZFYA'].mean())/df['ZFYA'].std()
    df = df[['LSAT', 'UGPA', 'sex', 'race', 'ZFYA']]


    _, df_test = train_test_split(df, test_size=0.2, random_state=0)

    """Load auto encoder"""
    df_autoencoder = df_test[full_features].copy()
    emb_size = 128
    ae_model = AutoEncoder(
        input_shape=df_test[full_features].shape[1],
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
    ae_model.to(device)
    # ae_model = load_aemodel(ae_model, conf['law_encoder'], df_test_autoencoder)
    ae_model.build_model(df_test[full_features].copy())
    ae_model.load_state_dict(torch.load(conf['law_encoder']))
    ae_model.eval()

    """Load generator"""
    emb_size = 128
    df_generator = df_test[normal_features]
    generator = AutoEncoder(
        input_shape = df_generator.shape[1],
        encoder_layers=[256, 256, emb_size],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        encoder_dropout = 0.5,
        decoder_dropout = 0.5,
        activation='tanh',
        swap_p=0.2,  # noise parameter
        lr=0.0001,
        lr_decay=.99,
        batch_size=512,  # 512
        verbose=False,
        optimizer='sgd',
        scaler='gauss_rank',  # gauss rank scaling forces your numeric features into standard normal distributions
    )
    generator.to(device)
    generator.build_model(df[normal_features])
    generator.eval()

    """Load discriminator"""
    discriminator = DiscriminatorLaw(emb_size)
    discriminator.to(device)
    discriminator.load_state_dict(torch.load(conf['law_discriminator']))
    discriminator.eval()

    # if run_lambda:
    #     for l in lambda_weight:
    #         print("Lambda ", l)
    #         generator.load_state_dict(torch.load(conf["lambda_law_generator"].format(l)))
    #         discriminator.load_state_dict(torch.load(conf["lambda_law_discriminator"].format(l)))
    #         df_test = get_predict(ae_model, generator, df_test, normal_features, full_features, l)
    # else:
    generator.load_state_dict(torch.load(conf['law_generator']))
    discriminator.load_state_dict(torch.load(conf['law_discriminator']))

    df_test = get_predict(ae_model, generator, discriminator, df_test, normal_features, full_features)

    # if run_lambda:
    #     df_test.to_csv(conf["ivr_law_lambda"], index = False)
    # else:
    df_test.to_csv(conf["ivr_law"], index = False)

    """Autoencoder + Linear regression"""
    # Z = ae_model.get_representation(df_autoencoder)
    # Z = Z.cpu().detach().numpy()
    # reg = LinearRegression()
    # reg.fit(Z, df['ZFYA'].values)
    # y_pred = reg.predict(Z)
    # df["AL_prediction"] = y_pred

    """Generator + Linear regression"""
    # Z = generator.custom_forward(df_generator)
    # Z = Z.cpu().detach().numpy()
    # reg = LinearRegression()
    # reg.fit(Z, df['ZFYA'].values)
    # y_pred = reg.predict(Z)
    # df["GL_prediction"] = y_pred

    """Generator + Discriminator"""
    # Z = generator.custom_forward(df_generator)
    # predictor_agnostic = discriminator_agnostic(Z)
    # y_pred = predictor_agnostic.cpu().detach().numpy().reshape(-1)
    # df["GD_prediction"] = y_pred





