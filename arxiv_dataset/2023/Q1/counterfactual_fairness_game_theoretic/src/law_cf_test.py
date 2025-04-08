import torch
import pandas as pd
import sys
import numpy as np

from tqdm import tqdm
from model_arch.discriminator import Discriminator_Law
from dfencoder.autoencoder import AutoEncoder
from dfencoder.dataframe import EncoderDataFrame
from utils.evaluate_func import evaluate_pred, evaluate_distribution, evaluate_fairness
from utils.helpers import preprocess_dataset
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
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
    df = preprocess_dataset(df, continuous_features, categorical_features)
    df['ZFYA'] = (df['ZFYA']-df['ZFYA'].mean())/df['ZFYA'].std()
    df = df[['LSAT', 'UGPA', 'sex', 'race', 'ZFYA']]

    df, df_test = train_test_split(df, test_size=0.1, random_state=0)
    df = df_test.copy()
    print(df_test[['LSAT', 'UGPA', 'sex', 'race', 'ZFYA']])
    sys.exit(1)

    """Load auto encoder"""
    df_autoencoder = df[full_features].copy()
    emb_size = 128
    ae_model = AutoEncoder(
        input_shape=df[full_features].shape[1],
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
    ae_model.build_model(df[full_features].copy())
    ae_model.load_state_dict(torch.load(conf['law_encoder']))
    ae_model.eval()

    """Load generator"""
    emb_size = 64
    df_generator = df[normal_features]
    generator= AutoEncoder(
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
    generator.build_model(df_generator)
    generator.load_state_dict(torch.load(conf['law_generator']))
    generator.eval()


    """Load discriminator"""
    emb_size = 64
    discriminator_agnostic = Discriminator_Law(emb_size)
    discriminator_agnostic.to(device)
    discriminator_agnostic.load_state_dict(torch.load(conf['law_discriminator']))
    discriminator_agnostic.eval()


    """Autoencoder + Linear regression"""
    Z = ae_model.get_representation(df_autoencoder)
    Z = Z.cpu().detach().numpy()
    reg = LinearRegression()
    reg.fit(Z, df['ZFYA'].values)
    y_pred = reg.predict(Z)
    df["AL_prediction"] = y_pred

    """Generator + Linear regression"""
    Z = generator.get_representation(df_generator)
    Z = Z.cpu().detach().numpy()
    reg = LinearRegression()
    reg.fit(Z, df['ZFYA'].values)
    y_pred = reg.predict(Z)
    df["GL_prediction"] = y_pred

    """Generator + Discriminator"""
    Z = generator.get_representation(df_generator)
    predictor_agnostic = discriminator_agnostic(Z)
    y_pred = predictor_agnostic.cpu().detach().numpy().reshape(-1)
    df["GD_prediction"] = y_pred

    df.to_csv(conf["result_ivr_law"], index = False)



