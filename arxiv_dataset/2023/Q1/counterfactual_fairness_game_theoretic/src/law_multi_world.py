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
import torch.nn.functional as F
import visdom


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

    df, df_test = train_test_split(df, test_size=0.1, random_state=0)

    """Setup features"""
    data_name = "law"
    dict_ = features_setting("law")
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    target = dict_["target"]
    col_sensitive = ['race_0', 'race_1', 'sex_0', 'sex_1']

    selected_race = ['White', 'Black']
    df = df[df['race'].isin(selected_race)]
    df = df.reset_index(drop=True)

    """Preprocess data"""
    df = preprocess_dataset(df, continuous_features, categorical_features)
    df['ZFYA'] = (df['ZFYA'] - df['ZFYA'].mean()) / df['ZFYA'].std()
    df = df[['LSAT', 'UGPA', 'sex', 'race', 'ZFYA']]

    """Setup hyperparameter"""
    logger.debug('Setup hyperparameter')
    parameters = {}
    parameters['epochs'] = 100
    parameters['learning_rate'] = 1e-2
    parameters['dataframe'] = df
    parameters['batch_size'] = 128
    parameters['problem'] = 'regression'

    """Hyperparameter"""
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    dataframe = parameters['dataframe']
    batch_size = parameters['batch_size']
    problem = parameters['problem']

    """Setup generator and discriminator"""
    emb_size = 64
    discriminator_agnostic = DiscriminatorLaw(emb_size, problem)
    discriminator_agnostic.to(device)
    learning_rate = 1e-6
    """Optimizer"""
    optimizer2 = torch.optim.SGD(discriminator_agnostic.parameters(), lr=learning_rate, momentum=0.9)
    scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer2, base_lr=learning_rate, max_lr=0.0001)

    """Training"""
    n_updates = len(df) // batch_size
    logger.debug('Training')
    logger.debug('Number of updates {}'.format(n_updates))
    logger.debug('Dataframe length {}'.format(len(df)))
    logger.debug('Batchsize {}'.format((batch_size)))

    loss_function = torch.nn.SmoothL1Loss()

    step = 0
    losses = []
    losses_aware = []
    losses_gen = []
    for i in (range(epochs)):


        loss_model_1 = F.relu()
        loss_model_1 = F.relu()
        final_loss = loss_function(y_pred, y) + 0.1* loss_model_1 + 0.1*loss_model_2

        optimizer2.zero_grad()
        final_loss.backward()
        optimizer2.step()


