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
import torch
import sys
import argparse

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.nuts import HMC
from pyro.infer import EmpiricalMarginal
from pyro.infer import MCMC, NUTS

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import preprocess_dataset
from utils.helpers import features_setting
from model_arch.discriminator import train_law
from sklearn.ensemble import GradientBoostingRegressor


def FalseModel_Level2():
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
    G = pyro.sample("UGPA", dist.Normal(10 * K + R + S, 1))
    L = pyro.sample("LSAT", dist.Normal(G + K + R + S, 1))
    F = pyro.sample("ZFYA", dist.Normal(K + R + S, 0.1))

def FalseModel_Level3():
    """


    Returns
    -------
    None.

    """
    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Nknowledge1': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge2': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge3': dist.Normal(torch.tensor(0.), torch.tensor(1.))
    }

    R = pyro.sample("Race", exo_dist['Nrace'])
    S = pyro.sample("Sex", exo_dist['Nsex'])
    K1 = pyro.sample("Knowledge1", exo_dist['Nknowledge1'])
    K2 = pyro.sample("Knowledge2", exo_dist['Nknowledge2'])
    K3 = pyro.sample("Knowledge3", exo_dist['Nknowledge3'])

    # PsuedoDelta
    G = pyro.sample("UGPA", dist.Normal(10*K1 + R + S, 1))
    L = pyro.sample("LSAT", dist.Normal(6*K2 + G + R + S, 1))
    F = pyro.sample("ZFYA", dist.Normal(7*K3 + R + S, 0.1))

def GroundTruthModel_Level2():
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
    G = pyro.sample("UGPA", dist.Normal(10*K + R + S, 1))
    L = pyro.sample("LSAT", dist.Normal(K + R + S, 1))
    F = pyro.sample("ZFYA", dist.Normal(K + R + S, 0.1))

def GroundTruthModel_Level3():
    """


    Returns
    -------
    None.

    """
    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Nknowledge1': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge2': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge3': dist.Normal(torch.tensor(0.), torch.tensor(1.))
    }

    R = pyro.sample("Race", exo_dist['Nrace'])
    S = pyro.sample("Sex", exo_dist['Nsex'])
    K1 = pyro.sample("Knowledge1", exo_dist['Nknowledge1'])
    K2 = pyro.sample("Knowledge2", exo_dist['Nknowledge2'])
    K3 = pyro.sample("Knowledge3", exo_dist['Nknowledge3'])

    # PsuedoDelta
    G = pyro.sample("UGPA", dist.Normal(10*K1 + 30*R + S, 1))
    L = pyro.sample("LSAT", dist.Normal(6*K2 + R + S + G, 1))
    F = pyro.sample("ZFYA", dist.Normal(7*K3 + R + S, 0.1))

def infer_knowledge_level2(df, flag=True, monte=True):
    """
    
    :param df: DESCRIPTION
    :type df: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """
    
    knowledge = []

    # df = df.sample(frac=0.01, replace=True, random_state=1).reset_index(drop=True)

    for i in tqdm(range(len(df))):
        if flag:
            conditioned = pyro.condition(GroundTruthModel_Level2, data={"UGPA": df["UGPA"][i],                                                     "LSAT": df["LSAT"][i]})
        else:
            conditioned = pyro.condition(FalseModel_Level2, data={"UGPA": df["UGPA"][i],
                                                                 "LSAT": df["LSAT"][i]})
        if monte:
            hmc_kernel = HMC(conditioned, step_size=1.9, num_steps=1)
            posterior = MCMC(hmc_kernel,
                             num_samples=100,
                             warmup_steps=10)
            posterior.run()
            post_samples = posterior.get_samples(100)['Knowledge'].detach().numpy()
        else:
            posterior = pyro.infer.Importance(conditioned, num_samples=100).run()
            post_marginal = pyro.infer.EmpiricalMarginal(posterior, "Knowledge")
            post_samples = [post_marginal().item() for _ in range(1)]

        mean = np.mean(post_samples)
        knowledge.append(mean)

    knowledge = np.array(knowledge).reshape(-1, 1)
    # knowledge1 = np.array(knowledge1).reshape(-1, 1)
    # knowledge2 = np.array(knowledge2).reshape(-1, 1)
    # knowledge3 = np.array(knowledge3).reshape(-1, 1)
    ugpa = df['UGPA'].values.reshape(-1, 1)

    features = np.concatenate((knowledge, ugpa), axis=1)

    return features

def infer_knowledge_level3(df, flag=True, monte=True):
    """

    :param df: DESCRIPTION
    :type df: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """

    knowledge1 = []
    knowledge2 = []
    knowledge3 = []

    # df = df.sample(frac=0.01, replace=True, random_state=1).reset_index(drop=True)

    for i in tqdm(range(len(df))):
        if flag:
            conditioned = pyro.condition(GroundTruthModel_Level3, data={"UGPA": df["UGPA"][i],
                                                                 "LSAT": df["LSAT"][i]})
        else:
            conditioned = pyro.condition(FalseModel_Level3, data={"UGPA": df["UGPA"][i],
                                                                 "LSAT": df["LSAT"][i]})
        if monte:
            hmc_kernel = HMC(conditioned, step_size=0.9, num_steps=10)
            posterior = MCMC(hmc_kernel,
                             num_samples=10,
                             warmup_steps=10)
            posterior.run()

            post_samples1 = posterior.get_samples(100)['Knowledge1'].detach().numpy()
            post_samples2 = posterior.get_samples(100)['Knowledge2'].detach().numpy()
            post_samples3 = posterior.get_samples(100)['Knowledge3'].detach().numpy()
        else:
            posterior = pyro.infer.Importance(conditioned, num_samples=100).run()

            post_marginal_1 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge1")
            post_marginal_2 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge2")
            post_marginal_3 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge3")

            post_samples1 = [post_marginal_1().item() for _ in range(100)]
            post_samples2 = [post_marginal_2().item() for _ in range(100)]
            post_samples3 = [post_marginal_3().item() for _ in range(100)]

        mean1 = np.mean(post_samples1)
        mean2 = np.mean(post_samples2)
        mean3 = np.mean(post_samples3)

        knowledge1.append(mean1)
        knowledge2.append(mean2)
        knowledge3.append(mean3)

    knowledge1 = np.array(knowledge1).reshape(-1, 1)
    knowledge2 = np.array(knowledge2).reshape(-1, 1)
    knowledge3 = np.array(knowledge3).reshape(-1, 1)

    ugpa = df['UGPA'].values.reshape(-1, 1)

    features = np.concatenate((knowledge1, knowledge2, knowledge3, ugpa), axis=1)

    return features


# def train(df_train, df_test, flag, monte, level):
#     if level == 2:
#         knowledged_train = infer_knowledge_level2(df_train, flag, monte)
#         knowledged_test = infer_knowledge_level2(df_test, flag, monte)
#     elif level == 3:
#         knowledged_train = infer_knowledge_level3(df_train, flag, monte)
#         knowledged_test = infer_knowledge_level3(df_test, flag, monte)
#
#     reg = GradientBoostingRegressor(random_state=0)
#     reg.fit(knowledged_train, df_train['ZFYA'])
#     # reg = LinearRegression().fit(knowledged_train, df_train['ZFYA'])
#     train_x = torch.tensor(knowledged_train).float()
#     test_x = torch.tensor(knowledged_test).float()
#     train_y = torch.tensor(df_train['ZFYA'].values).float().reshape(-1, 1)
#     model = train_law(train_x, train_y)
#     model.eval()
#     y_pred = model(test_x)
#     y_pred1 = reg.predict(knowledged_test)
#     y_pred2 = y_pred.detach().numpy()
#
#     if monte:
#         df_test['level{}_mont_lin_{}'.format(level, str(flag))] = y_pred1
#         df_test['level{}_mont_net_{}'.format(level, str(flag))] = y_pred2
#     else:
#         df_test['level{}_lin_{}'.format(level, str(flag))] = y_pred1
#         df_test['level{}_lin_{}'.format(level, str(flag))] = y_pred2
#     return df_test


def train(knowledged_train, knowledged_test, y_train, level, flag):
    print("Shape ", knowledged_train.shape, y_train.shape)
    reg = GradientBoostingRegressor(random_state=0)
    reg.fit(knowledged_train, y_train)

    y_pred = reg.predict(knowledged_test)
    df_test['level{}_lin_{}'.format(level, str(flag))] = y_pred

    return df_test

    # if flag:
    #     if monte:
    #
    #         df_test['level2_False_MCMC_linear'] = reg.predict(knowledged_test)

# def infer_knowledge_level2_MCMC(df, flag):
#     """
#
#     :param df: DESCRIPTION
#     :type df: TYPE
#     :return: DESCRIPTION
#     :rtype: TYPE
#     """
#
#     knowledge = []
#     for i in tqdm(range(len(df))):
#         if flag:
#             conditioned = pyro.condition(GroundTruthModel_Level2, data={"UGPA": df["UGPA"][i],
#                                                              "LSAT": df["LSAT"][i]})
#         else:
#             conditioned = pyro.condition(FalseModel_Level2, data={"UGPA": df["UGPA"][i],
#                                                              "LSAT": df["LSAT"][i]})
#         hmc_kernel = HMC(conditioned, step_size=1.9, num_steps=1)
#         posterior = MCMC(hmc_kernel,
#                          num_samples=1,
#                          warmup_steps=1)
#         posterior.run()
#         post_samples = posterior.get_samples(100)['Knowledge'].detach().numpy()
#         mean = np.mean(post_samples)
#         knowledge.append(mean)
#     knowledge = np.array(knowledge).reshape(-1, 1)
#     return knowledge
#
# def infer_knowledge_level3_MCMC(df, flag):
#     """
#
#     :param df: DESCRIPTION
#     :type df: TYPE
#     :return: DESCRIPTION
#     :rtype: TYPE
#     """
#
#     knowledge1 = []
#     knowledge2 = []
#     knowledge3 = []
#
#     # df = df.sample(frac=0.01, replace=True, random_state=1).reset_index(drop=True)
#
#     for i in tqdm(range(len(df))):
#         conditioned = pyro.condition(causal_model, data={"UGPA": df["UGPA"][i],
#                                                              "LSAT": df["LSAT"][i]})
#         hmc_kernel = HMC(conditioned, step_size=1.9, num_steps=1)
#         posterior = MCMC(hmc_kernel,
#                          num_samples=1,
#                          warmup_steps=1)
#         posterior.run()
#
#         post_samples1 = posterior.get_samples(100)['Knowledge1'].detach().numpy()
#         post_samples2 = posterior.get_samples(100)['Knowledge2'].detach().numpy()
#         post_samples3 = posterior.get_samples(100)['Knowledge3'].detach().numpy()
#
#         mean1 = np.mean(post_samples1)
#         mean2 = np.mean(post_samples2)
#         mean3 = np.mean(post_samples3)
#
#         knowledge1.append(mean1)
#         knowledge2.append(mean2)
#         knowledge3.append(mean3)
#
#     knowledge1 = np.array(knowledge1).reshape(-1, 1)
#     knowledge2 = np.array(knowledge2).reshape(-1, 1)
#     knowledge3 = np.array(knowledge3).reshape(-1, 1)
#     knowledged = np.concatenate((knowledge1, knowledge2, knowledge3), axis=1)
#
#     return knowledged

if __name__ == "__main__":
    """Load configuration"""
    config_path = "/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml"
    conf = load_config(config_path)

    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true')

    args = parser.parse_args()

    generate = args.generate

    """Set up logging"""
    logger = setup_logging(conf['log_baselines_law'])

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

    # le = preprocessing.LabelEncoder()
    # df['race'] = le.fit_transform(df['race'])
    # df['sex'] = le.fit_transform(df['sex'])
    # df['race']  = df['race'].astype(float)
    # df['sex'] = df['sex'].astype(float)

    # df, df_test = train_test_split(df, test_size=0.99, random_state=0)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    """Full model"""
    # logger.debug('Full model')

    """Full model"""
    # reg = LinearRegression().fit(df_train[full_features], df_train['ZFYA'])
    # y_pred = reg.predict(df_test[full_features].values)
    # df_test['full_linear'] = y_pred.reshape(-1)
    #
    # train_x, train_y = df_train[full_features].values, df_train['ZFYA'].values
    # test_x, _ = df_test[full_features].values, df_test['ZFYA'].values
    # train_x = torch.tensor(train_x).float()
    # test_x = torch.tensor(test_x).float()
    # train_y = torch.tensor(train_y).float().reshape(-1, 1)
    # model = train_law(train_x, train_y)
    # model.eval()
    # y_pred = model(test_x)
    # df_test['full_net'] = y_pred.detach().numpy()

    """Unaware model"""
    # logger.debug('Unware model')
    # reg = LinearRegression().fit(df_train[normal_features], df_train['ZFYA'])
    # y_pred = reg.predict(df_test[normal_features].values)
    # df_test['unaware_linear'] = y_pred.reshape(-1)
    # train_x, train_y = df_train[normal_features].values, df_train['ZFYA'].values
    # test_x, _ = df_test[normal_features].values, df_test['ZFYA'].values
    # train_x = torch.tensor(train_x).float()
    # test_x = torch.tensor(test_x).float()
    # train_y = torch.tensor(train_y).float().reshape(-1, 1)
    # model = train_law(train_x, train_y)
    # model.eval()
    # y_pred = model(test_x)
    # df_test['unaware_net'] = y_pred.detach().numpy()

    """Counterfactual fairness model"""
    # for i in ['LSAT', 'UGPA', 'race', 'sex']:
    #     df_train[i] = [torch.tensor(x) for x in df_train[i].values]
    #     df_test[i] = [torch.tensor(x) for x in df_test[i].values]

    if generate:
        print("Generating features.....................")
        for i in ['LSAT', 'UGPA', 'race', 'sex']:
            df_train[i] = [torch.tensor(x) for x in df_train[i].values]
            df_test[i] = [torch.tensor(x) for x in df_test[i].values]

        print("True causal model")
        flag, monte, level = True, False, 2
        knowledged_train = infer_knowledge_level2(df_train, flag, monte)
        knowledged_test = infer_knowledge_level2(df_test, flag, monte)
        np.save(conf['law_train2_true'], knowledged_train)
        np.save(conf['law_test2_true'], knowledged_test)

        flag, monte, level = True, False, 3
        knowledged_train = infer_knowledge_level3(df_train, flag, monte)
        knowledged_test = infer_knowledge_level3(df_test, flag, monte)
        np.save(conf['law_train3_true'], knowledged_train)
        np.save(conf['law_test3_true'], knowledged_test)

        print("False causal model")

        flag, monte, level = False, False, 2
        knowledged_train = infer_knowledge_level2(df_train, flag, monte)
        knowledged_test = infer_knowledge_level2(df_test, flag, monte)
        np.save(conf['law_train2_false'], knowledged_train)
        np.save(conf['law_test2_false'], knowledged_test)

        flag, monte, level = False, False, 3
        knowledged_train = infer_knowledge_level3(df_train, flag, monte)
        knowledged_test = infer_knowledge_level3(df_test, flag, monte)
        np.save(conf['law_train3_false'], knowledged_train)
        np.save(conf['law_test3_false'], knowledged_test)

        sys.exit(0)


    """Full model"""
    reg = LinearRegression().fit(df_train[full_features], df_train['ZFYA'])
    y_pred = reg.predict(df_test[full_features].values)
    df_test['full_linear'] = y_pred.reshape(-1)

    train_x, train_y = df_train[full_features].values, df_train['ZFYA'].values
    test_x, _ = df_test[full_features].values, df_test['ZFYA'].values
    train_x = torch.tensor(train_x).float()
    test_x = torch.tensor(test_x).float()
    train_y = torch.tensor(train_y).float().reshape(-1, 1)
    # model = train_law(train_x, train_y)
    # model.eval()
    # y_pred = model(test_x)
    # df_test['full_net'] = y_pred.detach().numpy()

    """Unaware model"""
    logger.debug('Unware model')
    reg = LinearRegression().fit(df_train[normal_features], df_train['ZFYA'])
    y_pred = reg.predict(df_test[normal_features].values)
    df_test['unaware_linear'] = y_pred.reshape(-1)
    train_x, train_y = df_train[normal_features].values, df_train['ZFYA'].values
    test_x, _ = df_test[normal_features].values, df_test['ZFYA'].values
    train_x = torch.tensor(train_x).float()
    test_x = torch.tensor(test_x).float()
    train_y = torch.tensor(train_y).float().reshape(-1, 1)
    # model = train_law(train_x, train_y)
    # model.eval()
    # y_pred = model(test_x)
    # df_test['unaware_net'] = y_pred.detach().numpy()

    print("True level 2 model")
    flag, monte, level = True, False, 2
    knowledged_train = np.load(conf['law_train2_true'],allow_pickle=True)
    knowledged_test = np.load(conf['law_test2_true'], allow_pickle=True)
    df_test = train(knowledged_train, knowledged_test, df_train['ZFYA'].values, level, flag)

    print("False level 2 model")
    flag, monte, level = False, False, 2
    knowledged_train = np.load(conf['law_train2_false'],allow_pickle=True)
    knowledged_test = np.load(conf['law_test2_false'],allow_pickle=True)
    print(knowledged_train.shape, knowledged_test.shape, df_train.shape)
    df_test = train(knowledged_train, knowledged_test, df_train['ZFYA'].values, level, flag)

    print("True level 3 model")
    flag, monte, level = True, False, 3
    knowledged_train = np.load(conf['law_train3_true'],allow_pickle=True)
    knowledged_test = np.load(conf['law_test3_true'],allow_pickle=True)
    df_test = train(knowledged_train, knowledged_test, df_train['ZFYA'].values, level, flag)

    print("False level 2 model")
    flag, monte, level = False, False, 3

    knowledged_train = np.load(conf['law_train3_true'],allow_pickle=True)
    knowledged_test = np.load(conf['law_test3_false'],allow_pickle=True)
    # print(knowledged_train.shape, knowledged_test.shape, df_train.shape)

    df_test = train(knowledged_train, knowledged_test, df_train['ZFYA'].values, level, flag)

    """Output the result"""
    df_test.to_csv(conf['law_baseline'], index = False)
    sys.modules[__name__].__dict__.clear()

    #
    # knowledged_train = np.load(path_train2)
    # knowledged_test = np.load(path_test2)
    # df_test = train(knowledged_train, knowledged_test, y_train=df_train['ZFYA'].values, level=2)
    #
    # knowledged_train = np.load(path_train3)
    # knowledged_test = np.load(path_test3)
    # df_test = train(knowledged_train, knowledged_test, y_train=df_train['ZFYA'].values, level=3)


    # knowledged_train = infer_knowledge_level2(df_train, flag, monte)
    # knowledged_test = infer_knowledge_level2(df_test, flag, monte)
    # np.save(path_train2, knowledged_train)
    # np.save(path_test2, knowledged_test)

    # flag, monte = True, False
    # knowledged_train = infer_knowledge_level3(df_train, flag, monte)
    # knowledged_test = infer_knowledge_level3(df_test, flag, monte)
    # np.save("/home/trduong/Data/counterfactual_fairness_game_theoric/data/knowledged_train_level3.npy", knowledged_train)
    # np.save("/home/trduong/Data/counterfactual_fairness_game_theoric/data/knowledged_test_level3.npy", knowledged_test)


    # flag, monte, level = True, False, 2
    # df_test = train(df_train, df_test, flag, monte, level)

    # flag, monte, level = True, True, 2
    # df_test = train(df_train, df_test, flag, monte, level)

    # flag, monte, level = False, True, 2
    # df_test = train(df_train, df_test, flag, monte, level)

    # logger.debug("SCM 1")
    # flag, monte, level = False, False, 2
    # df_test = train(df_train, df_test, flag, monte, level)

    # logger.debug("SCM 2")
    # flag, monte, level = True, False, 3
    # df_test = train(df_train, df_test, flag, monte, level)

    # flag, monte, level = True, True, 3
    # df_test = train(df_train, df_test, flag, monte, level)
    # flag, monte, level = False, True, 3
    # df_test = train(df_train, df_test, flag, monte, level)

    # logger.debug("SCM 3")
    # flag, monte, level = False, False, 3
    # df_test = train(df_train, df_test, flag, monte, level)







    """
    False structural causal model level 2
    """
    # logger.debug('Counterfactual fairness model level 2 False Monte Carlo')
    # flag = False
    # monte = False

    # knowledged_train = infer_knowledge_level2(df, flag, monte)
    # reg = LinearRegression().fit(knowledged_train, df['ZFYA'])
    # knowledged_test = infer_knowledge_level2(df_test)
    # df_test['level2_False_MCMC_linear'] = reg.predict(knowledged_test)
    #
    # knowledged_train = torch.tensor(knowledged_train).float()
    # knowledged_test = torch.tensor(knowledged_test).float()
    # train_y = torch.tensor(df['ZFYA'].values).float().reshape(-1, 1)
    # model = train_law(train_x, train_y)
    # model.eval()
    # y_pred = model(knowledged_test)
    # df_test['level2_False_MCMC_net'] = y_pred.detach().numpy()


    # logger.debug('Counterfactual fairness model level 2 False Monte Carlo')
    # knowledged_train = infer_knowledge_level2(df, flag)
    # reg = LinearRegression().fit(knowledged, df['ZFYA'])
    # knowledged_test = infer_knowledge_level2(df_test)
    # df_test['level2_False_Importance_linear'] = reg.predict(knowledged)
    #
    # knowledged_train = torch.tensor(knowledged_train).float()
    # knowledged_test = torch.tensor(knowledged_test).float()
    # train_y = torch.tensor(df['ZFYA'].values).float().reshape(-1, 1)
    # model = train_law(train_x, train_y)
    # model.eval()
    # y_pred = model(knowledged_test)
    # df_test['level2_False_Importance_net'] = y_pred.detach().numpy()

    """
    False structural causal model level 3 
    """
    # flag = False
    # logger.debug('Counterfactual fairness model level 3 Monte Carlo')
    # knowledged = infer_knowledge_level3_MCMC(df, flag)
    # reg = LinearRegression().fit(knowledged, df['ZFYA'])
    # knowledged = infer_knowledge_level3(df_test)
    # df_test['level3_MCMC'] = reg.predict(knowledged)
    #
    # logger.debug('Counterfactual fairness model level 3 Importance Sampling')
    # knowledged = infer_knowledge_level3(df, flag)
    # reg = LinearRegression().fit(knowledged, df['ZFYA'])
    # knowledged = infer_knowledge_level3(df_test)
    # df_test['level3_Importance'] = reg.predict(knowledged)

    """
    Structural causal model level 2 
    """
    # flag = True
    #
    # logger.debug('Counterfactual fairness model level 2 Monte Carlo')
    # knowledged = infer_knowledge_level2_MCMC(df, flag)
    # reg = LinearRegression().fit(knowledged, df['ZFYA'])
    # knowledged = infer_knowledge_level2(df_test)
    # df_test['level2_MCMC'] = reg.predict(knowledged)
    #
    # logger.debug('Counterfactual fairness model level 2 Importance Sampling')
    # knowledged = infer_knowledge_level2(df, flag)
    # reg = LinearRegression().fit(knowledged, df['ZFYA'])
    # knowledged = infer_knowledge_level2(df_test)
    # df_test['level2_Importance'] = reg.predict(knowledged)

    """
    Structural causal model level 3 
    """
    # logger.debug('Counterfactual fairness model level 3 Monte Carlo')
    # knowledged = infer_knowledge_level3_MCMC(df, flag)
    # reg = LinearRegression().fit(knowledged, df['ZFYA'])
    # knowledged = infer_knowledge_level3(df_test)
    # df_test['level3_MCMC'] = reg.predict(knowledged)
    #
    # logger.debug('Counterfactual fairness model level 3 Importance Sampling')
    # knowledged = infer_knowledge_level3(df, flag)
    # reg = LinearRegression().fit(knowledged, df['ZFYA'])
    # knowledged = infer_knowledge_level3(df_test)
    # df_test['level3_Importance'] = reg.predict(knowledged)

    # df_test['LSAT'] = [x.detach().numpy() for x in df_test['LSAT']]
    # df_test['UGPA'] = [x.detach().numpy() for x in df_test['UGPA']]
    # df_test['race'] = [x.detach().numpy() for x in df_test['race']]
    # df_test['sex'] = [x.detach().numpy() for x in df_test['sex']]

    """Output the result"""
    # df_test.to_csv(conf['law_baseline'], index = False)
    # sys.modules[__name__].__dict__.clear()