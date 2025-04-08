#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:55:24 2021
@author: trduong
"""

import pandas as pd
import numpy as np
import logging
import pyro
import torch
import pyro.distributions as dist
import sys
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
from utils.helpers import setup_logging
from utils.helpers import load_config
from utils.helpers import features_setting


# def GroundTruthModel_Adult():
#     exo_dist = {
#         'Nrace': dist.Bernoulli(torch.tensor(0.75)),
#         'Nsex': dist.Bernoulli(torch.tensor(0.5)),
#         'Nage': dist.Normal(torch.tensor(0.), torch.tensor(10.)),
#         'Nknowledge1': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
#         'Nknowledge2': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
#         'Nknowledge3': dist.Normal(torch.tensor(0.), torch.tensor(1.))
#
#     }
#
#     S = pyro.sample("Sex", exo_dist['Nsex'])
#     R = pyro.sample("Race", exo_dist['Nrace'])
#     A = pyro.sample("Age", exo_dist['Nage'])
#     K1 = pyro.sample("Knowledge1", exo_dist['Nknowledge1'])
#     K2 = pyro.sample("Knowledge2", exo_dist['Nknowledge2'])
#     K3 = pyro.sample("Knowledge3", exo_dist['Nknowledge3'])
#
#     M = pyro.sample("Martial", dist.Poisson(torch.exp(R+S+A)))
#     E = pyro.sample("Education", dist.Poisson(torch.exp(R+S+A+K1+K2)))
#     H = pyro.sample("Hour", dist.Normal(R+S+A+K1+K2+K3+E, torch.tensor(10.)))
#     O = pyro.sample("Occupation", dist.Poisson(torch.exp(R+S+A+K1+K3)))
#     W = pyro.sample("Workclass", dist.Poisson(torch.exp(R+S+A+K2+K3)))

def GroundTruthModel_Adult():
    count_dict = {'marital_status': 5,
                  'occupation': 6,
                  'race': 2,
                  'gender': 2,
                  'workclass': 4,
                  'education': 8}

    prob_education = torch.tensor([1 / count_dict['education']] * count_dict['education'])
    prob_occupation = torch.tensor([1 / count_dict['occupation']] * count_dict['occupation'])
    prob_maritalstatus = torch.tensor([1 / count_dict['marital_status']] * count_dict['marital_status'])

    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Neducation': dist.Categorical(probs=prob_education),
        'Nmarital_status': dist.Categorical(probs=prob_maritalstatus),
        'Noccupation': dist.Categorical(probs=prob_occupation),
        'Nknowledge1': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge2': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge3': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge4': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nage': dist.Normal(torch.tensor(0.), torch.tensor(1.))
    }

    R = pyro.sample("Race", exo_dist['Nrace'])
    S = pyro.sample("Sex", exo_dist['Nsex'])
    K1 = pyro.sample("Knowledge1", exo_dist['Nknowledge1'])
    K2 = pyro.sample("Knowledge2", exo_dist['Nknowledge2'])
    K3 = pyro.sample("Knowledge3", exo_dist['Nknowledge3'])
    K4 = pyro.sample("Knowledge4", exo_dist['Nknowledge4'])
    A = pyro.sample("Age", exo_dist['Nage'])

    mu_E = R + S + K1 + K2 + K3 + K4 + A
    x1, x2, x3, x4, x5, x6, x7 = mu_E - 3, mu_E - 2, mu_E - 1, mu_E, mu_E + 1, mu_E + 2, mu_E + 3
    E = pyro.sample("Education", dist.Normal(mu_E, 1))
    if (E >= x7) or (E <= x1):
        E = np.random.choice([2,5])
    elif (E>= x1 and E <= x2) or (E>= x6 and E <= x7):
        E = np.random.choice([0,6,4])
    elif (E>= x2 and E <= x3) or (E>= x5 and E <= x6):
        E = np.random.choice([1,7])
    elif (E >= x3) and (E <= x5):
        E = np.random.choice([3])
    E = torch.tensor(E)

    mu_M = R + S + E + K1 + K2 + K3 + K4 + A
    x1, x2, x3, x4, x5, x6, x7 = mu_M - 3, mu_M - 2, mu_M - 1, mu_M, mu_M + 1, mu_M + 2, mu_M + 3
    M = pyro.sample("Marital", dist.Normal(mu_M, 1))
    if (M >= x7) or (M <= x1):
        M = 4
    elif (M>= x1 and M <= x2) or (M>= x6 and M <= x7):
        M = np.random.choice([0,2])
    elif (M>= x2 and M <= x3) or (M>= x5 and M <= x6):
        M = 3
    elif (M >= x3) and (M <= x5):
        M = 1
    M = torch.tensor(M)

    mu_O = R + S + E + M + K1 + K2 + K3 + K4
    x1, x2, x3, x4, x5, x6, x7 = mu_O - 3, mu_O - 2, mu_O - 1, mu_O, mu_O + 1, mu_O + 2, mu_O + 3
    O = pyro.sample("Occupation", dist.Normal(mu_O, 1))
    if (O >= x7) or (O <= x1):
        O = 1
    elif (O>= x1 and O <= x2) or (O>= x6 and O <= x7):
        O = np.random.choice([2,3])
    elif (O>= x2 and O <= x3) or (O>= x5 and O <= x6):
        O = np.random.choice([4,5])
    elif (M >= x3) and (O <= x5):
        O = 0
    O = torch.tensor(O)

    mu_H = R + S + E + M + O + A + K1 + K2 + K3 + K4
    H = pyro.sample("Hour", dist.Normal(mu_H, 1))

def GroundTruthModel1_Adult():
    count_dict = {'marital_status': 5,
                  'occupation': 6,
                  'race': 2,
                  'gender': 2,
                  'workclass': 4,
                  'education': 8}

    prob_education = torch.tensor([1 / count_dict['education']] * count_dict['education'])
    prob_occupation = torch.tensor([1 / count_dict['occupation']] * count_dict['occupation'])
    prob_maritalstatus = torch.tensor([1 / count_dict['marital_status']] * count_dict['marital_status'])

    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Neducation': dist.Categorical(probs=prob_education),
        'Nmarital_status': dist.Categorical(probs=prob_maritalstatus),
        'Noccupation': dist.Categorical(probs=prob_occupation),
        'Nknowledge1': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge2': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge3': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge4': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nage': dist.Normal(torch.tensor(0.), torch.tensor(1.))
    }

    R = pyro.sample("Race", exo_dist['Nrace'])
    S = pyro.sample("Sex", exo_dist['Nsex'])
    K1 = pyro.sample("Knowledge1", exo_dist['Nknowledge1'])
    K2 = pyro.sample("Knowledge2", exo_dist['Nknowledge2'])
    K3 = pyro.sample("Knowledge3", exo_dist['Nknowledge3'])
    K4 = pyro.sample("Knowledge4", exo_dist['Nknowledge4'])
    A = pyro.sample("Age", exo_dist['Nage'])

    mu_E = R + S + K1 + K2 + K3 + K4 + A
    x1, x2, x3, x4, x5, x6, x7 = mu_E - 3, mu_E - 2, mu_E - 1, mu_E, mu_E + 1, mu_E + 2, mu_E + 3
    E = pyro.sample("Education", dist.Normal(mu_E, 1))
    if (E >= x7) or (E <= x1):
        E = np.random.choice([2,5])
    elif (E>= x1 and E <= x2) or (E>= x6 and E <= x7):
        E = np.random.choice([0,6,4])
    elif (E>= x2 and E <= x3) or (E>= x5 and E <= x6):
        E = np.random.choice([1,7])
    elif (E >= x3) and (E <= x5):
        E = np.random.choice([3])
    E = torch.tensor(E)

    mu_M = R + S + E + K1 + K2 + K3 + K4 + A
    x1, x2, x3, x4, x5, x6, x7 = mu_M - 3, mu_M - 2, mu_M - 1, mu_M, mu_M + 1, mu_M + 2, mu_M + 3
    M = pyro.sample("Marital", dist.Normal(mu_M, 1))
    if (M >= x7) or (M <= x1):
        M = 4
    elif (M>= x1 and M <= x2) or (M>= x6 and M <= x7):
        M = np.random.choice([0,2])
    elif (M>= x2 and M <= x3) or (M>= x5 and M <= x6):
        M = 3
    elif (M >= x3) and (M <= x5):
        M = 1
    M = torch.tensor(M)

    mu_O = R + S + E + M + K1 + K2 + K3 + K4
    x1, x2, x3, x4, x5, x6, x7 = mu_O - 3, mu_O - 2, mu_O - 1, mu_O, mu_O + 1, mu_O + 2, mu_O + 3
    O = pyro.sample("Occupation", dist.Normal(mu_O, 1))
    if (O >= x7) or (O <= x1):
        O = 1
    elif (O>= x1 and O <= x2) or (O>= x6 and O <= x7):
        O = np.random.choice([2,3])
    elif (O>= x2 and O <= x3) or (O>= x5 and O <= x6):
        O = np.random.choice([4,5])
    elif (M >= x3) and (O <= x5):
        O = 0
    O = torch.tensor(O)

    mu_H = R + S + E + M + O + A + K1 + K2 + K3 + K4
    H = pyro.sample("Hour", dist.Normal(mu_H, 1))

def GroundTruthModel2_Adult():
    count_dict = {'marital_status': 5,
                  'occupation': 6,
                  'race': 2,
                  'gender': 2,
                  'workclass': 4,
                  'education': 8}

    prob_education = torch.tensor([1 / count_dict['education']] * count_dict['education'])
    prob_occupation = torch.tensor([1 / count_dict['occupation']] * count_dict['occupation'])
    prob_maritalstatus = torch.tensor([1 / count_dict['marital_status']] * count_dict['marital_status'])

    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Neducation': dist.Categorical(probs=prob_education),
        'Nmarital_status': dist.Categorical(probs=prob_maritalstatus),
        'Noccupation': dist.Categorical(probs=prob_occupation),
        'Nknowledge1': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge2': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge3': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge4': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nage': dist.Normal(torch.tensor(0.), torch.tensor(1.))
    }

    R = pyro.sample("Race", exo_dist['Nrace'])
    S = pyro.sample("Sex", exo_dist['Nsex'])
    K1 = pyro.sample("Knowledge1", exo_dist['Nknowledge1'])
    K2 = pyro.sample("Knowledge2", exo_dist['Nknowledge2'])
    K3 = pyro.sample("Knowledge3", exo_dist['Nknowledge3'])
    K4 = pyro.sample("Knowledge4", exo_dist['Nknowledge4'])
    A = pyro.sample("Age", exo_dist['Nage'])

    mu_E = R + S + K3 + K4 + A
    x1, x2, x3, x4, x5, x6, x7 = mu_E - 3, mu_E - 2, mu_E - 1, mu_E, mu_E + 1, mu_E + 2, mu_E + 3
    E = pyro.sample("Education", dist.Normal(mu_E, 1))
    if (E >= x7) or (E <= x1):
        E = np.random.choice([2,5])
    elif (E>= x1 and E <= x2) or (E>= x6 and E <= x7):
        E = np.random.choice([0,6,4])
    elif (E>= x2 and E <= x3) or (E>= x5 and E <= x6):
        E = np.random.choice([1,7])
    elif (E >= x3) and (E <= x5):
        E = np.random.choice([3])
    E = torch.tensor(E)

    mu_M = R + S + E + K1 + K2
    x1, x2, x3, x4, x5, x6, x7 = mu_M - 3, mu_M - 2, mu_M - 1, mu_M, mu_M + 1, mu_M + 2, mu_M + 3
    M = pyro.sample("Marital", dist.Normal(mu_M, 1))
    if (M >= x7) or (M <= x1):
        M = 4
    elif (M>= x1 and M <= x2) or (M>= x6 and M <= x7):
        M = np.random.choice([0,2])
    elif (M>= x2 and M <= x3) or (M>= x5 and M <= x6):
        M = 3
    elif (M >= x3) and (M <= x5):
        M = 1
    M = torch.tensor(M)

    mu_O = R + S + E + M + K1 + K2 + K3
    x1, x2, x3, x4, x5, x6, x7 = mu_O - 3, mu_O - 2, mu_O - 1, mu_O, mu_O + 1, mu_O + 2, mu_O + 3
    O = pyro.sample("Occupation", dist.Normal(mu_O, 1))
    if (O >= x7) or (O <= x1):
        O = 1
    elif (O>= x1 and O <= x2) or (O>= x6 and O <= x7):
        O = np.random.choice([2,3])
    elif (O>= x2 and O <= x3) or (O>= x5 and O <= x6):
        O = np.random.choice([4,5])
    elif (M >= x3) and (O <= x5):
        O = 0
    O = torch.tensor(O)

    mu_H = R + S + E + M + O + A + K1 + K2 + K3 + K4
    H = pyro.sample("Hour", dist.Normal(mu_H, 1))

def GroundTruthModel3_Adult():
    count_dict = {'marital_status': 5,
                  'occupation': 6,
                  'race': 2,
                  'gender': 2,
                  'workclass': 4,
                  'education': 8}

    prob_education = torch.tensor([1 / count_dict['education']] * count_dict['education'])
    prob_occupation = torch.tensor([1 / count_dict['occupation']] * count_dict['occupation'])
    prob_maritalstatus = torch.tensor([1 / count_dict['marital_status']] * count_dict['marital_status'])

    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Neducation': dist.Categorical(probs=prob_education),
        'Nmarital_status': dist.Categorical(probs=prob_maritalstatus),
        'Noccupation': dist.Categorical(probs=prob_occupation),
        'Nknowledge1': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge2': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge3': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nknowledge4': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nage': dist.Normal(torch.tensor(0.), torch.tensor(1.))
    }

    R = pyro.sample("Race", exo_dist['Nrace'])
    S = pyro.sample("Sex", exo_dist['Nsex'])
    K1 = pyro.sample("Knowledge1", exo_dist['Nknowledge1'])
    K2 = pyro.sample("Knowledge2", exo_dist['Nknowledge2'])
    K3 = pyro.sample("Knowledge3", exo_dist['Nknowledge3'])
    K4 = pyro.sample("Knowledge4", exo_dist['Nknowledge4'])
    A = pyro.sample("Age", exo_dist['Nage'])

    mu_E = R + S + K3 + K4 + A
    x1, x2, x3, x4, x5, x6, x7 = mu_E - 3, mu_E - 2, mu_E - 1, mu_E, mu_E + 1, mu_E + 2, mu_E + 3
    E = pyro.sample("Education", dist.Normal(mu_E, 1))
    if (E >= x7) or (E <= x1):
        E = np.random.choice([2,5])
    elif (E>= x1 and E <= x2) or (E>= x6 and E <= x7):
        E = np.random.choice([0,6,4])
    elif (E>= x2 and E <= x3) or (E>= x5 and E <= x6):
        E = np.random.choice([1,7])
    elif (E >= x3) and (E <= x5):
        E = np.random.choice([3])
    E = torch.tensor(E)

    mu_M = R + S + E + K1 + K2
    x1, x2, x3, x4, x5, x6, x7 = mu_M - 3, mu_M - 2, mu_M - 1, mu_M, mu_M + 1, mu_M + 2, mu_M + 3
    M = pyro.sample("Marital", dist.Normal(mu_M, 1))
    if (M >= x7) or (M <= x1):
        M = 4
    elif (M>= x1 and M <= x2) or (M>= x6 and M <= x7):
        M = np.random.choice([0,2])
    elif (M>= x2 and M <= x3) or (M>= x5 and M <= x6):
        M = 3
    elif (M >= x3) and (M <= x5):
        M = 1
    M = torch.tensor(M)

    mu_O = R + S + E + M + K1 + K2 + K3
    x1, x2, x3, x4, x5, x6, x7 = mu_O - 3, mu_O - 2, mu_O - 1, mu_O, mu_O + 1, mu_O + 2, mu_O + 3
    O = pyro.sample("Occupation", dist.Normal(mu_O, 1))
    if (O >= x7) or (O <= x1):
        O = 1
    elif (O>= x1 and O <= x2) or (O>= x6 and O <= x7):
        O = np.random.choice([2,3])
    elif (O>= x2 and O <= x3) or (O>= x5 and O <= x6):
        O = np.random.choice([4,5])
    elif (M >= x3) and (O <= x5):
        O = 0
    O = torch.tensor(O)

    mu_H = R + S + E + M + O + A + K1 + K2 + K3 + K4
    H = pyro.sample("Hour", dist.Normal(mu_H, 1))

def infer_knowledge_adult(df, level=2):
    """

    :param df: DESCRIPTION
    :type df: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """

    knowledge1 = []
    knowledge2 = []
    knowledge3 = []
    hours = []

    for i in tqdm(range(len(df))):
        datadict = {
            "Education": df["education"][i],
            "Hour": df["hours_per_week"][i],
            "Marital": df["marital_status"][i],
            "Occupation": df["occupation"][i],
            "Age": df["age"][i]
        }
        if level == 2:
            conditioned = pyro.condition(GroundTruthModel_Adult, data=datadict)
        elif level == 3:
            conditioned = pyro.condition(GroundTruthModel1_Adult, data=datadict)
        elif level == 4:
            conditioned = pyro.condition(GroundTruthModel2_Adult, data=datadict)
        posterior = pyro.infer.Importance(conditioned, num_samples=10).run()

        post_marginal_1 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge1")
        post_marginal_2 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge2")
        post_marginal_3 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge3")

        post_samples1 = [post_marginal_1().item() for _ in range(10)]
        post_samples2 = [post_marginal_2().item() for _ in range(10)]
        post_samples3 = [post_marginal_3().item() for _ in range(10)]

        mean1 = np.mean(post_samples1)
        mean2 = np.mean(post_samples2)
        mean3 = np.mean(post_samples3)

        knowledge1.append(mean1)
        knowledge2.append(mean2)
        knowledge3.append(mean3)
        hours.append(df["hours_per_week"][i])

    knowledge1 = np.array(knowledge1).reshape(-1, 1)
    knowledge2 = np.array(knowledge2).reshape(-1, 1)
    knowledge3 = np.array(knowledge3).reshape(-1, 1)
    hours = np.array(hours).reshape(-1, 1)

    features = np.concatenate((knowledge1, knowledge2, knowledge3, hours), axis=1)
    return features


# def GroundTruthModel_Adult():
#     exo_dist = {
#         'Nrace': dist.Bernoulli(torch.tensor(0.75)),
#         'Nsex': dist.Bernoulli(torch.tensor(0.5)),
#         'Nage': dist.Normal(torch.tensor(0.), torch.tensor(10.)),
#         'Nknowledge1': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
#         'Nknowledge2': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
#         'Nknowledge3': dist.Normal(torch.tensor(0.), torch.tensor(1.))
#
#     }
#
#     S = pyro.sample("Sex", exo_dist['Nsex'])
#     R = pyro.sample("Race", exo_dist['Nrace'])
#     A = pyro.sample("Age", exo_dist['Nage'])
#     K1 = pyro.sample("Knowledge1", exo_dist['Nknowledge1'])
#     K2 = pyro.sample("Knowledge2", exo_dist['Nknowledge2'])
#     # K3 = pyro.sample("Knowledge3", exo_dist['Nknowledge3'])
#
#     # print("Sampling")
#     # print(S,R,A,K1,K2,K3)
#     M = pyro.sample("Martial", dist.Poisson(torch.exp(R+S+A)))
#     E = pyro.sample("Education", dist.Poisson(torch.exp(R+S+A)))
#     print("Sum ", R+S+A+K1+K2)
#     H = pyro.sample("Hour", dist.Normal(R+S+A+K1+K2, torch.tensor(10.)))
#     O = pyro.sample("Occupation", dist.Poisson(torch.exp(R+S+A+M+E)))
#     W = pyro.sample("Workclass", dist.Poisson(torch.exp(R+S+A+M+E)))

#
# def infer_knowledge_adult(df):
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
#     hours = []
#
#     for i in tqdm(range(len(df))):
#         print("Index ", i)
#         dict_ = {"Hour": df["hours_per_week"][i],
#                                                              "Age": df["age"][i],
#                                                              "Race": df["race"][i],
#                                                              "Sex": df["gender"][i],
#                                                              "Occupation": df["occupation"][i],
#                                                              "Martial": df["marital_status"][i],
#                                                              "Workclass": df['workclass']
#                                                              }
#
#         print(dict_)
#         conditioned = pyro.condition(GroundTruthModel_Adult, data={"Hour": df["hours_per_week"][i],
#                                                              "Age": df["age"][i],
#                                                              "Race": df["race"][i],
#                                                              "Sex": df["gender"][i],
#                                                              # "Occupation": df["occupation"][i],
#                                                              # "Martial": df["marital_status"][i],
#                                                              # "Workclass": df['workclass']
#                                                              }
#                                      )
#
#         posterior = pyro.infer.Importance(conditioned, num_samples=10).run()
#
#         post_marginal_1 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge1")
#         post_marginal_2 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge2")
#         # post_marginal_3 = pyro.infer.EmpiricalMarginal(posterior, "Knowledge3")
#
#         post_samples1 = [post_marginal_1().item() for _ in range(10)]
#         post_samples2 = [post_marginal_2().item() for _ in range(10)]
#         # post_samples3 = [post_marginal_3().item() for _ in range(10)]
#
#         mean1 = np.mean(post_samples1)
#         mean2 = np.mean(post_samples2)
#         mean3 = np.mean(post_samples3)
#
#         knowledge1.append(mean1)
#         knowledge2.append(mean2)
#         knowledge3.append(mean3)
#         hours.append(df["hours_per_week"][i])
#
#     knowledge1 = np.array(knowledge1).reshape(-1, 1)
#     knowledge2 = np.array(knowledge2).reshape(-1, 1)
#     knowledge3 = np.array(knowledge3).reshape(-1, 1)
#     hours = np.array(hours).reshape(-1, 1)
#
#     features = np.concatenate((knowledge1, knowledge2, knowledge3, hours), axis=1)
#     return features

def GroundTruthModel1_Compas():
    prob_age = torch.tensor([2/3, 1/6, 1/6])

    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Nage': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nagecat': dist.Categorical(probs=prob_age),
        'UJ': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'UD': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
    }

    S = pyro.sample("Sex", exo_dist['Nsex'])
    R = pyro.sample("Race", exo_dist['Nrace'])
    A = pyro.sample("Age", exo_dist['Nage'])
    AC = pyro.sample("Age_cat", exo_dist['Nagecat'])
    UJ = pyro.sample("UJ", exo_dist['UJ'])
    UD = pyro.sample("UD", exo_dist['UD'])

    JF = pyro.sample("JF", dist.Normal(R + S + A + AC + UJ, 1))
    JM = pyro.sample("JM", dist.Normal(R + S + A + AC + UJ, 1))
    JO = pyro.sample("JO", dist.Normal(R + S + A + AC + UD, 1))
    P = pyro.sample("P", dist.Normal(R + S + A + AC + UD, 1))

def GroundTruthModel2_Compas():
    prob_age = torch.tensor([2/3, 1/6, 1/6])

    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Nage': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nagecat': dist.Categorical(probs=prob_age),
        'UJ': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'UD': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
    }

    S = pyro.sample("Sex", exo_dist['Nsex'])
    R = pyro.sample("Race", exo_dist['Nrace'])
    A = pyro.sample("Age", exo_dist['Nage'])
    AC = pyro.sample("Age_cat", exo_dist['Nagecat'])
    UJ = pyro.sample("UJ", exo_dist['UJ'])
    UD = pyro.sample("UD", exo_dist['UD'])

    expF = torch.exp(R + S + A + AC + UJ)
    expM = torch.exp(R + S + A + AC + UJ)
    expO = torch.exp(R + S + A + 2*AC + UD)
    expP = torch.exp(R + S + A + 3*AC + UD)

    # print(expF, expM, expO)
    JF = pyro.sample("JF", dist.Poisson(expF))
    JM = pyro.sample("JM", dist.Poisson(expM))
    JO = pyro.sample("JO", dist.Poisson(expO))
    P = pyro.sample("P", dist.Poisson(expP))

def GroundTruthModel3_Compas():
    prob_age = torch.tensor([2/3, 1/6, 1/6])

    exo_dist = {
        'Nrace': dist.Bernoulli(torch.tensor(0.75)),
        'Nsex': dist.Bernoulli(torch.tensor(0.5)),
        'Nage': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'Nagecat': dist.Categorical(probs=prob_age),
        'UJ': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
        'UD': dist.Normal(torch.tensor(0.), torch.tensor(1.)),
    }

    S = pyro.sample("Sex", exo_dist['Nsex'])
    R = pyro.sample("Race", exo_dist['Nrace'])
    A = pyro.sample("Age", exo_dist['Nage'])
    AC = pyro.sample("Age_cat", exo_dist['Nagecat'])
    UJ = pyro.sample("UJ", exo_dist['UJ'])
    UD = pyro.sample("UD", exo_dist['UD'])

    expP = torch.exp(R + S + A + AC + UD)
    P = pyro.sample("P", dist.Poisson(expP))

    expF = torch.exp(R + S + A + 4*AC + UJ + P)
    JF = pyro.sample("JF", dist.Poisson(expF))

    expM = torch.exp(R + S + A + AC + UJ + P + JF)
    JM = pyro.sample("JM", dist.Poisson(expM))

    expO = torch.exp(R + S + A + AC + UD + P + JM)
    JO = pyro.sample("JO", dist.Poisson(expO))

def infer_knowledge_compas(df, model = 1):
    """

    :param df: DESCRIPTION
    :type df: TYPE
    :return: DESCRIPTION
    :rtype: TYPE
    """

    knowledge1 = []
    knowledge2 = []

    for i in tqdm(range(len(df))):
        if model == 1:

            conditioned = pyro.condition(GroundTruthModel1_Compas, data={
                                                                 "JF": df["juv_fel_count"][i],
                                                                 "JM": df["juv_misd_count"][i],
                                                                 "JO": df["juv_other_count"][i],
                                                                  "P": df["priors_count"][i]
                                                    }
                                         )
        elif model == 2:
            conditioned = pyro.condition(GroundTruthModel2_Compas, data={
                                                                 "JF": df["juv_fel_count"][i],
                                                                 "JM": df["juv_misd_count"][i],
                                                                 "JO": df["juv_other_count"][i],
                                                                  "P": df["priors_count"][i]
                                                    }
                                         )
        elif model == 3:
            conditioned = pyro.condition(GroundTruthModel3_Compas, data={
                                                                 "JF": df["juv_fel_count"][i],
                                                                 "JM": df["juv_misd_count"][i],
                                                                 "JO": df["juv_other_count"][i],
                                                                  "P": df["priors_count"][i]
                                                    }
                                         )

        posterior = pyro.infer.Importance(conditioned, num_samples=100).run()

        post_marginal_1 = pyro.infer.EmpiricalMarginal(posterior, "UJ")
        post_marginal_2 = pyro.infer.EmpiricalMarginal(posterior, "UD")

        post_samples1 = [post_marginal_1().item() for _ in range(100)]
        post_samples2 = [post_marginal_2().item() for _ in range(100)]

        mean1 = np.mean(post_samples1)
        mean2 = np.mean(post_samples2)

        knowledge1.append(mean1)
        knowledge2.append(mean2)

    knowledge1 = np.array(knowledge1).reshape(-1, 1)
    knowledge2 = np.array(knowledge2).reshape(-1, 1)
    knowledged = np.concatenate((knowledge1, knowledge2), axis=1)

    return knowledged

if __name__ == "__main__":
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='compas')
    # parser.add_argument('--data_name', type=str, default='adult')
    # parser.add_argument('--data_name', type=str, default='bank')
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()

    """Load configuration"""
    config_path = "/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml"
    conf = load_config(config_path)

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

    data_name = args.data_name
    generate = args.generate

    log_path = conf['log_train_{}'.format(data_name)]
    data_path = conf['data_{}'.format(data_name)]

    """Set up logging"""
    logger = setup_logging(log_path)

    """Load data"""
    df = pd.read_csv(data_path)

    """Setup features"""
    dict_ = features_setting(data_name)
    sensitive_features = dict_["sensitive_features"]
    normal_features = dict_["normal_features"]
    categorical_features = dict_["categorical_features"]
    continuous_features = dict_["continuous_features"]
    full_features = dict_["full_features"]
    discrete_features = dict_['discrete_features']
    target = dict_["target"]

    print("Normal features:", normal_features)
    print("Categorical features:", categorical_features)
    print("Full features:", full_features)

    """Preprocess data"""
    for c in continuous_features:
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    count_dict = {}
    for c in categorical_features:
        count_dict[c] = len(df[c].unique())
        le = preprocessing.LabelEncoder()
        df[c] = le.fit_transform(df[c])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(c)
        print(le_name_mapping)
        print(df[c].value_counts())
    # sys.exit(0)
    # print(df)
    # sys.exit(1)
    """Split dataset into train and test"""
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
    df_train = df_train.reset_index(drop = True)
    df_test = df_test.reset_index(drop=True)

    # """Full model"""
    # logger.debug('Full model')
    # clf = LogisticRegression()
    # clf.fit(df_train[full_features], df_train[target])
    # y_pred = clf.predict(df_test[full_features].values)
    # df_test['full'] = y_pred.reshape(-1)
    # y_pred = clf.predict_proba(df_test[full_features].values)[:, 0]
    # df_test['full_proba'] = y_pred.reshape(-1)
    #
    # """Unaware model"""
    # logger.debug('Unware model')
    # clf = LogisticRegression()
    # clf.fit(df_train[normal_features], df_train[target])
    # y_pred = clf.predict(df_test[normal_features].values)
    # df_test['unaware'] = y_pred.reshape(-1)
    # y_pred = clf.predict_proba(df_test[normal_features].values)[:, 0]
    # df_test['unaware_proba'] = y_pred.reshape(-1)

    """Counterfactual fairness model"""
    # logger.debug('Cf model')

    for i in full_features:
        df_train[i] = [torch.tensor(x) for x in df_train[i].values]
        df_test[i] = [torch.tensor(x) for x in df_test[i].values]

    if data_name == 'adult':
        print("Data {}".format(data_name))
        if generate:
            print("Generating features.....................")
            flag, monte, level = True, False, 1
            knowledged_train = infer_knowledge_adult(df_train, level=level)
            knowledged_test = infer_knowledge_adult(df_test, level=level)
            np.save(conf['adult_train1_true'], knowledged_train)
            np.save(conf['adult_test1_true'], knowledged_test)

            flag, monte, level = True, False, 2
            knowledged_train = infer_knowledge_adult(df_train, level=level)
            knowledged_test = infer_knowledge_adult(df_test, level=level)
            np.save(conf['adult_train3_true'], knowledged_train)
            np.save(conf['adult_test3_true'], knowledged_test)

            flag, monte, level = True, False, 3
            knowledged_train = infer_knowledge_adult(df_train, level=level)
            knowledged_test = infer_knowledge_adult(df_test, level=level)
            np.save(conf['adult_train3_true'], knowledged_train)
            np.save(conf['adult_test3_true'], knowledged_test)

            sys.exit(0)

        knowledged_train = np.load(conf['adult_train2_true'])
        knowledged_test = np.load(conf['adult_test2_true'])
        clf = LogisticRegression()
        clf.fit(knowledged_train, df[target])
        y_pred = clf.predict(knowledged_test)
        df_test['cf1'] = y_pred.reshape(-1)
        y_pred = clf.predict_proba(knowledged_test)[:, 0]
        df_test['cf1_proba'] = y_pred.reshape(-1)
        df_test.to_csv(conf[data_name], index=False)

        knowledged_train = np.load(conf['adult_train3_true'])
        knowledged_test = np.load(conf['adult_test3_true'])
        clf = LogisticRegression()
        clf.fit(knowledged_train, df[target])
        y_pred = clf.predict(knowledged_test)
        df_test['cf2'] = y_pred.reshape(-1)
        y_pred = clf.predict_proba(knowledged_test)[:, 0]
        df_test['cf2_proba'] = y_pred.reshape(-1)
        df_test.to_csv(conf[data_name], index=False)


    elif data_name == 'compas':
        """Groundtruth model 1"""

        if generate:
            print("Generating features.....................")

            flag, monte, level = True, False, 1
            knowledged_train = infer_knowledge_compas(df_train, level)
            knowledged_test = infer_knowledge_compas(df_test, level)
            np.save(conf['compas_train1'], knowledged_train)
            np.save(conf['compas_test1'], knowledged_test)

            # flag, monte, level = True, False, 2
            # knowledged_train = infer_knowledge_compas(df_train, level)
            # knowledged_test = infer_knowledge_compas(df_test, level)
            # np.save(conf['compas_train2'], knowledged_train)
            # np.save(conf['compas_test2'], knowledged_test)
            #
            # flag, monte, level = True, False, 3
            # knowledged_train = infer_knowledge_compas(df_train, level)
            # knowledged_test = infer_knowledge_compas(df_test, level)
            # np.save(conf['compas_train3'], knowledged_train)
            # np.save(conf['compas_test3'], knowledged_test)

            sys.exit(0)

        knowledged_train = np.load(conf['compas_train1'])
        knowledged_test = np.load(conf['compas_test1'])

        clf = LogisticRegression()
        clf.fit(knowledged_train, df_train[target])
        y_pred = clf.predict(knowledged_test)
        df_test['cf1'] = y_pred.reshape(-1)
        y_pred = clf.predict_proba(knowledged_test)[:, 0]
        df_test['cf1_proba'] = y_pred.reshape(-1)

        """Groundtruth model 2"""
        knowledged_train = np.load(conf['compas_train2'])
        knowledged_test = np.load(conf['compas_test2'])

        clf = LogisticRegression()
        clf.fit(knowledged_train, df_train[target])
        y_pred = clf.predict(knowledged_test)
        df_test['cf2'] = y_pred.reshape(-1)
        y_pred = clf.predict_proba(knowledged_test)[:, 0]
        df_test['cf2_proba'] = y_pred.reshape(-1)

        """Groundtruth model 3"""
        knowledged_train = np.load(conf['compas_train3'])
        knowledged_test = np.load(conf['compas_test3'])

        clf = LogisticRegression()
        clf.fit(knowledged_train, df_train[target])
        y_pred = clf.predict(knowledged_test)
        df_test['cf3'] = y_pred.reshape(-1)
        y_pred = clf.predict_proba(knowledged_test)[:, 0]
        df_test['cf3_proba'] = y_pred.reshape(-1)

    for i in full_features:
        df_test[i] = [x.detach().numpy() for x in df_test[i]]

    """Full model"""
    logger.debug('Full model')
    clf = LogisticRegression()
    clf.fit(df_train[full_features], df_train[target])
    y_pred = clf.predict(df_test[full_features].values)
    df_test['full_log'] = y_pred.reshape(-1)
    y_pred = clf.predict_proba(df_test[full_features].values)[:, 0]
    df_test['full_proba_log'] = y_pred.reshape(-1)

    clf = GradientBoostingClassifier()
    clf.fit(df_train[full_features], df_train[target])
    y_pred = clf.predict(df_test[full_features].values)
    df_test['full_gbr'] = y_pred.reshape(-1)
    y_pred = clf.predict_proba(df_test[full_features].values)[:, 0]
    df_test['full_proba_gbr'] = y_pred.reshape(-1)

    """Unaware model"""
    logger.debug('Unware model')
    clf = LogisticRegression()
    clf.fit(df_train[normal_features], df_train[target])
    y_pred = clf.predict(df_test[normal_features].values)
    df_test['unaware_log'] = y_pred.reshape(-1)
    y_pred = clf.predict_proba(df_test[normal_features].values)[:, 0]
    df_test['unaware_proba_log'] = y_pred.reshape(-1)

    clf = GradientBoostingClassifier()
    clf.fit(df_train[normal_features], df_train[target])
    y_pred = clf.predict(df_test[normal_features].values)
    df_test['unaware_gbr'] = y_pred.reshape(-1)
    y_pred = clf.predict_proba(df_test[normal_features].values)[:, 0]
    df_test['unaware_proba_gbr'] = y_pred.reshape(-1)

    df_test.to_csv(conf[data_name], index=False)
    print("Output to {}".format(conf[data_name]))


