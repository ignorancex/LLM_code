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
import torch
import sys

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from aif360.metrics import ClassificationMetric

from tqdm import tqdm
from geomloss import SamplesLoss
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio

from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from collections import Counter

import torch

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
         Convert the source domain data and target domain data into a kernel matrix, which is the K above
    Params:
	     source: source domain data (n * len(x))
	     target: target domain data (m * len(y))
	    kernel_mul:
	     kernel_num: take the number of different Gaussian kernels
	     fix_sigma: sigma values ​​of different Gaussian kernels
	Return:
		 sum(kernel_val): sum of multiple kernel matrices
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# Find the number of rows of the matrix. Generally, the scales of source and target are the same, which is convenient for calculation
    total = torch.cat([source, target], dim=0)#Combine source and target in column direction
    #Copy total (n+m) copies
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #Copy each row of total into (n+m) rows, that is, each data is expanded into (n+m) copies
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #Find the sum between any two data, the coordinates (i, j) in the obtained matrix represent the l2 distance between the i-th row of data and the j-th row of data in total (0 when i==j)
    L2_distance = ((total0-total1)**2).sum(2)
    #Adjust the sigma value of the Gaussian kernel function
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #Take fix_sigma as the median value, and take kernel_mul as a multiple of kernel_num bandwidth values ​​(for example, when fix_sigma is 1, you get [0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #Gaussian kernel function mathematical expression
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #Get the final kernel matrix
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
         Calculate the MMD distance between source domain data and target domain data
    Params:
	     source: source domain data (n * len(x))
	     target: target domain data (m * len(y))
	    kernel_mul:
	     kernel_num: take the number of different Gaussian kernels
	     fix_sigma: sigma values ​​of different Gaussian kernels
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#Generally the default is that the batchsize of the source domain and the target domain are the same
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #According to formula (3) divide the kernel matrix into 4 parts
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#Because it is generally n==m, the L matrix is ​​generally not added to the calculation

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
    evaluations['R2score'] = r2_score(y_true, y_pred)


    return evaluations

def evaluate_classifier(y_pred, y_true):
    """
    Evaluate performance of classifier in terms of precision, recall, f-measure, accuracy
    :param y_pred: DESCRIPTION
    :type y_pred: TYPE
    :param y_true: DESCRIPTION
    :type y_true: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    # print(y_true)
    # print(y_pred)
    # print(Counter(y_pred))
    # print(Counter(y_true))

    evaluations = {}
    evaluations['F1 Score'] = f1_score(y_true, y_pred, average='weighted')
    evaluations['Precision'] = precision_score(y_true, y_pred, average='weighted')
    evaluations['Recall'] = recall_score(y_true, y_pred, average='weighted')
    evaluations['Accuracy'] = accuracy_score(y_true, y_pred)



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

    backend = "auto"

    Loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, backend = backend)
    evaluation['sinkhorn'] = Loss(ys, ys_hat).cpu().detach().numpy() 
    
    Loss = SamplesLoss("energy", p=2, blur=0.05, scaling=0.5, backend = backend)
    evaluation["energy"] = Loss(ys, ys_hat).cpu().detach().numpy()
    
    Loss = SamplesLoss("gaussian", p=2, blur=0.5, scaling=0.5, backend = backend)
    evaluation["gaussian"] = Loss(ys, ys_hat).cpu().detach().numpy()

    Loss = SamplesLoss("laplacian", p=2, blur=0.5, scaling=0.5, backend = backend)
    evaluation["laplacian"] = Loss(ys, ys_hat).cpu().detach().numpy()



    
    return evaluation 

def fair_metrics(df, target, label, sensitive, data_name = None):
    if data_name == 'compas':
        if sensitive == 'sex':
            privileged_classes = 1
        if sensitive == 'race':
            privileged_classes = 0
    elif data_name == 'adult':
        privileged_classes = 1
    elif data_name == 'bank':
        privileged_classes = 2


    dataset = StandardDataset(df,
                              label_name=label,
                              favorable_classes=[1],
                              protected_attribute_names=[sensitive],
                              privileged_classes=[[privileged_classes]])

    dataset_pred = dataset.copy()
    dataset_pred.labels = df[target].values

    attr = dataset_pred.protected_attribute_names[0]
    idx = dataset_pred.protected_attribute_names.index(attr)
    privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
    unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

    classified_metric = ClassificationMetric(dataset,
                                             dataset_pred,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)

    result = {
              'true_positive_rate_difference_{}'.format(sensitive):
                  classified_metric.true_positive_rate_difference(),
              'generalized_entropy_index_{}'.format(sensitive):
                  classified_metric.generalized_entropy_index(alpha=2),
              # 'generalized_entropy_error': classified_metric.generalized_entropy_error(),
              # 'consistency_score': classified_metric.consistency_score(),
              'coefficient_of_variation_{}'.format(sensitive): classified_metric.coefficient_of_variation(),
              'theil_index_{}'.format(sensitive) : classified_metric.theil_index(),
              'equal_opportunity_difference_{}'.format(sensitive):
                  classified_metric.equal_opportunity_difference(),
              'average_abs_odds_difference_{}'.format(sensitive):
                  classified_metric.average_abs_odds_difference(),
              'balanced_acc': 0.5 * (classified_metric.true_positive_rate()
                                        + classified_metric.true_negative_rate()),
              # 'smoothed_empirical_differential_fairness': classified_metric.smoothed_empirical_differential_fairness(),
              # 'consistency': classified_metric.consistency(),
              'between_all_groups_theil_index_{}'.format(sensitive): classified_metric.between_all_groups_theil_index(),
              'between_group_generalized_entropy_index_{}'.format(sensitive): classified_metric.between_group_generalized_entropy_index(),
              'between_group_coefficient_of_variation_{}'.format(sensitive):  classified_metric.between_group_coefficient_of_variation(),
              # 'generalized_true_positive_rate_{}'.format(sensitive): classified_metric.generalized_true_positive_rate()
              }

    return result

def evaluate_fairness(sensitive_att, df, target, label=None, problem = "regression", dataname=None):
    """
    Evaluate the fairness aspects in terms of classification and regression
    For regerssion: use the distribution distance
    For classification: use the distribution distance + EOD

    :param sensitive_att:
    :param df:
    :param target:
    :param label:
    :param problem:
    :return: dictionary of evaluation metric
    """
    eval_performance = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sinkhorn, energy, gaussian, laplacian = 0, 0, 0, 0


    # df_term = df.sample(frac=0.1, replace=True, random_state=0)
    df_term = df.copy()
    if problem == "classification":
        for s in sensitive_att:
            # return eval_performance
            # ys = df_term[df_term[s] == 1][target + "_proba"].values
            # ys_hat = df_term[df_term[s] == 0][target + "_proba"].values
            # ys = torch.Tensor(ys).to(device).reshape(-1,1)
            # ys_hat = torch.Tensor(ys_hat).to(device).reshape(-1,1)
            # eval_fair = evaluate_distribution(ys, ys_hat)
            # sinkhorn += eval_fair['sinkhorn']
            # energy += eval_fair['energy']
            # gaussian += eval_fair['gaussian']
            # laplacian += eval_fair['laplacian']
            eval = fair_metrics(df, target, label, s, dataname)
            eval_performance.update(eval)
            # eval_performance.update(eval_fair)

    elif problem == "regression":
        df = df.sample(frac=0.5, replace=True, random_state=0)
        for s in sensitive_att:
            ys = df[df[s] == 1][target].values
            ys_hat = df[df[s] == 0][target].values
            ys = torch.Tensor(ys).to(device).reshape(-1,1)
            ys_hat = torch.Tensor(ys_hat).to(device).reshape(-1,1)
            eval_performance = evaluate_distribution(ys, ys_hat)
            sinkhorn += eval_performance['sinkhorn']
            energy += eval_performance['energy']
            gaussian += eval_performance['gaussian']
            laplacian += eval_performance['laplacian']
            # del ys
            # del ys_hat

        eval_performance['sinkhorn'] = sinkhorn
        eval_performance['energy'] = energy
        eval_performance['gaussian'] = gaussian
        eval_performance['laplacian'] = laplacian
    return eval_performance

def classification_performance(df,
                                        df_result,
                                        sensitive_features,
                                        label,
                                        prediction_columns,
                                        data_name):
    for m in prediction_columns:
        performance = {}
        performance['method'] = m
        print(m, label)
        print(df[m].values, df[label].values)
        performance_reg = evaluate_classifier(df[m].values, df[label].values)
        performance_fairness = evaluate_fairness(sensitive_features,
                                                 df,
                                                 target = m,
                                                 label = label,
                                                 problem="classification",
                                                 dataname=data_name)
        performance.update(performance_reg)
        performance.update(performance_fairness)
        df_result = df_result.append(performance, ignore_index=True)
    return df_result