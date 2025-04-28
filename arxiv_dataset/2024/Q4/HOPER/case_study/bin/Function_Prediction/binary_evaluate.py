"""
This module calculates evaluation metrics.The module structure is the following:
Parameters:
---------
kfold instance,protein representation dataframe,model prediction list,model label list,calculated f_max score list,classifier name,
representation_name_concated model representation name,protein id and protein representation dictionary, file name,
index,calculation type "test" or "training"

return:
---------
It returns metrics list and mean values of metrics list for model result report


"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil

# from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef
import pickle
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)
from random import random
from sklearn.metrics import multilabel_confusion_matrix
import sys
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

# from imblearn.pipeline import Pipeline
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from Function_Prediction import binary_pytorch_network

# from  pytorch_network import NN

from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Function_Prediction import report_results_of_training


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def evaluate(
    kf,
    model_label_pred_lst,
    label_lst,
    f_max_cv,
    classifier_name,
    representation_name,
    file_name,
    index,
    classifier_type
    
):

    accuracy_cv = []
    f1_micro_avg_cv = []
    f1_macro_avg_cv = []
    f1_weighted_avg_cv = []
    precision_micro_avg_cv = []
    precision_macro_avg_cv = []
    precision_weighted_avg_cv = []
    recall_micro_avg_cv = []
    recall_macro_avg_cv = []
    recall_weighted_avg_cv = []
    hamming_distance_cv = []
    matthews_correlation_coefficient_cv = []
    std_accuracy_cv = []
    std_f1_micro_avg_cv = []
    std_f1_macro_avg_cv = []
    std_f1_weighted_avg_cv = []
    std_precision_micro_avg_cv = []
    std_precision_macro_avg_cv = []
    std_precision_weighted_avg_cv = []
    std_recall_micro_avg_cv = []
    std_recall_macro_avg_cv = []
    std_recall_weighted_avg_cv = []
    std_hamming_distance_cv = []
    std_f_max_cv = []
    GO_ids = []
   

    matthews_correlation_coefficient_lst = []
    accuracy_lst = []
    tn_lst = []
    fn_lst = []
    tp_lst = []
    fp_lst = []
    mean_result_list = []
    result_list = []
    auc_cv = []
    std_auc_cv = []
    fp_lst = []

    for fold_index in range(len(model_label_pred_lst)):

        tn, fp, fn, tp = confusion_matrix(
            label_lst[fold_index], model_label_pred_lst[fold_index]
        ).ravel()
        tn_lst.append(tn)
        tp_lst.append(tp)
        fn_lst.append(fn)
        fp_lst.append(fp)
        accuracy_lst.append((tp + tn) / (tp + fn + fp + tn))
        #import pdb
        #pdb.set_trace()
        matthews_correlation_coefficient_lst.append(
            ((tp * tn) - (fp * fn))
            / (1+math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        )
        matthews_correlation_coefficient_cv.append(matthews_correlation_coefficient_lst)
        ac = np.mean(accuracy_lst)
        accuracy_cv.append(ac)
        f1_mi = f1_score(
            label_lst[fold_index], model_label_pred_lst[fold_index], average="micro"
        )
        f1_micro_avg_cv.append(f1_mi)
        f1_ma = f1_score(
            label_lst[fold_index], model_label_pred_lst[fold_index], average="macro"
        )
        f1_macro_avg_cv.append(f1_ma)
        f1_we = f1_score(
            label_lst[fold_index], model_label_pred_lst[fold_index], average="weighted"
        )
        f1_weighted_avg_cv.append(f1_we)
        pr_mi = precision_score(
            label_lst[fold_index], model_label_pred_lst[fold_index], average="micro"
        )
        precision_micro_avg_cv.append(pr_mi)
        pr_ma = precision_score(
            label_lst[fold_index], model_label_pred_lst[fold_index], average="macro"
        )
        precision_macro_avg_cv.append(pr_ma)
        pr_we = precision_score(
            label_lst[fold_index], model_label_pred_lst[fold_index], average="weighted"
        )
        precision_weighted_avg_cv.append(pr_we)
        rc_mi = recall_score(
            label_lst[fold_index], model_label_pred_lst[fold_index], average="micro"
        )
        recall_micro_avg_cv.append(rc_mi)
        rc_ma = recall_score(
            label_lst[fold_index], model_label_pred_lst[fold_index], average="macro"
        )
        recall_macro_avg_cv.append(rc_ma)
        rc_we = recall_score(
            label_lst[fold_index], model_label_pred_lst[fold_index], average="weighted"
        )
        recall_weighted_avg_cv.append(rc_we)
        hamm = hamming_loss(label_lst[fold_index], model_label_pred_lst[fold_index])
        hamming_distance_cv.append(hamm)
        auc = compute_roc(label_lst[fold_index], model_label_pred_lst[fold_index])
        auc_cv.append(auc)
    std_accuracy_cv.append(np.std(accuracy_cv))
    std_f1_micro_avg_cv.append(np.std(f1_micro_avg_cv))
    std_f1_macro_avg_cv.append(np.std(f1_macro_avg_cv))
    std_f_max_cv.append(np.std(f_max_cv))
    std_f1_weighted_avg_cv.append(np.std(f1_weighted_avg_cv))
    std_precision_micro_avg_cv.append(np.std(precision_micro_avg_cv))
    std_precision_macro_avg_cv.append(np.std(precision_macro_avg_cv))
    std_precision_weighted_avg_cv.append(np.std(precision_weighted_avg_cv))
    std_recall_micro_avg_cv.append(np.std(recall_micro_avg_cv))
    std_recall_macro_avg_cv.append(np.std(recall_macro_avg_cv))
    std_recall_weighted_avg_cv.append(np.std(recall_weighted_avg_cv))
    std_hamming_distance_cv.append(np.std(hamming_distance_cv))
    std_auc_cv.append(np.std(auc_cv))

    result_list.append(
        [
            representation_name,
            classifier_name,
            accuracy_cv,
            std_accuracy_cv,
            f1_micro_avg_cv,
            std_f1_micro_avg_cv,
            f1_macro_avg_cv,
            std_f1_macro_avg_cv,
            f_max_cv,
            std_f_max_cv,
            f1_weighted_avg_cv,
            std_f1_weighted_avg_cv,
            precision_micro_avg_cv,
            std_precision_micro_avg_cv,
            precision_macro_avg_cv,
            std_precision_macro_avg_cv,
            precision_weighted_avg_cv,
            std_precision_weighted_avg_cv,
            recall_micro_avg_cv,
            std_recall_micro_avg_cv,
            recall_macro_avg_cv,
            std_recall_macro_avg_cv,
            recall_weighted_avg_cv,
            std_recall_weighted_avg_cv,
            hamming_distance_cv,
            std_hamming_distance_cv,
            auc_cv,
            std_auc_cv,
            matthews_correlation_coefficient_cv,
        ]
    )
    mean_result_list.append(
        [
            representation_name,
            classifier_name,
            np.mean(accuracy_cv),
            np.mean(std_accuracy_cv),
            np.mean(f1_micro_avg_cv),
            np.mean(std_f1_micro_avg_cv),
            np.mean(f1_macro_avg_cv),
            np.mean(std_f1_macro_avg_cv),
            np.mean(f_max_cv),
            np.mean(std_f_max_cv),
            np.mean(f1_weighted_avg_cv),
            np.mean(std_f1_weighted_avg_cv),
            np.mean(precision_micro_avg_cv),
            np.mean(std_precision_micro_avg_cv),
            np.mean(precision_macro_avg_cv),
            np.mean(std_precision_macro_avg_cv),
            np.mean(precision_weighted_avg_cv),
            np.mean(std_precision_weighted_avg_cv),
            np.mean(recall_micro_avg_cv),
            np.mean(std_recall_micro_avg_cv),
            np.mean(recall_macro_avg_cv),
            np.mean(std_recall_macro_avg_cv),
            np.mean(recall_weighted_avg_cv),
            np.mean(std_recall_weighted_avg_cv),
            np.mean(hamming_distance_cv),
            np.mean(std_hamming_distance_cv),
            np.mean(auc_cv),
            np.mean(std_auc_cv),
            np.mean(matthews_correlation_coefficient_cv),
        ]
    )

    report_results_of_training.report_results_of_training(
        representation_name,
        result_list,
        mean_result_list,
        classifier_name,
        file_name,
        index,
        classifier_type,
    )

    return (result_list, mean_result_list)
