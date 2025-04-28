import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
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
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from Function_Prediction import binary_evaluate
import joblib
from sklearn.metrics import make_scorer
from sklearn import metrics

# from HoloProtRepAFPML import AutomatedFunctionPredictionML
from Function_Prediction import binary_pytorch_network
from Function_Prediction.binary_pytorch_network import NN
import ast

acc_cv = []
f1_mi_cv = []
f1_ma_cv = []
f1_we_cv = []
pr_mi_cv = []
pr_ma_cv = []
pr_we_cv = []
rc_mi_cv = []
rc_ma_cv = []
rc_we_cv = []
hamm_cv = []
mcc_cv = []
std_acc_cv = []
std_f1_mi_cv = []
std_f1_ma_cv = []
std_f1_we_cv = []
std_pr_mi_cv = []
std_pr_ma_cv = []
std_pr_we_cv = []
std_rc_mi_cv = []
std_rc_ma_ce_cv = []
std_hamm_cv = []
mc = []
acc = []
y_proba = []


def intersection(real_annot, pred_annot):
    count = 0
    tn = 0
    tp = 0
    for i in range(len(real_annot)):
        if real_annot[i] == pred_annot[i]:
            if real_annot[i] == 0:
                tn += 1
            else:
                tp += 1
            count += 1

    return tn, tp


def evaluate_annotation_f_max(real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total = 0
    tn = 0
    tp = 0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue

        tn, tp = intersection(real_annots[i], pred_annots[i])
        fp = len(pred_annots[i]) - tp
        fn = len(real_annots[i]) - tn
        total += 1
        recall = tp / (1.0 * (tp + fn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tp / (1.0 * (tp + fp))
            p += precision

    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)

    return f


def check_for_at_least_two_class_sample_exits(y):
    for column in list(y):
        column_sum = np.sum(y[column])
        if column_sum < 2:
            print(
                "At least 2 positive samples needed for each class {0} class has {1} positive samples".format(
                    column, column_sum
                )
            )
            return False
    return True


def create_valid_kfold_object_for_multilabel_splits(X, y, kf):
    check_for_at_least_two_class_sample_exits(y)
    y_df = pd.DataFrame(y)
    sample_class_occurance = dict(zip(y_df, np.zeros(len(y_df))))
    for column in y_df:
        for fold_train_index, fold_test_index in kf.split(X, y_df):
            fold_col_sum = np.sum(y_df.iloc[fold_test_index, :][column].array)
            if fold_col_sum > 0:
                sample_class_occurance[column] += 1

    for key in sample_class_occurance:
        value = sample_class_occurance[key]
        if value < 2:
            random_state = np.random.randint(1000)
            print(
                "Random state changed since at least two positive samples are needed in different train/test folds.\
                    \nHowever, only one fold exits with positive samples for class {0}".format(
                    key
                )
            )
            print("Selected random state is {0}".format(random_state))
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            create_valid_kfold_object_for_multilabel_splits(X, y_df, kf)
        else:
            return kf


import ast
import matplotlib.pyplot as plt


def Model_test(representation_name, integrated_dataframe, parameteter_file):

    parameters = parameteter_file

    if "Label" in integrated_dataframe[0].columns:
        model_label = np.array(integrated_dataframe[0]["Label"])
        # label_list = [ast.literal_eval(label) for label in integrated_dataframe[0]['Label']]
        protein_representation = integrated_dataframe[0].drop(
            ["Label", "Entry"], axis=1
        )

        model_label_array = np.array(model_label)
    else:
        protein_representation = integrated_dataframe[0].drop(["Entry"], axis=1)

    proteins = list(integrated_dataframe[0]["Entry"])
    vectors = list(protein_representation["Vector"])
    protein_and_representation_dictionary = dict(zip(proteins, vectors))
    row = protein_representation.shape[0]
    row_val = round(math.sqrt(row), 0)
    # mlt = MultiLabelBinarizer()
    # model_label = mlt.fit_transform(label_list)
    protein_representation_array = np.array(
        list(protein_representation["Vector"]), dtype=float
    )

    f_max_cv = []
    # representation_name="_".join(representation_name_list)
    path = os.getcwd() + "/case_study/case_study_results"
    if "test" not in os.listdir(path):
        os.makedirs(path + "/test", exist_ok=True)
    classifier_name_lst = []
    index = 0
    for i in range(len(parameters)):
        classifier_name_lst.append(parameters[i]["classifier_name"])
    # classifier_len=len( classifier_name_lst )
    file_name = "_"
    file_name = file_name.join(classifier_name_lst)

    for i in range(len(parameters)):
        label_lst = []
        model_label_pred_lst = []
        if parameters[i]["classifier_name"] == "RandomForestClassifier":
            classifier_name = "RandomForestClassifier"
            classifier_name_lst.append(classifier_name)
            # params=ast.literal_eval(parameters[i]['best parameter'][0])
            params = parameters[i]["best parameter"]
            classifier = RandomForestClassifier(
                max_depth=params["model_classifier__max_depth"],
                min_samples_leaf=params["model_classifier__min_samples_leaf"],
                n_estimators=params["model_classifier__n_estimators"],
            )
            model_pipline = Pipeline(
                [("scaler", StandardScaler()), ("model_classifier", classifier)]
            )
            # classifier_multilabel = OneVsRestClassifier(classifier, n_jobs=-1)

            # model_pipline = Pipeline([('scaler', StandardScaler()), ('model_classifier', classifier)])
            index = index + 1
        elif parameters[i]["classifier_name"] == "SVC":
            classifier_name = "SVC"
            classifier_name_lst.append(classifier_name)
            params = parameters[i]["best parameter"]
            classifier = SVC(
                C=params["model_classifier__C"],
                gamma=params["model_classifier__gamma"],
                kernel=params["model_classifier__kernel"],
                max_iter=params["model_classifier__max_iter"],
            )
            # classifier_multilabel = OneVsRestClassifier(classifier, n_jobs=-1)
            model_pipline = Pipeline(
                [("scaler", StandardScaler()), ("model_classifier", classifier)]
            )
            index = index + 1
        elif parameters[i]["classifier_name"] == "KNeighborsClassifier":
            classifier_name = "KNeighborsClassifier"
            classifier_name_lst.append(classifier_name)
            params = parameters[i]["best parameter"]  # ast.literal_eval()
            up_limit = int(math.sqrt(int(len(model_label) / 5)))
            k_range = list(range(1, up_limit))
            classifier = KNeighborsClassifier(
                n_neighbors=params["model_classifier__n_neighbors"],
                weights=params["model_classifier__weights"],
                algorithm=params["model_classifier__algorithm"],
                leaf_size=params["model_classifier__leaf_size"],
            )
            model_pipline = Pipeline(
                [("scaler", StandardScaler()), ("model_classifier", classifier)]
            )
            index = index + 1
        elif parameters[i]["classifier_name"] == "Fully Connected Neural Network":
            m = 0
            classifier_name = "Fully Connected Neural Network"
            classifier_name_lst.append(classifier_name)
            input_size = len(protein_representation_array[0])
            index = index + 1
        class_number = 1
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        kf = create_valid_kfold_object_for_multilabel_splits(
            protein_representation, model_label, kf
        )

        if classifier_name == "Fully Connected Neural Network":

            input_size = len(protein_representation_array[0])
            m = 0
            protein_name = []
            for fold_train_index, fold_test_index in kf.split(
                protein_representation, model_label
            ):

                class_number = 1
                protein_representation_fold = pd.DataFrame(
                    protein_representation["Vector"], index=list(fold_test_index)
                )
                model_label_pred, parameter, model = pytorch_network.NN(
                    protein_representation_fold["Vector"],
                    model_label[fold_test_index],
                    input_size,
                    class_number,
                    representation_name,
                )
                model_label_pred_lst.append(model_label_pred.detach().numpy())
                label_lst.append(model_label[fold_test_index])
                for vec in protein_representation_array[fold_test_index]:
                    for (
                        protein,
                        vector,
                    ) in protein_and_representation_dictionary.items():
                        if str(vector) == str(list(vec)):
                            protein_name.append(protein)
                            continue

            #paths =os.path.join(path,"test",representation_name+"_"+classifier_name+"_"+"binary_classifier.pt")
            #breakpoint()   
            #torch.save(model,paths )

            # Initialize optimizer
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

            # Print model's state_dict
            print("Model's state_dict:")
            import pdb; pdb.set_trace()
            for param_tensor in model.state_dict():
                print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            # Print optimizer's state_dict
            print("Optimizer's state_dict:")
            for var_name in optimizer.state_dict():
                print(var_name, "\t", optimizer.state_dict()[var_name])

        #            evaluate.evaluate(kf, mlt, protein_representation, model_label_pred_lst, label_lst, classifier_name,representation_name,protein_and_representation_dictionary,f_max_cv##,file_name,index)

        else:
            protein_name = []

            # joblib.dump(scaler, '/media/DATA/home/sinem/yayin_calismasi/results/test/std_scaler'+classifier_name+'.bin')
            model_label_pred = cross_val_predict(
                classifier, protein_representation_array, model_label, cv=kf, n_jobs=-1
            )
            for fold_train_index, fold_test_index in kf.split(
                protein_representation_array, model_label
            ):

                model_label_pred_lst.append(model_label_pred[fold_test_index])
                label_lst.append(model_label[fold_test_index])
                for vec in protein_representation_array[fold_test_index]:
                    for (
                        protein,
                        vector,
                    ) in protein_and_representation_dictionary.items():
                        if str(vector) == str(list(vec)):
                            protein_name.append(protein)
                            continue

                rep_name_and_go_id = representation_name
            filename =os.path.join(path,"test",classifier_name+"binary_classifier_test_model.joblib")
            
            classifier.fit(protein_representation_array, model_label)
            joblib.dump(classifier, filename)
            # joblib.dump(model_pipline, filename)
        col_names = ["Labels"]
        label_predictions = pd.DataFrame(
            np.concatenate(model_label_pred_lst), columns=col_names
        )
        # ls=np.concatenate(model_label_pred_lst)
        label_predictions.insert(0, "protein_id", protein_name)
        # label_predictions["prediction_values"]=[i for i in ls]
        label_predictions.to_csv(os.path.join(path,"test",classifier_name+"binary_classifier_test_predictions.tsv"),sep="\t",index=False,)


        binary_evaluate.evaluate(
            kf,
            protein_representation,
            model_label_pred_lst,
            label_lst,
            classifier_name,
            representation_name,
            protein_and_representation_dictionary,
            file_name,
            index,
            "test",
        )
