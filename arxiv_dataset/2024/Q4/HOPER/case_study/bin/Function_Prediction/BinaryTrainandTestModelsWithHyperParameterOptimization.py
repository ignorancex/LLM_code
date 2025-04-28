"""
This module trains and test protein function models and reveals best model and hyperparameters. The module structure is the following:

- The module implements ``check_for_at_least_two_class_sample_exits`` method. The method takes a dataframes as input.
The input dataframe has varying number of columns. Each column represent a class (i.e. GO ids). 
The methods analyze the data frame to control at least two positive sample exits for each class.

- The module implements ``select_best_model_with_hyperparameter_tuning`` method. The method takes representation name, a dataframe list and 
scoring function,list of preferred model names as input. The dataframe has 3 columns 'Label','Entry' and 'Vector'. The method 
trains models and search for best model and hyperparameters. Then module test modules

- The module implements ``binary_evaluate`` method. For calculation of model metrics.


"""
import ast
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil
from sklearn.model_selection import cross_validate
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

from sklearn.metrics import multilabel_confusion_matrix
import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import make_scorer
from sklearn import metrics

path = os.getcwd()
sys.path.append(path + "/case_study/bin/")
from Function_Prediction import binary_pytorch_network
from Function_Prediction.binary_pytorch_network import NN
from Function_Prediction import binary_prediction
from Function_Prediction import binary_evaluate
import torch
import joblib
random_state=42
from sklearn.metrics import make_scorer
from Function_Prediction.Model_Parameters import Kneighbors_Classifier_parameters
from Function_Prediction.Model_Parameters import SVC_Classifier_parameters
from Function_Prediction.Model_Parameters import RandomForest_Classifier_parameters
from Function_Prediction import F_max_scoring
import random

# if every fold contains at least 2 positive samples return true,otherwise return false
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


def neural_network_eval(
    f_max_cv,
    kf,
    model,
    model_label_pred_lst,
    label_lst,
    index_,
    representation_name,
    classifier_name,
    file_name,
    eval_type,
    protein_name,
    path,
    parameter,
):
    #breakpoint()
    representation_name_concated = "_".join(representation_name)
    #breakpoint()
    if eval_type=="training":
      paths =os.path.join(path,"training",representation_name_concated+"_"+classifier_name+"_binary_classifier.pt")
      #breakpoint()
      torch.save(model.state_dict(), paths)
      
      representation_name_concated = "_".join(representation_name)
      best_parameter_dataframe = pd.DataFrame(parameter)
      training_path=os.path.join(path,"training","Neural_network_"+ representation_name_concated+"_binary_classifier_best_parameter.csv")
      best_parameter_dataframe.to_csv(training_path,index=False)
     
    binary_evaluate.evaluate(
        kf,
        model_label_pred_lst,
        label_lst,
        f_max_cv,
        classifier_name,
        representation_name_concated,
        file_name,
        index_,
        eval_type,
    )

    label_predictions = pd.DataFrame(
        np.concatenate(model_label_pred_lst), columns=["Label"]
    )
    label_prediction_path=os.path.join(path,eval_type,representation_name_concated+"_binary_classifier_"+classifier_name+eval_type+"_predictions.csv")
    label_predictions.insert(0, "protein_id", protein_name)
    label_predictions.to_csv(
        label_prediction_path,
        index=False,
    )


best_param_list = []


def select_best_model_with_hyperparameter_tuning(
    representation_name,
    integrated_dataframe,
    scoring_key,
    models=[
        "RandomForestClassifier",
        "SVC",
        "KNeighborsClassifier",
        "Fully_Connected_Neural_ Network",
    ],
):
    
    scoring_function_dictionary = {
        "f1_micro": "f1_micro",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "f_max": F_max_scoring.scoring_f_max_machine,
    }
    class_len = len(models)
    #import pdb
    #pdb.set_trace()
    
    model_label = np.array(integrated_dataframe["Label"])
    # label_list = [ast.literal_eval(label) for label in integrated_dataframe['Label']]
    protein_representation = integrated_dataframe.drop(["Label", "Entry"], axis=1)
    proteins = list(integrated_dataframe["Entry"])
    vectors = list(protein_representation["Vector"])
    protein_and_representation_dictionary = dict(zip(proteins, vectors))
    row = protein_representation.shape[0]
    row_val = round(math.sqrt(row), 0)
    protein_representation_array = np.array(
        list(protein_representation["Vector"]), dtype=float
    )
    model_label_array = np.array(model_label)
    predictions_list, result_dict, classifier_name_lst = ([] for i in range(3))

    best_parameter_df = pd.DataFrame(
        columns={"representation_name", "classifier_name", "best parameter"}
    )

    index = 0
    model_count = 0
    representation_name_concated = ""
    file_name = "_"
    path = os.path.join(os.getcwd(),"case_study/case_study_results")
    path_train=os.path.join(path,"training")
    path_test=os.path.join(path,"test")
    if "training" not in os.listdir(path):
        os.makedirs(path_train, exist_ok=True)
        os.makedirs(path_test, exist_ok=True)
    file_name = file_name.join(models)
    best_param_list = []
    for classifier in models:
        index += 1
        m = 0
        model_label_pred_lst, label_lst, protein_name = ([] for i in range(3))

        input_size = len(protein_representation_array[0])

        if classifier == "RandomForestClassifier":
            
            random.seed(random_state)
            np.random.seed(random_state)
            classifier_ = RandomForestClassifier(random_state=random_state) 
            classifier_name = type(classifier_).__name__
            model_pipline = Pipeline(
                [("scaler", StandardScaler()), ("model_classifier", classifier_)]
            )
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            parameters = {
                "model_classifier__n_estimators": RandomForest_Classifier_parameters.n_estimators,
                "model_classifier__max_depth": RandomForest_Classifier_parameters.max_depth,
                "model_classifier__min_samples_leaf":RandomForest_Classifier_parameters.min_samples_leaf ,
            }

        elif classifier == "SVC":
            
            random.seed(random_state)
            np.random.seed(random_state)
            classifier_ = SVC(random_state=random_state)
            classifier_name = type(classifier_).__name__
            model_pipline = Pipeline(
                [("scaler", StandardScaler()), ("model_classifier", classifier_)]
            )
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            parameters = {
                "model_classifier__C": SVC_Classifier_parameters.C,
                "model_classifier__gamma": SVC_Classifier_parameters.gamma,
                "model_classifier__kernel":SVC_Classifier_parameters.kernel ,
                "model_classifier__max_iter":SVC_Classifier_parameters.max_iter,
            }

        elif classifier == "KNeighborsClassifier":
            
            random.seed(random_state)
            np.random.seed(random_state)
            classifier_ = KNeighborsClassifier()
            classifier_name = type(classifier_).__name__
            up_limit = int(math.sqrt(int(len(model_label_array) / 5)))
            model_pipline = Pipeline(
                [("scaler", StandardScaler()), ("model_classifier", classifier_)]
            )
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            k_range = list(range(1, up_limit))
            parameters = {
                "model_classifier__n_neighbors": k_range if len(Kneighbors_Classifier_parameters.n_neighbors)==0 else Kneighbors_Classifier_parameters.n_neighbors,
                "model_classifier__weights":  Kneighbors_Classifier_parameters.weights ,
                "model_classifier__algorithm": Kneighbors_Classifier_parameters.algorithm,
                "model_classifier__leaf_size": list(
                    range(1, int(len(model_label_array) / 5)) if len(Kneighbors_Classifier_parameters.leaf_size)==0 else Kneighbors_Classifier_parameters.leaf_size
                ),
                "model_classifier__p":Kneighbors_Classifier_parameters.p ,
            }

        
        '''kf = create_valid_kfold_object_for_multilabel_splits(
            protein_representation, model_label, kf
        )'''
        if classifier == "Fully_Connected_Neural_Network":
            
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            model_count = model_count + 1
            #breakpoint()
            # classifier_name_lst.append("Neural_Network")
            classifier_name = "Fully_Connected_Neural_Network"
            (
                f_max_cv,
                f_max_cv_train,
                f_max_cv_test,
                loss_train,
                loss,
                loss_tr,
                loss_test,
                protein_name_tr,
                model_label_pred_test_lst,
                label_lst_test,
                model_label_pred_lst,
            ) = ([] for i in range(11))
            (
                f_max_cv_train,
                f_max_cv_test,
                model,
                model_label_pred_lst,
                label_lst,
                protein_name_tr,
                parameter,
                protein_name,
                parameter,
                model_label_pred_test_lst,
                label_lst_test,
            ) = NN(
                kf,
                protein_representation,
                model_label,
                input_size,
                representation_name,
                protein_and_representation_dictionary,
            )
            
            neural_network_eval(
                f_max_cv_train,
                kf,
                model,
                model_label_pred_lst,
                label_lst,
                index,
                representation_name,
                classifier_name,
                file_name,
                "training",
                protein_name_tr,
                path,
                parameter,
            )

            neural_network_eval(
                f_max_cv_test,
                kf,
                model,
                model_label_pred_test_lst,
                label_lst_test,
                index,
                representation_name,
                classifier_name,
                file_name,
                "test",
                protein_name,
                path,
                parameter,
            )

        else:
            model_count = model_count + 1

            if(scoring_key[0] =="f_max"):        
                model_tunning = GridSearchCV(estimator=model_pipline, param_grid=parameters, cv=kf,pre_dispatch = 20,scoring=F_max_scoring.scoring_f_max_machine  , n_jobs=-1)
            else:
                model_tunning = GridSearchCV(estimator=model_pipline, param_grid=parameters, cv=kf,pre_dispatch = 20,scoring=scoring_function_dictionary[scoring_key[0]] , n_jobs=-1)   
            classifier_name_lst.append(classifier_name)
            model_tunning.fit(protein_representation_array, model_label)
            model_tunning.best_score_
            model_tunning.best_params_
            representation_name_concated = "_".join(representation_name)
            best_parameter_df = best_parameter_df.append(
                {
                    "representation_name": representation_name_concated
                    + "_"
                    + "binary_classifier",
                    "classifier_name": classifier_name,
                    "best parameter": model_tunning.best_params_,
                },
                ignore_index=True,
            )
            best_param_list.append(
                {
                    "representation_name": representation_name_concated
                    + "_"
                    + "binary_classifier",
                    "classifier_name": classifier_name,
                    "best parameter": model_tunning.best_params_,
                }
            )
            model_tunning.best_estimator_
            
            filename =os.path.join(path,"test",classifier_name+"_binary_classifier_test_model.joblib")
              
            joblib.dump(model_tunning.best_estimator_, filename)
            f_max_cv = []
            model_label_pred = cross_val_predict(
                model_tunning.best_estimator_,
                protein_representation_array,
                model_label,
                cv=kf,
                n_jobs=-1,
            )
            for fold_train_index, fold_test_index in kf.split(
                protein_representation, model_label
            ):
                model_label_pred = model_tunning.best_estimator_.predict(
                    protein_representation_array[fold_test_index]
                )
                model_label_pred_lst.append(model_label_pred)
                label_lst.append(model_label[fold_test_index])
                for vec in protein_representation_array[fold_test_index]:
                    for (
                        protein,
                        vector,
                    ) in protein_and_representation_dictionary.items():
                        if str(vector) == str(list(vec)):
                            protein_name.append(protein)
                            continue
                fmax = 0.0
                tmax = 0.0
                for t in range(1, 101):
                    threshold = t / 100.0
                    fscore = F_max_Scoring.evaluate_annotation_f_max(
                        model_label[fold_test_index], model_label_pred
                    )
                    if fmax < fscore:
                        fmax = fscore
                        tmax = threshold
                f_max_cv.append(fmax)
            representation_name_concated = "_".join(representation_name)
            #import pdb
            #pdb.set_trace()
            binary_evaluate.evaluate(
                kf,
                model_label_pred_lst,
                label_lst,
                f_max_cv,
                classifier_name,
                representation_name_concated,
                file_name,
                index,
                "test"
            )
            
            label_predictions = pd.DataFrame(
                np.concatenate(model_label_pred_lst), columns=["Label"]
            )
            label_predictions.insert(0, "protein_id", protein_name)
            label_predictions.to_csv(os.path.join(path,"test",representation_name_concated+"_binary_classifier_test_predictions.csv"),index=False,)

            class_name = "_".join(classifier_name_lst)
            best_parameter_df.to_csv(os.path.join(path,"test",representation_name_concated+"_"+class_name+"_binary_classifier_best_parameter.csv"),index=False,)


    return best_param_list
