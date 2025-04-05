# coding: utf-8

# Introduction

"""
The ensembles are based on the predictions of the three available multi-task models (asymmetry, border and color)
and two ensemble strategies: averaging and optimized weighted averaging.
"""

NAME = '2_ensembles'
PROJECT = 'HINTS'
PYTHON_VERSION = '3.8.2'
KERAS_VERSION = '2.3.1'

# Preamble

## Imports
from constants import *
import os, re
from keras.models import model_from_json
from get_data import get_train_validate_test_sets_annotations
from generate_data import generate_data_2
from report_results import report_auc
from sklearn.metrics import roc_auc_score
from get_data import annototation_type
import numpy as np
from scipy import optimize
import keras

## Settings


## Set working directory
workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)

## Set up pipeline folder if missing
pipeline = os.path.join('empirical', '2_pipeline', NAME)
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))


def loss_mse(weights, valid_labels, pred_val):
    predictions = np.average(pred_val, weights=weights, axis=0)
    mse = keras.losses.MeanSquaredError()
    loss = mse(valid_labels, predictions).numpy()
    return loss


def reconstruct_model(weights_path):
    weight_file = os.path.join('empirical', '2_pipeline', '1_multi_task', 'out') + weights_path
    json_file = open(weight_file + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_file + '.h5')
    if VERBOSE:
        model.summary()
    return model


def validate_model(model):
    data_path = os.path.join('empirical', '0_data', 'external')
    valid = generate_data_2(directory=data_path,
                            augmentation=False,
                            batch_size=BATCH_SIZE,
                            file_list=valid_id,
                            label_1=valid_label_c,
                            label_2=valid_label_a,
                            sample_weights=valid_mask)
    predictions = model.predict_generator(valid, 25)
    delta_size = predictions[0].size - valid_label_c.count()
    scores = np.resize(predictions[0], predictions[0].size - delta_size)
    return scores


def predict_model(model):
    data_path = os.path.join('empirical', '0_data', 'external')
    test = generate_data_2(directory=data_path,
                           augmentation=False,
                           batch_size=BATCH_SIZE,
                           file_list=test_id,
                           label_1=test_label_c,
                           label_2=test_label_a,
                           sample_weights=test_mask)
    predictions = model.predict_generator(test, 25)
    delta_size = predictions[0].size - test_label_c.count()
    scores = np.resize(predictions[0], predictions[0].size - delta_size)
    return scores


seeds = [1970, 1972, 2008, 2019, 2020]
for seed in seeds:
    pred_validation = np.zeros([3, 350])
    pred_test = np.zeros([3, 250])

    if VERBOSE:
        print_constants()

    if CONV_LAYER_FROZEN:
        asy_model = reconstruct_model('/model_frozen' + str(seed) + str(annototation_type.asymmetry))
        bor_model = reconstruct_model('/model_frozen' + str(seed) + str(annototation_type.border))
        col_model = reconstruct_model('/model_frozen' + str(seed) + str(annototation_type.color))
    else:
        asy_model = reconstruct_model('/model_not_frozen' + str(seed) + str(annototation_type.asymmetry))
        bor_model = reconstruct_model('/model_not_frozen' + str(seed) + str(annototation_type.border))
        col_model = reconstruct_model('/model_not_frozen' + str(seed) + str(annototation_type.color))

    ground_truth_file = os.path.join('empirical', '0_data', 'external', 'ISIC-2017_Training_Part3_GroundTruth.csv')
    group_path = os.path.join('empirical', '0_data', 'manual', 'student_2018_abc_features')
    train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a, valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights = get_train_validate_test_sets_annotations(
        group_path, ground_truth_file, seed, VERBOSE, SANITY_CHECK, annototation_type.asymmetry)
    asy_pred_score = predict_model(asy_model)
    if ENSEMBLE_LEARNING == ensemble_learning_type.optimizing:
        asy_val_score = validate_model(asy_model)
        pred_validation[0, :] = asy_val_score
        pred_test[0, :] = asy_pred_score

    train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a, valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights = get_train_validate_test_sets_annotations(
        group_path, ground_truth_file, seed, VERBOSE, SANITY_CHECK, annototation_type.border)
    bor_pred_score = predict_model(bor_model)
    if ENSEMBLE_LEARNING == ensemble_learning_type.optimizing:
        bor_val_sore = validate_model(bor_model)
        pred_validation[1, :] = bor_val_sore
        pred_test[1, :] = bor_pred_score

    train_id, valid_id, test_id, train_label_c, valid_label_c, test_label_c, train_label_a, valid_label_a, test_label_a, train_mask, valid_mask, test_mask, class_weights = get_train_validate_test_sets_annotations(
        group_path, ground_truth_file, seed, VERBOSE, SANITY_CHECK, annototation_type.color)
    col_pred_score = predict_model(col_model)
    if ENSEMBLE_LEARNING == ensemble_learning_type.optimizing:
        cor_val_score = validate_model(col_model)
        pred_validation[2, :] = cor_val_score
        pred_test[2, :] = col_pred_score

    if ENSEMBLE_LEARNING == ensemble_learning_type.optimizing:
        weights = np.array([0.33, 0.33, 0.33])
        weights_optim = optimize.differential_evolution(loss_mse,
                                                        bounds=[(0.0, 1.0) for _ in range(len(weights))],
                                                        args=(valid_label_c, pred_validation),
                                                        maxiter=1000,
                                                        tol=1e-7)
        if VERBOSE:
            print(weights_optim.x)
        scores = np.average(pred_test, weights=weights_optim.x, axis=0)
    else:
        scores = []
        for i in range(0, len(test_id)):
            if ENSEMBLE_LEARNING == ensemble_learning_type.soft_voting:
                max_prob = asy_pred_score[i]
                if bor_pred_score[i] > max_prob:
                    max_prob = bor_pred_score[i]
                if col_pred_score[i] > max_prob:
                    max_prob = col_pred_score[i]
                scores.append(max_prob)
            else:
                if ENSEMBLE_LEARNING == ensemble_learning_type.averaging:
                    scores.append((asy_pred_score[i] + bor_pred_score[i] + col_pred_score[i]) / 3.0)

    aucs = []
    auc = roc_auc_score(test_label_c, scores)
    aucs.append(auc)

    if CONV_LAYER_FROZEN:
        filename = os.path.join(pipeline, 'out', 'aucs_frozen' + str(seed) + '.csv')
    else:
        filename = os.path.join(pipeline, 'out', 'aucs_not_frozen' + str(seed) + '.csv')

    report_auc(aucs, filename)