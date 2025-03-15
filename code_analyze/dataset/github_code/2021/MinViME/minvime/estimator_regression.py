""" 
    Performance Estimator for Regression Models
    Functions for estimating the required model performance requirements from business criteria
"""
import random
import numpy as np
import pandas as pd

from .generator import extract_distribution_from_sample
from .generator import produce_distribution_sample
from .generator import resample_toward_mean
from .generator import generate_min_max_baseline
from .generator import generate_candidate_predictions
from .generator import copy_with_noise

######################################################################
def estimate_model_requirements_thresholded(dist, cases, pred_value, under_pred, under_pred_unit, under_pred_threshold, 
  over_pred, over_pred_unit, over_pred_threshold, minroi):
    candidates = generate_candidate_predictions(dist)
    rez_rmse = 0
    rez_mape = 0
    rez_mae = 0
    rez_return = 0
    current_roi = -999
    for candidate in candidates:
        roi = calculate_candidate_roi(dist, candidate, pred_value, under_pred, under_pred_unit, over_pred, over_pred_unit)
        if current_roi == -999 or ( roi >= minroi and current_roi > roi):
            rez_rmse, rez_mape, rez_mae = calculate_candidate_metrics(dist, candidate)
    return rez_rmse, rez_mape, rez_mae


######################################################################
def estimate_model_requirements_proportional( dist, cases, pred_value, under_pred, under_pred_unit, over_pred, over_pred_unit, minroi):
    """ Given a sample of target values and requirements -- estimate baseline performance characteristics """
    candidates = generate_candidate_predictions(dist)
    rez_rmse = 0
    rez_mape = 0
    rez_mae = 0
    rez_return = 0
    current_roi = -999
    for candidate in candidates: 
        roi = calculate_candidate_roi(dist, candidate, pred_value, under_pred, under_pred_unit, over_pred, over_pred_unit)
        if current_roi == -999 or ( roi >= minroi and current_roi > roi):
            rez_rmse, rez_mape, rez_mae = calculate_candidate_metrics(dist, candidate)
    return rez_rmse, rez_mape, rez_mae

########################################################################
def calculate_candidate_metrics(dist, candidate):
    errors = [ x-y for x,y in zip(dist, candidate) ]
    sqrerrors = [ (x-y)*(x-y) for x,y in zip(dist, candidate) ] 
    zero_adj_dist = [ 0.0001 if x==0 else x for x in dist]
    abspcterror = [ 100*abs(x-y)/x for x,y in zip(zero_adj_dist, candidate) ]
    rmse = np.sqrt(np.mean(sqrerrors))
    mae = np.mean(np.abs(errors))
    mape = np.mean(abspcterror)
    return rmse, mape, mae


########################################################################
def calculate_candidate_roi(dist, candidate, pred_value, under_pred, under_pred_unit, over_pred, over_pred_unit): 
    total = 0
    for pred, act in zip(candidate, dist):
        error = pred - act
        abs_error = np.abs(error)
        if act == 0:
            p_error = 100 * abs_error/0.0001
        else:
            p_error = 100 * abs_error/act
        if error<0:
            if under_pred_unit=='percent':
                total = total + pred_value - (under_pred * p_error)
            else:
                total = total + pred_value - (under_pred * abs_error)
        else:
            if over_pred_unit=='percent':
                total = total + pred_value - (over_pred * p_error)
            else:
                total = total + pred_value - (over_pred * abs_error)
    return total

