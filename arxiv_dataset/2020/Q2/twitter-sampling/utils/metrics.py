import numpy as np
import scipy.stats


def symmetric_mean_absolute_percentage_error(true, pred):
    # percentage error, zero if both true and pred are zero
    true = np.array(true)
    pred = np.array(pred)
    daily_smape_arr = 100 * np.nan_to_num(np.abs(true - pred) / (np.abs(true) + np.abs(pred)))
    return np.mean(daily_smape_arr), daily_smape_arr


def mean_absolute_percentage_error(true, pred):
    # true is positive
    true = np.array(true)
    pred = np.array(pred)
    daily_mape_arr = np.abs(true - pred) / true
    return np.mean(daily_mape_arr), daily_mape_arr


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
