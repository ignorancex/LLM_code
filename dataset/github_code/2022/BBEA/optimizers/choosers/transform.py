import math
import numpy as np

from sklearn.preprocessing import power_transform
from xoa.commons.logger import *

##
# Functions for response shaping

LOG_ERR_LOWER_BOUND = -5.0

def apply_no_shaping(err):
    return err


def apply_log_err(err):    
    
    err = np.log10(err)
    
    if err < LOG_ERR_LOWER_BOUND:
        err = LOG_ERR_LOWER_BOUND
    
    scale_err = (err - LOG_ERR_LOWER_BOUND) / abs(LOG_ERR_LOWER_BOUND)
    return scale_err


def apply_hybrid_log(err, threshold=0.3, err_lower_bound=0.00001):    
    log_th = math.log10(threshold)
    beta = threshold - log_th

    if err > threshold:
        return err  # linear scale
    else:
        if err > 0:
            log_applied = math.log10(err)
        else:
            log_applied = math.log10(err_lower_bound)
        return  log_applied + beta # log scale


def apply_power_transform(errs, err_std, method):
  
    shaped = np.array(errs).reshape(-1, 1)
    try:
        shaped = power_transform(shaped / err_std, method=method)
        if not np.isfinite(shaped).all():
            raise ValueError("Invalid transformation due to non-finite component(s)")
    except Exception as ex:
        debug("{} Original errors: {}".format(ex, errs))
        return errs
    
    errs = np.array(shaped).reshape(-1)
    return errs