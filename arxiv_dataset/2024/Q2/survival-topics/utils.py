import numpy as np
from multiprocessing import Pool


def compute_median_survival_time_last_observed_pmap_helper(args):
    # same code logic as lifelines
    survival_curve, duration_index = args
    if survival_curve[-1] > .5:
        return duration_index[-1]
    else:
        return duration_index[np.searchsorted(-survival_curve, [-.5])[0]]


def compute_median_survival_time_pmap_helper(args):
    # same code logic as lifelines
    survival_curve, duration_index = args
    if survival_curve[-1] > .5:
        return np.inf
    else:
        return duration_index[np.searchsorted(-survival_curve, [-.5])[0]]


def compute_median_survival_times(survival_curves, duration_index,
                                  num_threads=None,
                                  never_cross_half_behavior='inf'):
    # warning: assumes each survival curve is in chronological order with times
    # specified in `duration_index`; there is no input sanitization being done
    pmap_args = [(survival_curve, duration_index)
                 for survival_curve in survival_curves]
    with Pool(num_threads) as p:
        if never_cross_half_behavior == 'last observed':
            results = p.map(
                compute_median_survival_time_last_observed_pmap_helper,
                pmap_args)
        else:
            results = p.map(
                compute_median_survival_time_pmap_helper, pmap_args)
    return np.array(results)
