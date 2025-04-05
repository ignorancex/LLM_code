# Copyright 2019 by Ben Granett, granett@gmail.com
# All rights reserved.
# This file is part of the VIPERS-SHAM project,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import logging
import numpy as np
from sklearn import neighbors
from scipy import interpolate


def nn_regression(data, y, nn=10, rank=True):
    """ Nearest-neighbor regression
    
    Parameters
    ----------
    data : numpy.ndarray
        The dependent variables shape (M,N) array
    y : numpy.ndarray
        The regression variable length N array
    nn : int
        Number of neighbors
    rank : bool
        Use rank distances
    """
    if rank:
        datat = []
        for v in data.transpose():
            order = v.argsort()
            rank = np.zeros(len(order), dtype=float)
            rank[order] = np.arange(len(rank))
            datat.append(rank)
        datat = np.transpose(datat)
    else:
        datat = data

    reg = neighbors.KNeighborsRegressor(n_neighbors=nn, weights='uniform')
    reg.fit(datat, y)
    yfit = reg.predict(datat)
    return yfit


def fit_cumu_dist(x, n=1000):
    """ Compute the cumulative distribution function
    """
    h, e = np.histogram(x, bins=n)
    hc = np.cumsum(h)
    hc = hc*1./hc[-1]
    f = interpolate.interp1d(e[:-1], hc, bounds_error=False, fill_value=(0, 1))
    f_inv = interpolate.interp1d(hc, e[:-1], bounds_error=False, fill_value=(e[0],e[-1]))
    return f, f_inv


class CompletenessHD(object):
    """ High-dimensional completeness estimator
    
    Parameters
    ----------
    alpha : nump.ndarray
    alpha_limit: numpy.ndarray
    params : numpy.ndarray
    """
    
    def __init__(self, alpha, alpha_limit, params, nn=10, rank=True):
        """ """
        data = np.transpose(params)

        # subtract the mean and divide by stddev to whiten data
        mu = np.mean(data, axis=0)
        sig = np.std(data, axis=0)
        data_t = (data-mu)/sig

        mu_alpha = np.mean(alpha)
        alpha_t = alpha - mu_alpha

        # compute the trend alpha using nearest-neighbor regressor
        self.alpha_trend = mu_alpha + nn_regression(data_t, alpha_t, nn=nn, rank=rank)

        # the residual around the trend
        resid = alpha - self.alpha_trend

        logging.info("Fit residual scatter %f", np.std(resid)) 

        # compute cumulative distribution function of residuals
        self._compl_func, self._compl_func_inv = fit_cumu_dist(resid)
        
        logging.info("90th percentile %4.3f", self._compl_func_inv(0.9))
        logging.info("10th percentile %4.3f", self._compl_func_inv(0.1))
        
        self._comp = self._compl_func(alpha_limit - self.alpha_trend)

    def get_completeness(self, alpha_limit=None):
        """ """
        if not alpha_limit:
            return self._comp
        return self._compl_func(alpha_limit - self.alpha_trend)

    def get_limit(self, p=0.9):
        """ """
        assert (p > 0) & (p < 1)
        return self._compl_func_inv(p)
