import numpy as np
import scipy
from sklearn.neighbors import BallTree
import tensorflow as tf
import tensorflow_probability as tfp


def knn_log_prob(fitting_samples, pred, k=None):
    """Empirically estimates the predictor log probability using K Nearest
    Neighbords.
    """
    knn_tree = BallTree(fitting_samples)
    if len(pred.shape) == 1:
        # single sample
        pred = pred.reshape(1, -1)

    radius = knn_tree.query(pred, k)[0][:, -1]

    # log(k) - log(n) - log(volume)
    log_prob = np.log(k) - np.log(len(fitting_samples))

    # calculate the n-1 sphere volume being contained w/in the n-1 simplex
    n = pred.shape[1] - 1
    log_prob -= n * (np.log(np.pi) / 2 + np.log(radius)) - scipy.special.gammaln(n /
 2 + 1)

    return log_prob


def fit_dirichlet(concentration, k, fit_size, test_size):
    diri = tfp.distributions.Dirichlet(concentration)
    uni = tfp.distributions.Dirichlet([1] * len(concentration))

    fit = diri.sample(fit_size).eval(session=tf.Session())
    x = diri.sample(test_size).eval(session=tf.Session())

    diri_log = diri.log_prob(x).eval(session=tf.Session())
    knn_log = knn_log_prob(fit, x, k)
    uni_log = uni.log_prob(x).eval(session=tf.Session())

    print('\ndiri_log:')
    print(f'min: {diri_log.min()}, max: {diri_log.max()}')
    print(f'mean: {diri_log.mean()}, median: {np.median(diri_log)}')
    print(f'sum: {diri_log.sum()}')

    print('\nknn_log:')
    print(f'min: {knn_log.min()}, max: {knn_log.max()}')
    print(f'mean: {knn_log.mean()}, median: {np.median(knn_log)}')
    print(f'sum: {knn_log.sum()}')

    print('\nuni_log:')
    print(f'min: {uni_log.min()}, max: {uni_log.max()}')
    print(f'mean: {uni_log.mean()}, median: {np.median(uni_log)}')
    print(f'sum: {uni_log.sum()}')

    return diri_log, knn_log, uni_log
