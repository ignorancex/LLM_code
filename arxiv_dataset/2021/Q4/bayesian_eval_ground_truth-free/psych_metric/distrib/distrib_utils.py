"""Functions for performing distribution model selection and helper
functions.
"""
import functools
import logging
import math
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib import tfp_mvst


@functools.total_ordering
class MLEResults(object):
    def _is_valid_operand(self, other):
        return (
            hasattr(other, "neg_log_likelihood")
            and hasattr(other, "params")
            and hasattr(other, "info_criterion")
        )

    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.neg_log_likelihood == other.neg_log_likelihood

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.neg_log_likelihood < other.neg_log_likelihood

    def __init__(
        self,
        neg_log_likelihood: float,
        params=None,
        info_criterion=None,
    ):
        self.neg_log_likelihood = neg_log_likelihood
        self.params = params
        self.info_criterion = info_criterion


def aic(mle, num_params, log_mle=True):
    """Akaike information criterion (unadjusted).

    Parameters
    ----------
    mle : float
        Maximum Likelihood Estimate (MLE).
    num_params : int
        The number of parameters of the model whose MLE is given.
    log_mle : bool, optional
        True if the given `mle` is already the Log MLE, otherwise the log of
        the `mle` is taken in the equation.

    Returns
    -------
    float
        The Akaike information criterion of the given MLE for the model.
    """
    if log_mle:
        return 2 * (num_params - mle)
    return 2 * (num_params - math.log(mle))


def bic(mle, num_params, num_samples, log_mle=True):
    """Bayesian information criterion. Approx. Bayes Factor.

    Parameters
    ----------
    mle : float
        Maximum Likelihood Estimate (MLE).
    num_params : int
        The number of parameters of the model whose MLE is given.
    num_samples : int
        The number of samples used to fit the model whose MLE is given.
    log_mle : bool, optional
        True if the given `mle` is already the Log MLE, otherwise the log of
        the `mle` is taken in the equation.

    Returns
    -------
    float
        The Bayesian information criterion of the given MLE for the model.
    """
    if log_mle:
        return num_params * math.log(num_samples) - 2 * mle
    return num_params * math.log(num_samples) - 2 * math.log(mle)


def hqc(mle, num_params, num_samples, log_mle=True):
    """Hannan-Quinn information criterion.

    Parameters
    ----------
    mle : float
        Maximum Likelihood Estimate (MLE).
    num_params : int
        The number of parameters of the model whose MLE is given.
    num_samples : int
        The number of samples used to fit the model whose MLE is given.
    log_mle : bool, optional
        True if the given `mle` is already the Log MLE, otherwise the log of
        the `mle` is taken in the equation.

    Returns
    -------
    float
        The HWC of the given MLE for the model.
    """
    if log_mle:
        return 2 * (num_params * math.log(math.log(num_samples)) - mle)
    return 2 * (num_params * math.log(math.log(num_samples)) - math.log(mle))


def dic(likelihood_function, num_params, num_samples, mle=None, mean_lf=None):
    """Deviance information criterion.

    Parameters
    ----------
    mle : float
        the Maximum Likelihood Estimate to repreesnt the distribution's
        likelihood function.
    """
    raise NotImplementedError('Need to implement finding the MLE from '
        + 'Likelihood function, and the expected value of the likelihood '
        + 'function.')

    if mean_lf is None:
        raise NotImplementedError('Need to implement finding the expected '
            + 'value from Likelihood function.')
    if mle is None:
        raise NotImplementedError('Need to implement finding the MLE from '
            + 'Likelihood function.')

    # DIC = 2 * (pd - D(theta)), where pd is spiegelhalters: pd= E(D(theta)) - D(E(theta))
    # return 2 * (expected_value_likelihood_func - math.log(mle) - math.log(likelihood_funciton))


def calc_info_criterion(mle, num_params, criterions, num_samples=None):
    """Calculate information criterions with mle_list and other information."""
    info_criterion = {}

    if 'bic' in criterions:
        info_criterion['bic'] = bic(mle, num_params, num_samples)

    if 'aic' in criterions:
        info_criterion['aic'] = aic(mle, num_params)

    if 'hqc' in criterions:
        info_criterion['hqc'] = hqc(mle, num_params, num_samples)

    return info_criterion


def get_num_params(distrib, dims):
    """Convenience function for getting the number of params of a distribution
    given the number of dimensions.
    """
    if isinstance(distrib, tfp.distributions.Distribution):
        distrib = distrib.__class__.__name__.lower()
        # TODO perhaps add specific tfp distribution extraction of classes?
    elif isinstance(distrib, str):
        distrib = distrib.lower()
    else:
        raise TypeError(' '.join([
            'expected `distrib` to be of type `str` or',
            f'`tfp.distributions.Distribution`, not `{type(distrib)}`',
        ]))
    if not isinstance(dims, int):
        raise TypeError(
            f'expected `dims` to be of type `int`, not `{type(dims)}`',
        )

    if distrib in {'multivariatenormal', 'multivariatecauchy'}:
        # loc = dims, and scale matrix is a triangle matrix
        return dims + dims * (dims + 1) / 2
    if distrib == 'multivariatestudentt':
        # Same as Multivariate Normal, but with a degree of freedom per dim
        return dims + dims + dims * (dims + 1) / 2
    if distrib == 'dirichlet':
        # Concentration is number of classes
        return dims
    if distrib == 'dirichletmultinomial':
        # Concentration is number of classes + 1 for total counts
        return dims + 1


def is_prob_distrib(
    vector,
    rtol=1e-09,
    atol=0.0,
    equal_nan=False,
    axis=1,
):
    # TODO beware of single prob vector checks, needs to be 2d shaped array still
    """Checks if the vector is a valid discrete probability distribution."""
    # check if each row sums to 1
    sums_to_one = np.isclose(vector.sum(axis), 1, rtol, atol, equal_nan)

    # TODO set all vector values that are near 0 or 1 to 0 or 1, respectively
    # check if all values are w/in range
    in_range = ((vector + rtol) >= 0).all(axis) \
        & ((vector - rtol) <= 1).all(axis)

    return sums_to_one & in_range


def tf_is_prob_distrib(vector, axis=1):
    """Checks if the vector is a valid discrete probability distribution."""
    # check if each row sums to 1
    # NOTE does not use isclose() as its numpy equivalent does!
    sample_sum = tf.reduce_sum(vector, axis)
    sums_to_one = tf.equal(sample_sum, 1)

    # check if all values are w/in range
    in_range = tf.reduce_all(vector >= 0, axis) == tf.reduce_all(vector <= 1, axis)

    return tf.logical_and(sums_to_one, in_range)


def mvst_tf_log_prob(x, df, loc, sigma):
    raise NotImplementedError('This is not properly implemented. Use psych_metric.distrib.tfp_mvst.MultivariateStudentT.log_prob() instead.')
    with tf.name_scope('multivariate_student_t_log_prob') as scope:
        dims = tf.cast(loc.shape[0], tf.float32)

        # TODO Ensure this is broadcastable
        return (
            tf.math.lgamma((df + dims) / 2.0)
            - (df + dims) / 2.0 * (
                1.0 + (1.0 / df) * (x - loc) @ tf.linalg.inv(sigma) @ tf.transpose(x - loc)
            ) - (
                tf.math.lgamma(df / 2.0)
                + .5 * (dims * (tf.log(df) + tf.log(np.pi))
                    + tf.log(tf.linalg.norm(sigma))
                )
            )
        )


def get_tfp_distrib(distrib_id):
    distrib_id = distrib_id.lower()
    if distrib_id == 'dirichlet':
        return tfp.distributions.Dirichlet
    if distrib_id == 'multivariatenormal' or distrib_id == 'mvn':
        return tfp.distributions.MultivariateNormalFullCovariance
    if distrib_id == 'multivariatestudentt' or distrib_id == 'mvst':
        return tfp_mvst.MultivariateStudentT


def get_tfp_distrib_params(
    distrib,
    valid_params = {
        'concentration',
        'loc',
        'scale',
        'df',
        'total_count',
        'covariance_matrix',
    },
):
    """Returns the actual distribution parameters of the tfp distribution."""
    return {k: v for k, v in distrib._parameters.items() if k in valid_params}
