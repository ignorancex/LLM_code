"""Bayesian Joint Probability of target distribution and predictor output
distribution. Assumes that the predictor output is dependent on the target
distribution and that the prediction output can be obtained with some error by
a transformation function of the target distribution.
"""
from copy import deepcopy
import logging
from multiprocessing import Pool

import numpy as np
import scipy
from sklearn.neighbors import BallTree
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib import distrib_utils
from psych_metric.distrib import mle_gradient_descent
from psych_metric.distrib.conditional.simplex_differences_transform import \
    DifferencesTransform

# TODO handle continuous distrib of continuous distrib, only discrete atm.


class SupervisedJointDistrib(object):
    """Bayesian distribution fitting of a joint probability distribution whose
    values correspond to the supervised learning context with target data and
    predictor output data.

    Attributes
    ----------
    independent : bool
        True if the random variables are indpendent of one anothers, False
        otherwise. Default is False
    target_distrib : tfp.distribution.Distribution
        The probability distribution of the target data.
    transform_distrib : tfp.distribution.Distribution | DifferencesTransform
        The probability distribution of the transformation function for
        transforming the target distribution to the predictor output data. If
        `independent` is true, then this is a tfp distribution, otherwise it is
        a DifferencesTransform.
    tf_sample_sess : tf.Session
    tf_target_samples : tf.Tensor
    tf_pred_samples : tf.Tensor
    tf_num_samples : tf.placeholder
    sample_dim : int, optional
        The number of dimensions of a single sample of both the target and
        predictor distribtutions.
    """

    def __init__(
        self,
        target_distrib,
        transform_distrib,
        target=None,
        pred=None,
        data_type='nominal',
        independent=False,
        knn_num_neighbors=10,
        total_count=None,
        tf_sess_config=None,
        mle_args=None,
        knn_num_samples=int(1e6),
        hyperbolic=False,
        sample_dim=None,
        dtype=np.float32,
        n_jobs=1,
        zero_mean=False,
    ):
        """
        Parameters
        ----------
        target_distrib : str | dict | tfp.distribution.Distribution
            Either the parameters to a  distribution to be used as the fitted
            distribution the target data, or the actual distribution to be
            used.
        transform_distrib : str | dict | tfp.distribution.Distribution
            Either the parameters to a  distribution to be used as the fitted
            distribution the target data, or the actual distribution to be
            used.
        target : np.ndarray, optional
            The target data of the supervised learning task.
        pred : np.ndarray, optional
            The predictions of the predictor for the samples corresponding to
            the target data.
        data_type : str, optional
            Identifier of the data type of the target and the predictor output.
            Values include 'nominal', 'ordinal', and 'continuous'. Defaults to
            'nominal'.
        independent : bool, optional
            The joint probability distribution's random variables are treated
            as independent if True, otherwise the second random variable is
            treated as dependent on the first (this is default). This affects
            how the sampling is handled, specifically in how the
            `transform_distrib` is treated. If True, the `transform_distrib` is
            instead treated as the distrib of second random variable.
        total_count : int
            Non-zero, positive integer of total count for the
            Dirichlet-multinomial target distribution.
        mle_args : dict, optional
            Dictionary of arguments for the maximum likelihood estimation
            method used. Contains an `optimizer_id` key with a str identifier
            of which optimization method is used: 'adam', 'nadam',
            'nelder-mead', 'simulated_annealing', 'nuts', 'random_walk'
        knn_num_samples : int
            number of samples to draw for the KNN density estimate. Defaults to
            int(1e6).
        dtype : type, optional
            The data type of the data. If a single type, all data is assumed to
            be of that type. Defaults to np.float32
        num_params : int
            The total number of parameters of the entire joint probability
            distribution model.
        num_params_target : int
            The number of parameters of the target distribution
        num_params_transform : int
            The number of parameters of the transform distribution
        """
        self.independent = independent
        self.dtype = dtype
        if target is not None and pred is not None:
            self.sample_dim = target.shape[1]
        elif sample_dim is not None:
            self.sample_dim = sample_dim
        else:
            raise TypeError(' '.join([
                '`target` and `pred` must be provided together, otherwise',
                '`sample_dim` must be given instead, along with',
                '`target_distrib` and `transform_distrib` given explicitly as',
                'an already defined distribution each.',
            ]))


        if not isinstance(total_count, int):
            # Check if any distribs are DirMult()s. They need total_counts
            if (
                isinstance(target_distrib, tfp.distributions.DirichletMultinomial)
                or (
                    isinstance(target_distrib, dict)
                    and 'DirichletMultinomial' == target_distrib['distrib_id']
                )
                or isinstance(transform_distrib, tfp.distributions.DirichletMultinomial)
                or (
                    isinstance(transform_distrib, dict)
                    and 'DirichletMultinomial' == transform_distrib['distrib_id']
                )
            ):
                raise TypeError(' '.join([
                    '`total_count` of type `int` must be passed in order to',
                    'use DirichletMultinomial distribution as the target',
                    f'distribution, but recieved type {type(total_count)}',
                    'instead.',
                ]))
        self.total_count = total_count

        # Fit the data (simply modularizes the fitting code,)
        self.fit(
            target_distrib,
            transform_distrib,
            target,
            pred,
            independent,
            mle_args,
            knn_num_samples,
            knn_num_neighbors,
            n_jobs,
            hyperbolic,
            zero_mean=zero_mean,
        )

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        for k, v in self.__dict__.items():
            if k == 'transform_distrib' or k == 'target_distrib':
                setattr(new, k, v.copy())
            else:
                setattr(new, k, deepcopy(v, memo))

        return new

    def _fit_independent_distrib(
        self,
        distrib,
        data=None,
        mle_args=None,
    ):
        """Fits the given data with an independent distribution."""
        if isinstance(distrib, tfp.distributions.Distribution):
            # Use given distribution as the fitted distribution.
            return distrib
        elif isinstance(distrib, str):
            distrib = distrib.lower()

            # By default, use UMVUE of params of given data. No MLE fitting.
            if data is None:
                raise ValueError('`data` must be provided when `distrib` '
                    + 'is of type `str`')

            if distrib == 'dirichletmultinomial':
                total_count = np.sum(data, axis=1).max(0)

                if mle_args:
                    data = data.astype(np.float32)
                    mle_results = mle_gradient_descent.mle_adam(
                        distrib,
                        np.maximum(data, np.finfo(data.dtype).tiny),
                        init_params={
                            'total_count': total_count,
                            'concentration': np.mean(data, axis=0),
                        },
                        const_params=['total_count'],
                        **mle_args,
                    )

                    return  tfp.distributions.DirichletMultinomial(
                        **mle_results[0].params
                    )

                return  tfp.distributions.DirichletMultinomial(
                    total_count,
                    np.mean(data, axis=0), #/ total_count,
                )
            elif distrib == 'dirichlet':
                if mle_args:
                    # TODO need to 1) be given a dtype, 2) enforce that in all
                    # data and tensors.
                    data = data.astype(np.float32)
                    mle_results = mle_gradient_descent.mle_adam(
                        distrib,
                        np.maximum(data, np.finfo(data.dtype).tiny),
                        init_params={'concentration': np.mean(data, axis=0)},
                        **mle_args,
                    )

                    if mle_args['alt_distrib']:
                        return  tfp.distributions.Dirichlet(
                            mle_results[0].params['mean'] * mle_results[0].params['precision']
                        )

                    return  tfp.distributions.Dirichlet(
                        **mle_results[0].params
                    )

                return  tfp.distributions.Dirichlet(np.mean(data, axis=0))
            elif distrib == 'multivariatenormal':
                return tfp.distributions.MultivariateNormalFullCovariance(
                    np.mean(data, axis=0),
                    np.cov(data, bias=False, rowvar=False),
                )
            elif distrib == 'multivariatestudentt':
                raise NotImplementedError('Need to return a non tfp distrib')
                # TODO ensure all is well in mle_adam and return the tfp
                # MultivariateStudentT Linear Operator.

                # initial df is 3 for loc = mean and Covariance exist.
                # initial random scale is cov * (df -2) / df / 2
                # as a poor attempt to ge a matrix in the proper range of
                # values of scale.

                if mle_args:
                    mle_results = mle_gradient_descent.mle_adam(
                        distrib,
                        data,
                        init_params={
                            'df': 3.0,
                            'loc': np.mean(data, axis=0),
                            'covariance_matrix': np.cov(data, bias=False, rowvar=False),
                        },
                        **mle_args,
                    )

                    # TODO handle returning a NON tfp distrib, as MVSTLinOp
                    # relies on scale, rather than Sigma.
                    #return  tfp.distributions.MultivariateStudentTLinearOperator(
                    #    **mle_results[0].params
                    #)
                    raise NotImplementedError('Need to return a non tfp distrib')

                    #return  mvst.MultivariateStudentT(**mle_results[0].params)
            else:
                raise ValueError(' '.join([
                    'Currently only "Dirichlet", "DirichletMultinomial",',
                    '"MultivariateNormal", or "MultivariateStudentT" for,',
                    'independent distributions are supported as proof of',
                    'concept.',
                ]))
        elif isinstance(distrib, dict):
            # If given a dict, use as initial parameters and fit with MLE
            data = data.astype(np.float32)
            if (
                distrib['distrib_id'] == 'DirichletMultinomial'
                or distrib['distrib_id'] == 'Dirichlet'
                or distrib['distrib_id'] == 'MultivariateNormal'
            ):
                return  mle_gradient_descent.mle_adam(
                    distrib['distrib_id'],
                    np.maximum(data, np.finfo(data.dtype).tiny),
                    init_params=distrib['params'],
                    **mle_args,
                )
            else:
                raise ValueError(' '.join([
                    'Currently only "DirichletMultinomial", "Dirichlet", or',
                    '"MultivariateNormal" for ' 'independent distributions',
                    'are supported as proof of concept.',
                ]))
        else:
            raise TypeError(' '.join([
                '`distrib` is expected to be either of type',
                '`tfp.distributions.Distribution`, `str`, or `dict`, not',
                f'{type(dict)}.',
                'If a `str`, then it is the name of the distribution.',
                'If `dict`, then it is the parameters of the distribution ',
                'with "distrib_id" as a key to indicate which distribution.',
            ]))

    def _create_sampling_attributes(self, sess_config=None):
        """Creates the Tensorflow session and ops for sampling."""
        self.tf_sample_sess = tf.Session(config=sess_config)

        num_samples = tf.placeholder(tf.int32, name='num_samples')

        self.tf_target_samples = self.target_distrib.sample(num_samples)

        if self.independent:
            # just sample from transform_distrib and return paired RVs.
            self.tf_pred_samples = self.transform_distrib.sample(num_samples)
        else:
            # The pred sampling is handled by the Conditional Distrib class
            self.tf_pred_samples = None

        self.tf_num_samples = num_samples

        # Run once. the saved memory is freed upon this class instance deletion.
        self.tf_sample_sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

    @property
    def num_params(self):
        return self.target_num_params + self.transform_num_params

    @property
    def params(self):
        params_dict = {
            'target': distrib_utils.get_tfp_distrib_params(
                self.target_distrib,
            ),
        }
        if isinstance(self.transform_distrib, DifferencesTransform):
            params_dict['transform'] = distrib_utils.get_tfp_distrib_params(
                self.transform_distrib.distrib
            )
        else:
            params_dict['transform'] = distrib_utils.get_tfp_distrib_params(
                self.transform_distrib
            )
        return params_dict

    def fit(
        self,
        target_distrib,
        transform_distrib,
        target,
        pred,
        independent=False,
        mle_args=None,
        knn_num_samples=int(1e6),
        knn_num_neighbors=10,
        n_jobs=1,
        hyperbolic=False,
        tf_sess_config=None,
        zero_mean=False,
    ):
        """Fits the target and transform distributions to the data."""
        # TODO check if the target and pred match the distributions' sample
        # space
        # TODO parallelize the fitting of the target and predictor distribs

        # Fit the target data
        self.target_distrib = self._fit_independent_distrib(
            target_distrib,
            target,
            mle_args,
        )

        # Set the number of parameters for the target distribution
        self.target_num_params = distrib_utils.get_num_params(
            target_distrib,
            self.sample_dim,
        )

        # Fit the transformation function of the target to the predictor output
        if self.independent:
            self.transform_distrib = self._fit_independent_distrib(
                transform_distrib,
                pred,
                mle_args,
            )
        #elif isinstance(transform_distrib, tfp.distributions.Distribution):
        #    # Use given distribution as the fitted dependent distribution
        #    self.transform_distrib = transform_distrib
        else:
            if (
                isinstance(transform_distrib, str)
                and transform_distrib.lower() == 'bnn'
            ):
                raise NotImplementedError(
                    'BNN Transform pipeline needs created.',
                )
            else:
                self.transform_distrib = DifferencesTransform(
                    target,
                    pred,
                    transform_distrib,
                    knn_num_neighbors,
                    hyperbolic,
                    n_jobs,
                    knn_num_samples,
                    input_dim=self.sample_dim,
                    mle_args=mle_args,
                    sess_config=tf_sess_config,
                    zero_mean=zero_mean,
                )

        # Set the number of parameters for the transform distribution
        self.transform_num_params = distrib_utils.get_num_params(
            transform_distrib,
            self.sample_dim,
        )

        # Create the Tensorflow session and ops for sampling
        self._create_sampling_attributes(tf_sess_config)

    def sample(self, num_samples, normalize=False):
        """Sample from the estimated joint probability distribution.

        Parameters
        ----------
        normalize : bool, optional (default=False)
            If True, the samples' distributions will be normalized such that
            their values are in the range [0, 1] and all sum to 1, otherwise
            they will be in the range [0, total_count] and sum to total_count.
            This is intended specifically for when the two random variables are
            distributions of distributions (ie. Dirichlet-multinomial).

            Ensure the elements sum to one (under assumption they sum to total
            counts.)

        Returns
        -------
        (np.ndarray, np.ndarray), shape(samples, input_dim, 2)
            Two arrays of samples from the joint probability distribution of
            the target and the predictor output aligned by samples in their
            first dimension.
        """
        if self.independent:
            target_samples, pred_samples = self.tf_sample_sess.run(
                [self.tf_target_samples, self.tf_pred_samples],
                feed_dict={self.tf_num_samples: num_samples}
            )
        else:
            target_samples = self.tf_sample_sess.run(
                self.tf_target_samples,
                feed_dict={self.tf_num_samples: num_samples}
            )

            pred_samples = self.transform_distrib.sample(target_samples)

        if normalize:
            if isinstance(
                self.target_distrib,
                tfp.distributions.DirichletMultinomial,
            ):
                target_samples = target_samples / self.total_count

            if isinstance(
                self.transform_distrib.distrib,
                tfp.distributions.DirichletMultinomial,
            ):
                pred_samples = pred_samples / self.total_count

        return target_samples, pred_samples

    def log_prob(
        self,
        target,
        pred,
        num_neighbors=None,
        return_individuals=False,
    ):
        """Log probability density/mass function calculation for the either the
        joint probability by default or the individual random variables.

        Parameters
        ----------
        target : np.ndarray
            The samples who serve as the independent random variable.
        pred : np.ndarray
            The samples who serve as the 2nd random variable, either dependent
            or independent.
        num_neighbors : int, optional
            The number of neighbors to use in the K Nearest Neighbors density
            estimate of the probability of the 2nd random variable's pdf. Only
            used if the 2nd random variable is dependent upon 1st, meaning we
            use a stochastic transform funciton.
        return_individuals : bool
            If False then returns the joint log probability p(target, pred),
            otherwise returns a tuple/list of the two random variables
            individual log probabilities, and their joint.

        Returns
        -------
            Returns the joint log probability as a float when `joint` is True
            or returns the individual random variables unconditional log
            probability when `joint` is False.
        """
        # If a Dirichlet tfp distrib, must set minimum to tiny for that dtype
        if (
            isinstance(
                self.target_distrib,
                tfp.distributions.DirichletMultinomial,
            )
            or isinstance(self.target_distrib, tfp.distributions.Dirichlet)
        ):
            target = target.astype(self.dtype)
            target = np.maximum(target, np.finfo(target.dtype).tiny)

        if (
            self.independent
            and (
                isinstance(
                    self.transform_distrib,
                    tfp.distributions.DirichletMultinomial,
                )
                or isinstance(
                    self.transform_distrib,
                    tfp.distributions.Dirichlet,
                )
            )
        ):
            pred = pred.astype(self.dtype)
            pred = np.maximum(pred, np.finfo(pred.dtype).tiny)

        if self.independent:
            # If independent, then measure the MLE of the distribs separately.
            log_prob_target, log_prob_pred = self.tf_sample_sess.run((
                self.target_distrib.log_prob(target),
                self.transform_distrib.log_prob(pred)
            ))
        else:
            log_prob_target = self.target_distrib.log_prob(target).eval(
                session=self.tf_sample_sess,
            )
            log_prob_pred = self.transform_distrib.log_prob(target, pred)

        if return_individuals:
            return (
                log_prob_target + log_prob_pred,
                log_prob_target,
                log_prob_pred,
            )
        return log_prob_target + log_prob_pred

    def info_criterion(self, mle, criterions='bic', num_samples=None, data=None):
        """Calculate information criterions.

        Parameters
        ----------
        mle : float | np.ndarray
            The pre-calculated maximum likelihood estimate.
        criterions : str | array-like, optional
            array-like of strings that indicate which criterions to be
            calculated from the following: 'bic', 'aic', 'hqc'. Defaults to
            'bic' for Bayesian Information Criterion.
        num_samples : int, optional
            Only necessary when criterions is or includes `bic` or 'hqc' to
            calculate the Hannan-Quinn information criterion. This is
            unnecessary if data is provided.
        data : tuple, optional
            Tuple of two np.ndarrays containing the data pertaining to the
            target and the predictor respectively. They must be of the shape
            matching their respective distribution's sample space. The usage of
            this parameter overrides the usage of MLE and the mle will be
            calculated from the log probability of the data.

        Returns
        -------
            A dict of floats for each criterion calculated.
        """
        # TODO calculate log prob if data is given

        # TODO need to add num_params property (target num params, transform
        # num params) and num params of sjd = sum of that.
        info_criterion = {}

        # TODO type checking and exception raising

        if 'bic' in criterions:
            info_criterion['bic'] = distrib_utils.bic(
                mle,
                self.num_params,
                num_samples,
            )

        if 'aic' in criterions:
            info_criterion['aic'] = distrib_utils.aic(
                mle,
                self.num_params,
            )

        if 'hqc' in criterions:
            info_criterion['hqc'] = distrib_utils.hqc(
                mle,
                self.num_params,
                num_samples,
            )

        return info_criterion

    def entropy(self):
        # TODO calculate / approximate the entropy of the joint distribution
        raise NotImplementedError
