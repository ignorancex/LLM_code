"""The Class and related functions for a BNN implemented via MCMC."""
from copy import deepcopy
import logging
from functools import partial

import tensorflow as tf

from psych_metric.distrib import bnn_transform
from psych_metric.distrib.simplex.euclidean import EuclideanSimplexTransform
from psych_metric.distrib.bnn.mcmc import MCMC


class BNNMCMC(object):
    """An implementation of a Bayesian Neural Network implemented via Markov
    Chain Monte Carlo.

    Attributes
    ----------
    input : tf.placeholder
    output : tf.Tensor
    weight_placeholder : list(tf.placeholders)
    sess_config :
    dtype : tf.dtype
    mcmc_sample_log_prob : function | partial
        Function used as the target log probability function of the MCMC chain.
    ? mcmc : MCMC
        The MCMC object that contains all of the MCMC chains and their related
        attributes.
    converged_weights_set : list(np.ndarray)
        The set of BNN weights that have converged and are used for
        initialization of the MCMC chains for sampling.
    """

    def __init__(
        self,
        dim,
        num_layers=1,
        num_hidden=5,
        hidden_activation=tf.math.sigmoid,
        hidden_use_bias=True,
        output_activation=None, #, tf.math.sigmoid,
        output_use_bias=True, # False,
        #kernel_args, # TODO
        dtype=tf.float32,
        #sess=None,
        sess_config=None,
        simplex_transform=None,
    ):
        """
        Parameters
        ----------
        hidden_activation : function
            Defaults to tf.math.sigmoid. None is same as linear.
        hidden_use_bias : bool
        output_activation : function
            Default to None, thus linear. example is tf.math.sigmoid
        output_use_bias : bool
        kernel :
        sess_config :
        """
        self.dtype = dtype
        self.sess_config = sess_config

        if simplex_transform is None:
            self.simplex_transform = EuclideanSimplexTransform(dim)
        elif simplex_transform.input_dim != dim:
            raise ValueError(' '.join([
                'The given simplex transform expects a different number of',
                'input dimensions than given to the BNNMCMC.',
            ]))

        # Make BNN and weight_placeholders, input & output tensors
        self.input = tf.placeholder(
            dtype=dtype,
            shape=[None, dim],
            name='input_labels',
        )

        self.output, self.weight_placeholders = bnn_transform.bnn_softmax_placeholders(
            self.input,
            self.simplex_transform,
            num_layers,
            num_hidden,
            hidden_activation,
            hidden_use_bias,
            output_activation,
            output_use_bias,
            dtype,
        )

        # TODO need to add actual fitting via MCMC, sampling, etc.
        # for now, just a way to better contain the args and functions.
        #self.mcmc_sample_log_prob = partial(
        #    bnn_transform.mcmc_sample_log_prob,
        #    origin_adjust=self.simplex_transform.origin_adjust,
        #    rotation_mat=self.simplex_transform.change_of_basis_matrix,
        #    scale_identity_multiplier=scale_identity_multiplier,
        #)

        # TODO MCMC(mcmc_args)
        #self.mcmc = MCMC(self.mcmc_sample_log_prob, **mcmc_args)

    def __copy__(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo):
        raise NotImplementedError

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new

        for k, v in self.__dict__.items():
            setattr(new, k, deepcopy(v, memo))

        return new

    @property
    def scale_identity_multiplier(self):
        # NOTE is this a safe way of returning this value as read only?
        #return self.mcmc_sample_log_prob.keywords['scale_identity_multipler']
        pass

    def fit(self, *args, **kwargs):
        #kernel_args

        raise NotImplementedError(' '.join([
            'The API does not yet contain the BNN MCMC training code. It',
            'exists in prototype format as a series of functions.',
            'See `bnn_exp.py`, and `proto_bnn_mcmc.py`.',
        ]))

        #self.mcmc.fit(*args, **kwargs)

        # TODO First, find step size that yields desired acceptance rate.
        #   Will have to repeated check the status of accept rate and possibly
        #   loop back to this part if it diverges from the desired accept rate.

        # TODO Second, run MCMC chains until convergence or

        # TODO Third, check last run's acf or pcf for the first lag that is
        # under the desired threshold of autocorrelation.

        # TODO If all MCMC chains converge to similar location, MCMC is fit.

        # TODO apply some timer, or amount of max iterations. to stop infinite
        # loops
        # TODO use logging to indicate useful info of the process.

    def get_weight_sets(self, num_sets, parallel_chains):
        """After fitting the BNN via MCMC, get a set of weights for different
        instances of the BNN to obtian the distribution of outputs.
        """
        raise NotImplementedError(' '.join([
            'The API does not yet contain the BNN MCMC sampling code. It',
            'exists in prototype format as a series of functions.',
            'See `bnn_exp.py`',
        ]))

    def predict(
        self,
        given_samples,
        weight_sets,
        data_dim_first=True,
        sess_config=None,
    ):
        """Returns the predictions of the BNN with the given weight_sets.

        Returns
        -------
        np.ndarray
            A list or an array of shape (num_given_samples,
            num_valid_weight_sets_per_given_sample, discrete distrib
            dimensions). In the case of Euclidean transform, the returned
            array may contain some invalid discrete probability distributions.
        """
        if sess_config is None and self.sess_config is not None:
            sess_config = self.sess_config

        return bnn_transform.assign_weights_bnn(
            weight_sets,
            self.weight_placeholders,
            self.output,
            given_samples,
            self.input,
            dtype=self.dtype,
            sess_config=sess_config, # TODO, replace with class attrib?
            data_dim_first=data_dim_first,
        )
