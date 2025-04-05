"""Functions for performing distribution model selection and helper
functions.
"""
import logging

import numpy as np
import tensorflow as tf

from psych_metric.distrib.tf_distrib_vars import get_distrib_param_vars
from psych_metric.distrib import mle_utils


def mle_adam(
    distrib_id,
    data,
    init_params=None,
    const_params=None,
    optimizer_args=None,
    num_top_likelihoods=1,
    max_iter=10000,
    tol_param=1e-8,
    tol_loss=1e-8,
    tol_grad=1e-8,
    tb_summary_dir=None,
    random_seed=None,
    shuffle_data=True,
    batch_size=1,
    name='MLE_adam',
    sess_config=None,
    optimizer_id='adam',
    tol_chain=1,
    alt_distrib=False,
    constraint_multiplier=1e5,
    dtype=tf.float32,
):
    """Uses tensorflow's ADAM optimizer to search the parameter space for MLE.

    Parameters
    ----------
    distrib_id : str
        Name of the distribution whoe MLE is being found.
    data : np.ndarray
    init_params : dict, optional
        The initial parameters of the distribution. Otherwise, selected randomly.
    const_params : list(str), optional
        The names of the parameters to be kept constant, all others are assumed
        to be variables.
    optimizer_args : dict, optional
    num_top_likelihoods : int, optional
        The number of top best likelihoods and their respective parameters. If
        equal to or less than -1, returns full history of likelihoods.
    tb_summary_dir : str, optional
        directory path to save TensorBoard summaries.
    name : str
        Name prefixed to Ops created by this class.
    alt_distrib : bool
        whether to use the alternate version of parameterization of the given
        distribution.
    constraint_multiplier : float, optional
        The multiplier to use to enforce constraints on the params in the loss.
        Typically a large positive value.

    Returns
    -------
    top_likelihoods
        the `num_top_likelihoods` likelihoods and their respective parameters
        in decending order.
    """
    if optimizer_args is None:
        # Ensure that opt args is a dict for use with **
        optimizer_args = {}
    if random_seed:
        # Set random seed if given.
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

    # tensor to hold the data
    with tf.name_scope(name) as scope:
        # tensorflow prep data
        dataset = tf.data.Dataset.from_tensor_slices(data)
        if shuffle_data:
            # NOTE batchsize is default to 1, is this acceptable?
            dataset = dataset.shuffle(batch_size, seed=random_seed)
        dataset = dataset.repeat(max_iter)
        # TODO No batching of this process... Decide if batching is okay
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        tf_data = tf.cast(iterator.get_next(), dtype)

        # create distribution and dict of the distribution's parameters
        distrib, params = get_distrib_param_vars(
            distrib_id,
            init_params,
            const_params,
            alt_distrib=alt_distrib,
        )

        neg_log_prob, loss = mle_utils.get_mle_loss(
            data,
            distrib,
            params,
            const_params,
            alt_distrib=alt_distrib,
            constraint_multiplier=constraint_multiplier,
            neg_loss=True,
        )

        # Create optimizer
        if optimizer_id == 'adam':
            optimizer = tf.train.AdamOptimizer(**optimizer_args)
        elif optimizer_id == 'nadam':
            optimizer = tf.contrib.opt.NadamOptimizer(**optimizer_args)
        else:
            raise ValueError(f'Unexpected optimizer_id value: {optimizer_id}')

        global_step = tf.Variable(0, name='global_step', trainable=False)

        if const_params:
            grad = optimizer.compute_gradients(
                loss,
                [v for k, v in params.items() if k not in const_params],
            )
        else:
            grad = optimizer.compute_gradients(
                loss,
                list(params.values()),
            )
        train_op = optimizer.apply_gradients(grad, global_step)

        # Create the dictionary of tensors whose results are desired
        results_dict = {
            'train_op': train_op,
            'loss': loss,
            'neg_log_prob': neg_log_prob,
            'params': params,
            'grad': grad,
        }

        if tb_summary_dir:
            # Visualize the gradients
            for g in grad:
                tf.summary.histogram(f'{g[1].name}-grad', g[0])

            results_dict['summary_op'] = tf.summary.merge_all()

    return mle_utils.run_session(
        distrib_id,
        results_dict,
        tf_data,
        data,
        params,
        const_params,
        num_top=num_top_likelihoods,
        max_iter=max_iter,
        sess_config=sess_config,
        tb_summary_dir=tb_summary_dir,
    )


#def alternating_adam(
def coordinate_descent_adam(
    distrib_id,
    data,
    init_params=None,
    const_params=None,
    optimizer_args=None,
    num_top_likelihoods=1,
    max_iter=10000,
    tol_param=1e-8,
    tol_loss=1e-8,
    tol_grad=1e-8,
    tb_summary_dir=None,
    random_seed=None,
    shuffle_data=True,
    batch_size=1,
    name='mle_coordinate_descent_adam',
    sess_config=None,
    optimizer_id='adam',
    tol_chain=1,
    alt_distrib=False,
    constraint_multiplier=1e5,
    dtype=tf.float32,
    alternation_method='cyclic',
    max_iter_per_coordinate=False,
):
    """Performs Coordinate Descent using ADAM to find the MLE of a distribution
    on the data where parameters being optimized are focused one at a time
    while the rest are held constant.

    May do Coordinate Gradient Descent, where a set of parameters are held
    constant as another set are optimized, and then alternate.
    """
    # TODO implement adam with swapping of at least 2 params.
    raise NotImplementedError

    # Good for MVST when loc is constant, then swap between df and scale

    # Good for MVCauchy when no variable is constant
    with tf.name_scope(name) as scope:
        # Create optimizer
        if optimizer_id == 'adam':
            optimizer = tf.train.AdamOptimizer(**optimizer_args)
        elif optimizer_id == 'nadam':
            optimizer = tf.contrib.opt.NadamOptimizer(**optimizer_args)
        else:
            raise ValueError(f'Unexpected optimizer_id value: {optimizer_id}')

        global_step = tf.Variable(0, name='global_step', trainable=False)

        if const_params:
            grad = optimizer.compute_gradients(
                loss,
                [v for k, v in params.items() if k not in const_params],
            )
        else:
            grad = optimizer.compute_gradients(
                loss,
                list(params.values()),
            )


        # TODO apply gradients selectively, boolean switch for 2 coord.
        # iterate through the "coordinates" one at a time.  Full pass on data
        # each time.
        train_op = optimizer.apply_gradients(grad, global_step)


        # Create the dictionary of tensors whose results are desired
        results_dict = {
            'train_op': train_op,
            'loss': loss,
            'neg_log_prob': neg_log_prob,
            'params': params,
            'grad': grad,
        }

        if tb_summary_dir:
            # Visualize the gradients
            for g in grad:
                tf.summary.histogram(f'{g[1].name}-grad', g[0])

            results_dict['summary_op'] = tf.summary.merge_all()
