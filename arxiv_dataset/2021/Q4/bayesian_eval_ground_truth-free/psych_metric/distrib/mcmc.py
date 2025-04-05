"""The Tensorflow optimization of either a distribution or a Bayesian Neural
Network using MCMC methods.
"""
import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib.distrib_utils import get_tfp_distrib
from psych_metric.distrib import mle_utils

# TODO add tensorboard for visualizing and keeping track of progress.

def mcmc_distrib_params(
    distrib_id,
    data,
    params,
    const_params=None,
    kernel_id='NoUTurnSampler',
    kernel_args=None,
    step_adjust_args=None,
    num_top=1,
    num_samples=int(1e4),
    burnin=int(1e5),
    lag=int(1e4),
    parallel_iter=10,
    dtype=tf.float32,
    random_seed=None,
    alt_distrib=False,
    constraint_multiplier=1e5,
    sess_config=None,
):
    """Performs MCMC over the parameter distributions and returns the parameter
    set with the highest maxmimum likelihood estimate.
    """
    if kernel_args is None:
        # Ensure that opt args is a dict for use with **
        # TODO decide on default values for the kernel args
        if kernel_id == 'RandomWalkMetropolis':
            kernel_args = {'scale': 0.5}
        elif kernel_id == 'HamiltonianMonteCarlo':
            kernel_args = {
                'step_size': 0.001, # initial step size
                'num_leapfrog_steps': 2,
            }
        elif kernel_id == 'NoUTurnSampler':
            kernel_args = {
                'step_size': 0.1, # initial step size
                'num_leapfrog_steps': 2,
            }

        # create default step adjust args?
        if step_adjust_args is None:
            step_adjust_args = {
                'num_adaptation_steps': np.floor(burnin * 0.6),
            }
    if random_seed:
        # Set random seed if given.
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

    if const_params is None:
        const_params = {}
    elif isinstance(const_params, list):
        const_params = {k:v for k, v in params.items() if k in const_params}

    loss_fn = lambda x: mle_utils.tfp_log_prob(
        x,
        const_params,
        data,
        get_tfp_distrib(distrib_id),
        lambda y: mle_utils.unpack_mvst_params(
            y,
            data.shape[1],
            'df' not in const_params,
            'loc' not in const_params,
            'scale' not in const_params and 'sigma' not in const_params,
        ),
    )

    kernel = get_mcmc_kernel(loss_fn, kernel_id, kernel_args)

    current_state = mle_utils.pack_mvst_params(params, const_params)

    # TODO consider running multiple of these sample chains w/ diff inits.
    samples, trace = tfp.mcmc.sample_chain(
        num_results=num_samples,
        current_state=current_state,
        kernel=kernel,
        num_burnin_steps=burnin,
        num_steps_between_results=lag,
        parallel_iterations=parallel_iter,
    )

    # TODO get maximum likelihood estimate parameter from parameter set
    log_probs = tf.map_fn(
        loss_fn,
        samples,
        parallel_iterations=parallel_iter,
        back_prop=False,
    )

    with tf.Session(config=sess_config) as sess:
        sess.run((
            tf.global_variables_initializer(),
            tf.local_variables_initializer(),
        ))

        iter_results = sess.run({
            'log_probs': log_probs,
            'samples': samples,
            'trace': trace,
        })

    max_idx = np.argmax(iter_results['log_probs'])

    return iter_results, max_idx
    #return iter_results['log_probs'][max_idx], iter_results['samples'][max_idx]


def get_mcmc_kernel(
    loss_fn,
    kernel_id,
    kernel_args,
    step_adjust_id='Simple',
    step_adjust_args=None,
):
    """Creates the MCMC kernel used to fit a distribution."""
    kernel_id = kernel_id.lower()

    if kernel_id == 'randomwalk' or kernel_id == 'randomwalkmetropolis':
        return tfp.mcmc.RandomWalkMetropolis(
            loss_fn,
            tfp.mcmc.random_walk_uniform_fn(kernel_args['scale']),
        )
    if kernel_id == 'nuts' or kernel_id == 'nouturnsampler':
        nuts = tfp.mcmc.NoUTurnSampler(loss_fn, **kernel_args)

        if step_adjust_args:
            if step_adjust_id == 'DualAveraging':
                return tfp.mcmc.DualAveragingStepSizeAdaptation(
                    nuts,
                    **step_adjust_args,
                )
            return tfp.mcmc.SimpleStepSizeAdaptation(nuts, **step_adjust_args)
        return nuts
    if (
        kernel_id == 'hmc'
        or kernel_id == 'hmcmc'
        or kernel_id == 'hamiltonianmontecarlo'
    ):
        hmc = tfp.mcmc.HamiltonianMonteCarlo(loss_fn, **kernel_args)

        if step_adjust_args:
            if step_adjust_id == 'DualAveraging':
                return tfp.mcmc.DualAveragingStepSizeAdaptation(
                    hmc,
                    **step_adjust_args,
                )
            return tfp.mcmc.SimpleStepSizeAdaptation(hmc, **step_adjust_args)
        return hmc

    raise ValueError(f'Unexpected value for `kernel_id`: {kernel_id}')
