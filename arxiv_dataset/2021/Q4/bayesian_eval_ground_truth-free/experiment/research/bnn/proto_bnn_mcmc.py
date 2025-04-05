"""Prototyping bnn mcmc on CRC."""
from datetime import datetime
import logging
import os
import json
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.tsa.stattools import acf
import tensorflow as tf
import tensorflow_probability as tfp

from psych_metric.distrib import bnn_transform

from experiment.research.bnn import bnn_exp
from experiment import io
from experiment.research.sjd import src_candidates


def setup_rwm_sim(
    target_log_prob,
    width=10,
    sample_size=10,
    #scale_identity_multiplier=0.01,
    dim=3,
    src_id='tight_dir_small_mvn',
):
    rdm = src_candidates.get_src_sjd(src_id, dim)
    givens, conditionals = rdm.sample(sample_size)
    #data = rdm.transform_distrib.simplex_transform.to(s[0])
    #targets = rdm.transform_distrib.simplex_transform.to(s[1])

    init_state = [
        np.random.normal(scale=12**0.5 , size=(dim-1,width)).astype(np.float32),
        np.zeros([width], dtype=np.float32),
        np.random.normal(scale=0.48**0.5 , size=(width,dim-1)).astype(np.float32),
        np.zeros([dim-1], dtype=np.float32)]

    sample_log_prob = partial(
        target_log_prob,
        origin_adjust=rdm.transform_distrib.simplex_transform.origin_adjust.astype(np.float32),
        rotation_mat=rdm.transform_distrib.simplex_transform.change_of_basis_matrix.astype(np.float32),
    )

    #return data, targets, sample_log_prob, init_state
    return givens, conditionals, sample_log_prob, init_state, rdm


def adam_init(data, targets, width, dim, epochs, cpus=1, cpu_cores=16, gpus=0, init_vars=None):
    config = io.get_tf_config(cpus, cpu_cores, gpus)

    tf_in = tf.placeholder(dtype=tf.float32, shape=[None, data.shape[1]])
    tf_out = tf.placeholder(dtype=tf.float32, shape=[None, data.shape[1]])

    feed_dict = {tf_in: data, tf_out: targets}

    bnn_out, tf_vars = bnn_transform.bnn_mlp(
        tf_in,
        1,
        width,
        output_activation=None,
        output_use_bias=True,
    )

    if init_vars is not None:
        init_vars = {tf_vars[i]:init_vars[i] for i in range(len(init_vars))}

    new_state, iter_results = bnn_transform.bnn_adam(
        bnn_out,
        tf_vars,
        tf_out,
        feed_dict,
        epochs=epochs,
        tf_config=config,
        init_vars=init_vars,
    )

    return new_state, iter_results['loss']


def run_rwm(
    data,
    targets,
    sample_log_prob,
    init_state,
    num_results=10000,
    burnin=0,
    lag=0,
    rwm_scale=2e-4,
    config=None,
    parallel=1,
):
    kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=lambda x,y,z,q: sample_log_prob((x,y,z,q),data,targets),
        new_state_fn = tfp.mcmc.random_walk_normal_fn(scale=rwm_scale),
    )

    return sample_chain_run(
        init_state,
        kernel,
        num_results,
        burnin,
        lag,
        config=config,
        parallel=parallel,
    )


def run_hmc(
    data,
    targets,
    sample_log_prob,
    init_state,
    num_results=10000,
    burnin=0,
    lag=0,
    step_size=5e-4,
    num_leapfrog_steps=5,
    num_adaptation_steps=None,
    step_adjust_id='Simple',
    config=None,
    parallel=1,
    target_accept_prob=0.75,
):
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=lambda x,y,z,q: sample_log_prob((x,y,z,q),data,targets),
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )
    if num_adaptation_steps:
        if step_adjust_id == 'Simple':
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                kernel,
                num_adaptation_steps=num_adaptation_steps,
                target_accept_prob=target_accept_prob,
            )
        else:
            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                kernel,
                num_adaptation_steps=num_adaptation_steps,
                target_accept_prob=target_accept_prob,
            )

    return sample_chain_run(
        init_state,
        kernel,
        num_results,
        burnin,
        lag,
        config=config,
        parallel=parallel,
    )


def run_nuts(
    data,
    targets,
    sample_log_prob,
    init_state,
    num_results=10000,
    burnin=0,
    lag=0,
    step_size=5e-4,
    num_adaptation_steps=0,
    step_adjust_id='Simple',
    config=None,
    parallel=1,
    target_accept_prob=0.75,
):
    kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=lambda x,y,z,q: sample_log_prob((x,y,z,q),data,targets),
        step_size=step_size,
    )

    if num_adaptation_steps > 0:
        # TODO setup NUTS to use adaptative step sizes
        trans_kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=kernel,
            bijector=[tfp.bijectors.Identity()] * len(init_state),
        )

        if step_adjust_id == 'Simple':
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                inner_kernel=trans_kernel,
                num_adaptation_steps=num_adaptation_steps,
                target_accept_prob=target_accept_prob,
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                    inner_results=pkr.inner_results._replace(step_size=new_step_size)
                ),
                step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
            )
        else:
            kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=trans_kernel,
                num_adaptation_steps=num_adaptation_steps,
                target_accept_prob=target_accept_prob,
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                    inner_results=pkr.inner_results._replace(step_size=new_step_size)
                ),
                step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
            )

    return sample_chain_run(
        init_state,
        kernel,
        num_results,
        burnin,
        lag,
        config=config,
        parallel=parallel,
    )


def sample_chain_run(
    init_state,
    kernel,
    num_results=10000,
    burnin=0,
    lag=0,
    config=None,
    parallel=1,
):
    if parallel <= 1:
        samples, trace = tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=init_state,
            kernel=kernel,
            num_burnin_steps=burnin,
            num_steps_between_results=lag,
            parallel_iterations=1,
        )
        with tf.Session(config=config) as sess:
            sess.run((
                tf.global_variables_initializer(),
                tf.local_variables_initializer(),
            ))

            output = sess.run([samples,trace])
            new_starting_state = [x[-1] for x in output[0]]

        return output, new_starting_state
    else:
        # NOTE parallel chains saves only is_accepted for trace_fn
        # create multiple of the same chains, w/ diff seeds, gets samples fast
        chains_results = []
        for i in range(parallel):
            chains_results.append(tfp.mcmc.sample_chain(
                num_results=num_results,
                current_state=init_state,
                kernel=kernel,
                num_burnin_steps=burnin,
                num_steps_between_results=lag,
                parallel_iterations=1,
                trace_fn=lambda current_state, kernel_results: kernel_results.is_accepted,
            ))

        with tf.Session(config=config) as sess:
            sess.run((
                tf.global_variables_initializer(),
                tf.local_variables_initializer(),
            ))
            output = sess.run(chains_results)

        return output


def save_stats(
    output_dir,
    weights_sets,
    is_accepted,
    acf_log_prob,
    final_step_size,
    burnin,
    lag,
    num_hidden_units,
    num_layers,
    scale_identity_multiplier,
    input_output_dim,
    bnn_weights_file,
    data_file,
    log_prob_linregress=None,
    acf_thresholds=[.5, .4, .3, .2, .1, .05, .01],
    first_n_acf=10,
    note=None,
    datetime_fmt='%Y-%m-%d_%H-%M-%S',
):
    """Saves important information of the MCMC chain runs."""
    io.save_json(
        os.path.join(output_dir, 'last_weights.json'),
        weights_sets,
    )

    idx_30percent = int(np.ceil(len(is_accepted) * .3))

    acf_lag = {
        'datetime_saving': datetime.now().strftime(datetime_fmt),
        'paths': {
            'bnn_weights_file': bnn_weights_file,
            'data_file': data_file,
        },
        'mcmc':{
            'params':{
                'final_step_size': final_step_size,
                'burnin': burnin,
                'lag': lag,
            },
            'results':{
                'accept_total': is_accepted.sum(),
                'accept_rate': {
                    'overall': is_accepted.mean(),
                    'first_30%': is_accepted[:idx_30percent].mean(),
                    'in_between':
                        is_accepted[idx_30percent : -idx_30percent].mean(),
                    'last_30%': is_accepted[-idx_30percent:].mean(),
                },
                'acf_thresholds': {
                    x: np.where(np.abs(acf_log_prob) <= x)[0][:first_n_acf]
                    for x in acf_thresholds
                },
            },
            'bnn':{
                'num_hidden_units': num_hidden_units,
                'num_layers': num_layers,
                'input_output_dim': input_output_dim,
                'scale_identity_multiplier': scale_identity_multiplier,
            },
        },
    }

    if note is not None:
        acf_lag['note'] = note

    if log_prob_linregress is not None:
        acf_lag['target_func_linreg'] = {
            'slope': log_prob_linregress.slope,
            'intercept': log_prob_linregress.intercept,
            'rvalue': log_prob_linregress.rvalue,
            'pvalue': log_prob_linregress.pvalue,
            'stderr': log_prob_linregress.stderr,
        }

    io.save_json(
        os.path.join(output_dir, 'acf_lag.json'),
        acf_lag,
    )


def add_custom_args(parser):
    bnn_exp.add_custom_args(parser)

    parser.add_argument(
        '--num_samples',
        default=10,
        type=int,
        help='The number of src simulation samples.',
    )

    parser.add_argument(
        '--dim',
        default=None,
        type=int,
        help='The number of dimensions of the discrete distribution data (input and output).',
    )

    parser.add_argument(
        '--parallel_chains',
        default=1,
        type=int,
        help='The number of chains to be run in parallel for sampling.',
    )

    parser.add_argument(
        '--bnn_weights_file',
        default=None,
        help='Path to the bnn weights file.',
    )

    parser.add_argument(
        '--src_sjd_id',
        default='tight_dir_small_mvn',
        help='The id of joint distribution to use in simulation of samples.',
    )

    parser.add_argument(
        '--no_visuals',
        action='store_true',
        help='Does not save visuals if single chain (single process).',
    )

    parser.add_argument(
        '--target_accept_prob',
        default=0.75,
        type=float,
        help='The target accept probability for MCMC step adjustment methods.',
    )

    parser.add_argument(
        '--random_bnn_init',
        action='store_true',
        help='If given, and dataset_filepath is a file and no bnn weights given, then the bnn weights are randomly initialized.',
    )

    parser.add_argument(
        '--mcmc_target_log_prob',
        default='mcmc_sample_log_prob',
        choices=['mcmc_sample_log_prob', 'l2_dist'],
        help='The target log prob function to use for the MCMC.',
    )

    parser.add_argument(
        '--note',
        default=None,
        type=str,
        help='A Note to be added in the results JSON.',
    )


if __name__ == '__main__':
    args = io.parse_args(custom_args=add_custom_args)

    # Create directory
    output_dir = args.output_dir
    output_dir = io.create_dirs(output_dir)
    logging.info('Created the output directories')

    if args.mcmc_target_log_prob == 'l2_dist':
        target_log_prob = bnn_transform.l2_dist
    else:
        # defaults to bnn_transform.mcmc_sample_log_prob (regression MVN)
        target_log_prob = partial(
            bnn_transform.mcmc_sample_log_prob,
            scale_identity_multiplier=args.mcmc.scale_identity_multiplier,
        )

    if os.path.isfile(args.data.dataset_filepath):
        with open(args.data.dataset_filepath, 'r') as f:
            data = json.load(f)
            givens = np.array(data['givens'], dtype=np.float32)
            conditionals = np.array(data['conditionals'], dtype=np.float32)
            change_of_basis = np.array(data['change_of_basis'], dtype=np.float32)
            origin_adjust = np.array(data['origin_adjust'], dtype=np.float32)

        #scale_identity_multiplier=args.mcmc.scale_identity_multiplier,
        sample_log_prob = partial(
            target_log_prob,
            origin_adjust=origin_adjust,
            rotation_mat=change_of_basis,
            #scale_identity_multiplier=args.mcmc.scale_identity_multiplier,
        )

        if args.random_bnn_init:
            init_state = [
                np.random.normal(
                    scale=12**0.5,
                    size=(givens.shape[1]-1,args.bnn.num_hidden),
                ).astype(np.float32),
                np.zeros([args.bnn.num_hidden], dtype=np.float32),
                np.random.normal(
                    scale=0.48**0.5 ,
                    size=(args.bnn.num_hidden, givens.shape[1]-1),
                ).astype(np.float32),
                np.zeros([givens.shape[1]-1], dtype=np.float32),
            ]
        elif not isinstance(args.bnn_weights_file, str):
            raise TypeError('bnn_weights_file must be provided when dataset_filepath is given')
        else:
            with open(args.bnn_weights_file, 'r') as f:
                init_state = json.load(f)
                init_state = [np.array(x, dtype=np.float32) for x in init_state]

                if args.bnn.num_hidden != init_state[0].shape[1]:
                    raise ValueError(' '.join([
                        '`args.bnn.num_hidden` does not equal the',
                        'number of hidden units in the loaded weights file!',
                        f'`args.bnn.num_hidden` = {args.bnn.num_hidden}',
                        '`init_state[0].shape[1]` =',
                        f'{init_state[0].shape[1]}',
                    ]))
    else:
        if args.parallel_chains > 1:
            raise ValueError('The dataset_filepath is an invalid filepath.')

        if args.dim is None:
            raise ValueError('Must provide dim when doing simulation runs.')

        givens, conditionals, sample_log_prob, init_state, src = setup_rwm_sim(
            target_log_prob,
            width=args.bnn.num_hidden,
            sample_size=args.num_samples,
            #scale_identity_multiplier=args.mcmc.scale_identity_multiplier,
            dim=args.dim,
            src_id=args.src_sjd_id,
        )

        io.save_json(
            os.path.join(output_dir, 'data_for_bnn.json'),
            {
                'givens': givens,
                'conditionals': conditionals,
                'src_sjd': src.params,
                'change_of_basis': src.transform_distrib.simplex_transform.change_of_basis_matrix,
                'origin_adjust': src.transform_distrib.simplex_transform.origin_adjust,
            },
        )

    logging.info('Setup the simulation data and the log prob function')

    if args.adam_epochs > 0:
        logging.info('Starting ADAM initialization training')
        if args.bnn_weights_file:
            init_vars = init_state
        init_state, loss = adam_init(
            givens,
            conditionals,
            args.bnn.num_hidden,
            givens.shape[1],
            args.adam_epochs,
            cpus=args.cpu,
            cpu_cores=args.cpu_cores,
            gpus=args.gpu,
            init_vars=init_vars
        )
        logging.info('Finished ADAM initialization training')

    config = io.get_tf_config(args.cpu, args.cpu_cores, 0)

    logging.info('Starting MCMC training')

    if args.mcmc.kernel_id == 'RandomWalkMetropolis':
        output, new_starting_state = run_rwm(
            givens,
            conditionals,
            sample_log_prob,
            init_state,
            num_results=args.mcmc.sample_chain.num_results,
            burnin=args.mcmc.sample_chain.burnin,
            lag=args.mcmc.sample_chain.lag,
            rwm_scale=args.mcmc.kernel.step_size,
            config=config,
        )
        logging.info('Finished RandomWalkMetropolis')

        acf_log_prob = acf(
            output[1].accepted_results.target_log_prob,
            nlags=int(args.mcmc.sample_chain.num_results / 4),
        )

        final_step_size = args.mcmc.kernel.step_size

        save_stats(
            output_dir,
            new_starting_state,
            output[1].is_accepted,
            acf_log_prob,
            final_step_size,
            args.mcmc.sample_chain.burnin,
            args.mcmc.sample_chain.lag,
            args.bnn.num_hidden,
            args.bnn.num_layers,
            args.mcmc.scale_identity_multiplier,
            givens.shape[1],
            args.bnn_weights_file,
            args.data.dataset_filepath,
            note=args.note,
            #log_prob_linreg=scipy.stats.linregress(
            #    np.arange(args.mcmc.sample_chain.num_results),
            #    mcmc_results.accepted_results.target_log_prob,
            #),
        )

        if not args.no_visuals:
            logging.info('Starting RandomWalkMetropolis specific visuals')
            plt.plot(output[1].accepted_results.target_log_prob)
            plt.savefig(os.path.join(output_dir, 'log_prob.png'), dpi=400, bbox_inches='tight')
            plt.close()

            pd.DataFrame(acf_log_prob).plot(kind='bar')
            plt.savefig(os.path.join(output_dir, 'log_prob_acf_fourth.png'), dpi=400, bbox_inches='tight')
            plt.close()

    elif args.mcmc.kernel_id == 'HamiltonianMonteCarlo':
        output = run_hmc(
            givens,
            conditionals,
            sample_log_prob,
            init_state,
            num_results=args.mcmc.sample_chain.num_results,
            burnin=args.mcmc.sample_chain.burnin,
            lag=args.mcmc.sample_chain.lag,
            step_size=args.mcmc.kernel.step_size,
            num_leapfrog_steps=args.mcmc.kernel.num_leapfrog_steps,
            num_adaptation_steps=args.mcmc.num_adaptation_steps,
            config=config,
            step_adjust_id=args.mcmc.step_adjust_id,
            parallel=args.parallel_chains,
            target_accept_prob=args.target_accept_prob,
        )
        logging.info('Finished HamiltonianMonteCarlo')

        if args.parallel_chains <= 1:
            output, new_starting_state = output

            if args.mcmc.num_adaptation_steps > 0:
                mcmc_results = output[1].inner_results
                final_step_size = output[1].new_step_size[-1]
            else:
                mcmc_results = output[1]
                final_step_size = args.mcmc.kernel.step_size

            acf_log_prob = acf(
                mcmc_results.accepted_results.target_log_prob,
                nlags=int(args.mcmc.sample_chain.num_results / 4),
            )

            save_stats(
                output_dir,
                new_starting_state,
                mcmc_results.is_accepted,
                acf_log_prob,
                final_step_size,
                args.mcmc.sample_chain.burnin,
                args.mcmc.sample_chain.lag,
                args.bnn.num_hidden,
                args.bnn.num_layers,
                args.mcmc.scale_identity_multiplier,
                givens.shape[1],
                args.bnn_weights_file,
                args.data.dataset_filepath,
                log_prob_linregress=scipy.stats.linregress(
                    np.arange(args.mcmc.sample_chain.num_results),
                    mcmc_results.accepted_results.target_log_prob,
                ),
                note=args.note,
            )


            if not args.no_visuals:
                logging.info('Starting HamiltonianMonteCarlo specific visuals')
                plt.plot(mcmc_results.accepted_results.target_log_prob)
                plt.savefig(os.path.join(output_dir, 'log_prob.png'), dpi=400, bbox_inches='tight')
                plt.close()

                pd.DataFrame(acf_log_prob).plot(kind='bar')
                plt.savefig(os.path.join(output_dir, 'log_prob_acf_fourth.png'), dpi=400, bbox_inches='tight')
                plt.close()
        else:
            sampled_weights = {}
            for i in range(0, len(output)):
                sampled_weights[i] = {}
                sampled_weights[i]['weights'] = output[i].all_states
                # the trace is only boolean is_accepted
                sampled_weights[i]['is_accepted'] = output[i].trace

            io.save_json(
                os.path.join(output_dir, 'sampled_weights.json'),
                sampled_weights,
            )

    elif args.mcmc.kernel_id == 'NoUTurnSampler':
        output, new_starting_state = run_nuts(
            givens,
            conditionals,
            sample_log_prob,
            init_state,
            num_results=args.mcmc.sample_chain.num_results,
            burnin=args.mcmc.sample_chain.burnin,
            lag=args.mcmc.sample_chain.lag,
            step_size=args.mcmc.kernel.step_size, # 5e-4
            config=config,
            num_adaptation_steps=args.mcmc.num_adaptation_steps,
            step_adjust_id=args.mcmc.step_adjust_id,
            target_accept_prob=args.target_accept_prob,
        )
        logging.info('Finished NoUTurnSampler')

        if args.mcmc.num_adaptation_steps > 0:
            # Need to extract from Adaptive step and from Transformed kernels
            mcmc_results = output[1].inner_results.inner_results
            final_step_size = output[1].new_step_size[-1]
        else:
            mcmc_results = output[1]
            final_step_size = args.mcmc.kernel.step_size

        acf_log_prob = acf(
            mcmc_results.target_log_prob,
            nlags=int(args.mcmc.sample_chain.num_results / 4),
        )

        if args.parallel_chains <= 1:
            save_stats(
                output_dir,
                new_starting_state,
                mcmc_results.is_accepted,
                acf_log_prob,
                final_step_size,
                args.mcmc.sample_chain.burnin,
                args.mcmc.sample_chain.lag,
                args.bnn.num_hidden,
                args.bnn.num_layers,
                args.mcmc.scale_identity_multiplier,
                givens.shape[1],
                args.bnn_weights_file,
                args.data.dataset_filepath,
                log_prob_linregress=scipy.stats.linregress(
                    np.arange(args.mcmc.sample_chain.num_results),
                    mcmc_results.target_log_prob
                ),
                note=args.note,
            )

        if not args.no_visuals:
            logging.info('Starting NoUTurnSampler specific visuals')
            plt.plot(mcmc_results.target_log_prob)
            plt.savefig(os.path.join(output_dir, 'log_prob.png'), dpi=400, bbox_inches='tight')
            plt.close()

            pd.DataFrame(acf_log_prob).plot(kind='bar')
            plt.savefig(os.path.join(output_dir, 'log_prob_acf_fourth.png'), dpi=400, bbox_inches='tight')
            plt.close()

    logging.info('Finished MCMC training and specific kernel givens saving.')
