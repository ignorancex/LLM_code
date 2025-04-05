import os
import json

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
import tensorflow as tf

from psych_metric.distrib import bnn_transform

from experiment import io
from experiment.research.sjd import src_candidates


def bnn_mcmc(
    bnn_args,
    src_samples=10,
    adam_epochs=15,
    cpu_cores=1,
    gpus=0,
    mcmc_args=None,
):
    """Tests the BNN MCMC on simulated data"""
    rdm = src_candidates.get_src_sjd('tight_dir_small_mvn', 4)
    s = rdm.sample(src_samples)
    ss0= (s[0] - [1, 0, 0, 0]) @ rdm.transform_matrix.T
    ss1= (s[1] - [1, 0, 0, 0]) @ rdm.transform_matrix.T
    #bnn_args={'output_activation': None, 'num_layers':1, 'num_hidden': 100}
    tf_input = tf.placeholder(
        dtype=tf.float32,
        shape=[None, ss0.shape[1]],
        name='input_label',
    )

    bnn_out, tf_vars = bnn_transform.bnn_mlp(tf_input, **bnn_args)
    tf_labels = tf.placeholder(
        dtype=tf.float32,
        shape=[None, ss1.shape[1]],
        name='output_labels',
    )
    feed_dict = {
        tf_input: ss0,
        tf_labels: ss1,
    }
    cpu_config = io.get_tf_config(cpu_cores)
    gpu_config = io.get_tf_config(cpu_cores, gpus=gpus)

    if adam_epochs > 0:
        weights, badir = bnn_transform.bnn_adam(
            bnn_out,
            tf_vars,
            tf_labels,
            feed_dict,
            gpu_config,
            epochs=adam_epochs,
        )
    else:
        weights = None

    results_dict, feed_dict = bnn_transform.get_bnn_transform(
        ss0,
        ss1,
        bnn_args=bnn_args,
        tf_vars_init=weights,
        tf_input=tf_input,
        **mcmc_args,
    )

    iter_results = bnn_transform.bnn_mlp_run_sess(
        results_dict,
        feed_dict,
        sess_config=cpu_config,
    )

    return locals()


def get_bnn_output(
    last_weights,
    path=None,
    data=None,
):
    # NOTE mostly just for remembering how to do this later on.
    if path and data is None and targets is None:
        with open(os.path.join(path, 'data.json'), 'r') as f:
            data = json.load(f)
            targets = np.array(data['output'])
            data = np.array(data['input'])

    if path and last_weights is None:
        with open(os.path.join(path, 'last_weights.json'), 'r') as f:
            last_weights = [np.array(w, dtype=np.float32) for w in json.load(f)]


    tf_in = tf.placeholder(
        dtype=tf.float32,
        shape=[None, data.shape[1]],
        name='input_label',
    )

    if bnn_ph is None and tf_ph is None:
        bnn_ph, tf_ph = bnn_transform.bnn_mlp_placeholders(
            data,
            1,
            10,
            output_use_bias=True,
            output_activation=None,
        )

    bnn_output = bnn_transform.assign_weights_bnn(
        [last_weights],
        tf_ph,
        bnn_ph,
        data,
        tf_in,
    )

    return bnn_output


def add_mcmc_args(parser):
    """Adds the MCMC related args."""
    mcmc = parser.add_argument_group(
        'mcmc',
        'Arguments pertaining to the Markov chain Monte Carlo sample chain',
    )

    mcmc.add_argument(
        #'--diff_scale',
        '--scale_identity_multiplier',
        default=0.01,
        type=float,
        help=' '.join([
            'The scalar multiplier to use on the end of the BNN log',
            'probability in the MCMC fitting. Use -1.0 to negate the loss.',
        ]),
        #dest='mcmc.diff_scale',
        dest='mcmc.scale_identity_multiplier',
    )

    # Sample chain
    sample_chain = parser.add_argument_group(
        'sample_chain',
        'Arguments pertaining to the Markov chain Monte Carlo kernel',
    )

    sample_chain.add_argument(
        '--num_results',
        default=1000,
        type=int,
        help='The number of samples obtained from the MCMC sample chain.',
        dest='mcmc.sample_chain.num_results',
    )

    sample_chain.add_argument(
        '--burnin',
        default=0,
        type=int,
        help='The number of burn in samples for the MCMC sample chain.',
        dest='mcmc.sample_chain.burnin',
    )

    sample_chain.add_argument(
        '--lag',
        default=0,
        type=int,
        help=' '.join([
            'The number of samples to skip before the next sample is accepted',
            'for the MCMC sample chain.',
        ]),
        dest='mcmc.sample_chain.lag',
    )

    sample_chain.add_argument(
        '--parallel_iter',
        default=1,
        type=int,
        help='The number of MCMC sample chains to run in parallel.',
        dest='mcmc.sample_chain.parallel_iter',
    )


    # Kernel Arguments (dependent on which kernel was selected)
    mcmc.add_argument(
        '--kernel_id',
        default='RandomWalkMetropolis',
        choices=[
            'RandomWalkMetropolis',
            'HamiltonianMonteCarlo',
            'NoUTurnSampler',
        ],
        help='The MCMC transition kernel.',
        dest='mcmc.kernel_id',
    )

    kernel = parser.add_argument_group(
        'kernel',
        'Arguments pertaining to the Markov chain Monte Carlo kernel',
    )
    kernel.add_argument(
        '--step_size',
        default=0.01,
        type=float,
        help=' '.join([
            'The MCMC transition kernel step size (for HMC or NUTS, is scale',
            'of next state in RWM).',
        ]),
        dest='mcmc.kernel.step_size',
    )
    kernel.add_argument(
        '--num_leapfrog_steps',
        default=10,
        type=int,
        help='The number of leap frog steps for HMC.',
        dest='mcmc.kernel.num_leapfrog_steps',
    )

    # Step Size Adaptation
    mcmc.add_argument(
        '--step_adjust_id',
        default=None,
        choices=[
            'DualAveraging',
            'Simple',
        ],
        help='The identifier of which Step Size Adaptation method to use.',
        dest='mcmc.step_adjust_id',
    )

    mcmc.add_argument(
        '--step_adjust_fraction',
        default=None,
        type=float,
        help=' '.join([
            'The percent or fraction of burnin samples to be used to adjust',
            'the step size.',
        ]),
        dest='mcmc.step_adjust_fraction',
    )

    mcmc.add_argument(
        '--num_adaptation_steps',
        default=0,
        type=int,
        help=' '.join([
            'The exact number of steps used to determine the step size.',
        ]),
        dest='mcmc.num_adaptation_steps',
    )

    # TODO consider adding the entire parameter set as args.


def add_bnn_transform_args(parser):
    """Adds the test SJD arguments to the argparser."""
    bnn = parser.add_argument_group(
        'bnn',
        'Arguments pertaining to the Basyesian Neural Network',
    )

    bnn.add_argument(
        '--num_layers',
        default=1,
        type=int,
        help='The number of hidden layers in the BNN.',
        dest='bnn.num_layers',
    )

    bnn.add_argument(
        '--num_hidden',
        default=10,
        type=int,
        help='The number of units per hidden layer in the BNN.',
        dest='bnn.num_hidden',
    )

    bnn.add_argument(
        '--hidden_use_bias',
        action='store_false',
        help='If given, the hidden layers DO NOT use biases (set to zeros)',
        dest='bnn.hidden_use_bias',
    )

    #bnn.add_argument(
    #    '--linear_outputs',
    #    action='store_true',
    #    help='If given, the output layer activations will be linear.',
    #    dest='bnn.linear_outputs',
    #)


def add_custom_args(parser):
    add_mcmc_args(parser)
    add_bnn_transform_args(parser)

    parser.add_argument(
        '--adam_epochs',
        default=0,
        type=int,
        help='The number of epochs for ADAM initialization of the BNN.',
    )

    parser.add_argument(
        '--checkpoint_count',
        default=5,
        type=int,
        help='The number of weights to be saved from each trace.',
    )

    parser.add_argument(
        '--bnn_sample_pics',
        default=3,
        type=int,
        help='The number of samples to visualize per checkpoint.',
    )


if __name__ == '__main__':
    args = io.parse_args(custom_args=add_custom_args)

    # handle kernel args when RWM: change step_size to scale
    if args.mcmc.kernel_id == 'RandomWalkMetropolis':
        args.mcmc.kernel.scale = args.mcmc.kernel.step_size
        del args.mcmc.kernel.step_size
        del args.mcmc.kernel.num_leapfrog_steps

    # handle parallel_iter if cpu only:
    if args.gpu <= 0:
        args.mcmc.sample_chain.parallel_iter = args.cpu_cores

    # handle the linear output function:
    if args.bnn.linear_outputs:
        args.bnn.output_activation = None
        del args.bnn.linear_outputs

    # check num adaptation steps
    if args.mcmc.step_adjust_id is not None:
        if args.mcmc.step_adjust_fraction is None:
            if args.mcmc.num_adaptation_steps is None:
                raise ValueError(
                    'Step adjust requires either the fraction or the exact '
                    + 'num adaptation steps.'
                )

        else:
            args.mcmc.num_adaptation_steps = int(np.ceil(
                args.mcmc.sample_chain.burnin * args.mcmc.step_adjust_fraction
            ))
    del args.mcmc.step_adjust_fraction

    # make mcmc_args contain most args for get_bnn_transform()
    mcmc_args = vars(args.mcmc)
    mcmc_args.update(vars(mcmc_args.pop('sample_chain')))
    mcmc_args['kernel_args'] = vars(mcmc_args.pop('kernel'))

    # Create directory
    output_dir = args.data.dataset_filepath
    output_dir = io.create_dirs(output_dir)

    # Run sample chain
    local_res = bnn_mcmc(
        bnn_args=vars(args.bnn),
        adam_epochs=args.adam_epochs,
        cpu_cores=args.cpu_cores,
        gpus=args.gpu,
        mcmc_args=mcmc_args,
    )
    iter_results = local_res.pop('iter_results')

    results = {}
    results['mcmc_args'] = mcmc_args
    results['is_accepted_total'] = iter_results['trace'].is_accepted.sum()
    results['is_accepted_mean'] = iter_results['trace'].is_accepted.mean()
    results['adam_epochs'] = args.adam_epochs
    results['checkpoint_count'] = args.checkpoint_count

    # save params:
    if results['is_accepted_total'] <= 0:
        io.save_json(
            os.path.join(output_dir, 'bnn_mcmc_results.json'),
            results,
        )
    else:
        results['log_prob_max'] = iter_results['trace'].accepted_results.target_log_prob.max()
        results['log_prob_argmax'] = iter_results['trace'].accepted_results.target_log_prob.argmax()

        results['log_prob_min'] = iter_results['trace'].accepted_results.target_log_prob.min()
        results['log_prob_argmin'] = iter_results['trace'].accepted_results.target_log_prob.argmin()

        # create visualizations
        # target log prob line plot
        plt.plot(iter_results['trace'].accepted_results.target_log_prob)
        plt.savefig(
            os.path.join(output_dir, 'target_log_prob.png'),
            bbox_inches='tight',
            dpi=300,
        )

        # pacf of target log prob:
        if not (iter_results['trace'].accepted_results.target_log_prob == iter_results['trace'].accepted_results.target_log_prob[0]).all():
            pd.DataFrame(acf(
                iter_results['trace'].accepted_results.target_log_prob,
                nlags=int(np.ceil(len(iter_results['trace'].accepted_results.target_log_prob) / 2))
            )).plot(kind='bar')
            plt.savefig(
                os.path.join(output_dir, 'acf_log_prob.png'),
                bbox_inches='tight',
                dpi=300,
            )

            pd.DataFrame(pacf(
                iter_results['trace'].accepted_results.target_log_prob,
                nlags=int(np.ceil(len(iter_results['trace'].accepted_results.target_log_prob) / 10))
            )).plot(kind='bar')
            plt.savefig(
                os.path.join(output_dir, 'pacf_log_prob.png'),
                bbox_inches='tight',
                dpi=300,
            )

        # TODO perhaps save the picture zoomed in w/o the initial spike that
        # tends to occur?

        # Save initial weights
        if args.adam_epochs > 0:
            io.save_json(
                os.path.join(output_dir, 'weights_adam_inital.json'),
                local_res['weights'],
            )

        weights_sets = [w[iter_results['trace'].is_accepted]
            for w in iter_results['samples']]

        if results['is_accepted_total'] <= args.checkpoint_count:
            # save all of the weights, or the entire trace
            io.save_json(
                os.path.join(output_dir, f'trace.json'),
                iter_results['trace'],
            )
        else:
            if args.checkpoint_count <= 2:
                chkpt_idx = [0]
                if results['is_accepted_total'] > 1:
                    chkpt_idx.append(results['is_accepted_total'] - 1)
            else:
                chkpt_idx = [
                    i for i in range(results['is_accepted_total'])
                    if i % np.ceil(results['is_accepted_total']
                    / (args.checkpoint_count - 1)) == 0
                ]

            # Save last accepted weights, first accepted weights already saved
            if results['is_accepted_total'] - 1 != chkpt_idx[-1]:
                chkpt_idx.append(results['is_accepted_total'] - 1)

            for i in chkpt_idx:
                io.save_json(
                    os.path.join(output_dir, f'weights_{i}_idx.json'),
                    [w[i] for w in weights_sets],
                )

        # max target log prob accepted, if not already saved
        io.save_json(
            os.path.join(output_dir, 'weights_max_log_prob.json'),
            [w[results['log_prob_argmax']] for w in iter_results['samples']],
        )
        # min target log prob accepted, if not already saved
        io.save_json(
            os.path.join(output_dir, 'weights_min_log_prob.json'),
            [w[results['log_prob_argmin']] for w in iter_results['samples']],
        )

        # BNN output of a few samples
        if args.bnn_sample_pics > 0:
            bnn_ph, tf_placeholders = bnn_transform.bnn_mlp_placeholders(
                **vars(args.bnn),
            )
            np_density = bnn_transform.assign_weights_bnn(
                weights_sets,
                tf_placeholders,
                bnn_ph,
                local_res['ss0'],
                local_res['tf_input'],
                sess_config=local_res['cpu_config'],
            )

            """
            for i in range(args.bnn_sample_pics):
                #visualization_snippets.overlaid_pairplot(pd.DataFrame(
                visualization_snippets.pair_plot_info(pd.DataFrame(
                    np_density[:, i, :]
                ))
                plt.savefig(
                    os.path.join(output_dir, f'sample_{i}.png'),
                    bbox_inches='tight',
                    dpi=300,
                )
            """

        io.save_json(
            os.path.join(output_dir, 'bnn_mcmc_results.json'),
            results,
        )
