"""Performs the Experiment 2 of paper using ROC AUC and MCC"""
from functools import partial

import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef

from psych_metric.metrics import measure

from experiment import io
from experiment.research.bnn.bnn_mcmc_fwd import load_bnn_fwd
from experiment.research.measure import kldiv

def add_custom_args(parser):
    kldiv.add_custom_args(parser)

    parser.add_argument(
        '--multi_class',
        default='ovo',
        choices=['ovo', 'ovr'],
        help='Multiclass arg of sklearn roc_auc_score. One vs One or vs Rest.',
    )

    parser.add_argument(
        '--measure',
        default='both',
        choices=['both', 'mcc', 'roc_auc'],
        help='Which measure(s) to compute.'
    )


if __name__ == "__main__":
    # Create argparser
    args = io.parse_args(
        ['sjd'],
        custom_args=add_custom_args,
        description=' '.join([
            'Runs ROC AUC and MCC on ouputs of euclidean BNN given the',
            'sampled weights.'
        ]),
    )

    output_dir = io.create_dirs(args.output_dir)

    # Manage bnn mcmc args from argparse
    bnn_mcmc_args = vars(args.bnn)
    bnn_mcmc_args['sess_config'] = io.get_tf_config(
        args.cpu_cores,
        args.cpu,
        args.gpu,
    )

    givens, pred, weights_sets, bnn = load_bnn_fwd(
        args.data.dataset_filepath,
        args.bnn_weights_file,
        bnn_mcmc_args,
    )
    del pred

    givens_argmax = givens.argmax(1).reshape(-1, 1)
    bnn_pred = bnn.predict(givens, weights_sets)
    bnn_pred_argmax = bnn_pred.argmax(2)[..., np.newaxis]

    print('givens shape = ', givens_argmax.shape)
    print('bnn_pred shape = ', bnn_pred.shape)
    print('bnn_pred_argmax shape = ', bnn_pred_argmax.shape)

    print('args.measure = ', args.measure)

    if args.measure == 'both' or args.measure == 'mcc':
        measurements = measure.measure(
            matthews_corrcoef,
            givens_argmax,
            bnn_pred_argmax,
        ).reshape(-1, 1)

        print('measure = mcc')
        print(measurements.shape)
        print(measurements)

        kldiv.save_measures(
            output_dir,
            'matthews_corrcoef',
            measurements,
            args.quantiles_frac,
            save_raw=not args.do_not_save_raw,
        )

    if args.measure == 'both' or args.measure == 'roc_auc':
        measurements = measure.measure(
            partial(
                roc_auc_score,
                multi_class=args.multi_class,
                labels=np.arange(args.bnn.num_hidden + 1),
            ),
            givens_argmax.squeeze(),
            bnn_pred,
        ).reshape(-1, 1)

        print('roc_auc')
        print(measurements.shape)
        print(measurements)

        kldiv.save_measures(
            output_dir,
            'roc_auc',
            measurements,
            args.quantiles_frac,
            save_raw=not args.do_not_save_raw,
        )
