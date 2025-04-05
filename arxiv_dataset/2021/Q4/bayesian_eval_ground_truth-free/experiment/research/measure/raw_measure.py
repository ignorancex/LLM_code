"""Runs the measures on the raw data (no modeling of distributions).

Notes
-----
Experiement 2 for residuals when target is given target label to predictor:
    Performing this where the target is the actual target label of the task and
    pred is the estimated predictions of the predictor via the BNN MCMC
    generates the distribution of residuals, which is a distribution of a
    measure and part of experiment 2.
"""
import os

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, matthews_corrcoef

from psych_metric.metrics import measure

from experiment import io
from experiment.research.bnn.bnn_mcmc_fwd import load_bnn_io_json
from experiment.research.measure import kldiv

def add_custom_args(parser):
    kldiv.add_custom_args(parser)

    # add other args
    parser.add_argument(
        '--measure',
        default='euclid_dist',
        choices=['euclid_dist', 'kldiv', 'roc_auc', 'all'],
        help=' '.join([
            'Pass if target is task\'s target labels (Exp 2 where measure is',
            'residuals), rather than where the predictor\'s predictions are',
            'the target (Exp 1).',
        ])
    )

    parser.add_argument(
        '--multi_class',
        default='ovo',
        choices=['ovr', 'ovo'],
        help='multi_class for sklearn.metrics.roc_auc_score()',
    )


if __name__ == "__main__":
    # Create argparser
    args = io.parse_args(
        ['sjd'],
        custom_args=add_custom_args,
        description=' '.join([
            'Runs measure on raw data, as one does normally without the',
            'proposed framework.',
        ]),
    )

    #output_dir = io.create_dirs(args.output_dir)
    output_dir = args.output_dir

    givens, conds = load_bnn_io_json(args.data.dataset_filepath)

    # Perform the measurement
    if args.measure == 'all' or args.measure == 'euclid_dist':
        measurements = kldiv.get_l2dists(givens, conds, args.normalize, axis=1)

        kldiv.save_measures(
            output_dir,
            'euclid_dist',
            measurements,
            args.quantiles_frac,
            save_raw=not args.do_not_save_raw,
            axis=0,
        )

    if args.measure == 'all' or args.measure == 'kldiv':
        #measurements = measure.measure(measure.kldiv_probs, givens, conds)
        measurements = entropy(givens, conds, axis=1)

        kldiv.save_measures(
            output_dir,
            'kldiv',
            measurements,
            args.quantiles_frac,
            save_raw=not args.do_not_save_raw,
            axis=0,
        )

    # TODO Be aware that the BNN measures per col need to be done on axis=0
    if args.measure == 'all' or args.measure == 'roc_auc':
        givens_argmax = givens.argmax(axis=1)
        conds_argmax = conds.argmax(axis=1)

        if (
            'cha' in args.data.dataset_filepath.lower()
            and 'test' in args.data.dataset_filepath.lower()
        ):
            # labels, whenever givens argmax is missing a class it
            # fails, happens in test and all_half_1 in CHA Learn.
            # NOTE hardcoded hotfix
            labels = np.arange(20)
        else:
            labels = None

        # Save the ROC AUC
        io.save_json(
            io.create_filepath(os.path.join(output_dir, 'roc_auc.json')),
            {
                'roc_auc':  roc_auc_score(
                    givens_argmax,
                    conds,
                    multi_class=args.multi_class,
                    labels=labels,
                ),
                'matthews_corrcoef': matthews_corrcoef(
                    givens_argmax,
                    conds_argmax,
                ),
            },
        )

    # TODO multiclass ROC in sklearn does not seem to be working. May need to make this myself.
