"""General functions used in experiments 1, 2, and 3."""
from functools import partial
import os

import numpy as np
from scipy.stats import entropy

from psych_metric.metrics import measure

from experiment import io
from experiment.research.bnn.bnn_mcmc_fwd import load_bnn_fwd
from experiment.research.bnn import proto_bnn_mcmc

# TODO load bnn in experiment/research/bnn

# Summary func
def summary_arr(arr, quantile_set=None, axis=None):
    summary = {
        'mean': np.mean(arr, axis=axis),
        'max': np.max(arr, axis=axis),
        'min': np.min(arr, axis=axis),
        'median': np.median(arr, axis=axis),
    }

    if quantile_set is not None:
        summary['quantile'] = np.quantile(arr, quantile_set, axis=axis)

    return summary


def summary_dict(measurements, quantile_set=None, axis=1):
    """Returns the summaries of different statistics of the measurements. By default, expects the measurements to be of shape

        [samples, num_measurements_per_sample]
    """
    return {
        'overview': summary_arr(measurements, quantile_set=quantile_set),
        'summary_of_means': summary_arr(
            np.mean(measurements, axis=axis),
            quantile_set,
        ),
        'summary_of_maxs': summary_arr(
            np.max(measurements, axis=axis),
            quantile_set,
        ),
        'summary_of_mins': summary_arr(
            np.min(measurements, axis=axis),
            quantile_set,
        ),
        'summary_of_medians': summary_arr(
            np.median(measurements, axis=axis),
            quantile_set,
        ),
    }


def save_raw_measurements(output_dir, measure_id, measurements, overwrite=False):
    """Saves raw measures as csv or hdf5 file."""
    if len(measurements.shape) <= 2:
        np.savetxt(
            io.create_filepath(
                os.path.join(output_dir, f'{measure_id}.csv'),
                overwrite,
            ),
            measurements,
            delimiter=',',
        )
    else:
        pass
        # TODO save h5py files


def save_measures(
    output_dir,
    measure_id,
    measurements,
    quantiles_frac=None,
    save_raw=True,
    axis=1,
):
    """Convenience function to save measurement output and summarize."""
    if save_raw:
        save_raw_measurements(output_dir, measure_id, measurements)

    if quantiles_frac > 2:
        quantile_set = np.arange(1 + quantiles_frac) / quantiles_frac
    else:
        quantile_set = None

    # Save the summary of the euclidean distances
    io.save_json(
        os.path.join(output_dir, f'{measure_id}_summary.json'),
        summary_dict(measurements, quantile_set, axis=axis),
    )

    # Save the flattening of the conditionals via different summarization methods
    io.save_json(
        os.path.join(output_dir, f'{measure_id}_target_samples_flat.json'),
        summary_arr(measurements, quantile_set, axis=axis),
    )


def get_l2dists(targets, preds, prob_simplex_normalize=False, axis=2, copy=True):
    if copy:
        preds = preds.copy()
    """Obtains euclidean distances from sampling of the predictions."""
    for target_idx in range(len(targets)):
        # NOTE assumes shape of [targets, conditionals, classes]
        preds[target_idx] = targets[target_idx] - preds[target_idx]
    differences = preds

    if prob_simplex_normalize:
        # Normalizes by the largest possible distance within the probability
        # simplex, which is the distance form one vertex to any other vertex
        # because the probability simplex is regular (ie. 2-simplex is a
        # equilateral triangle).
        return np.linalg.norm(differences, axis=axis) / np.sqrt(2)

    return np.linalg.norm(differences, axis=axis)


def add_custom_args(parser):
    proto_bnn_mcmc.add_custom_args(parser)

    # add other args
    parser.add_argument(
        '--target_is_task_target',
        action='store_true',
        help=' '.join([
            'Pass if target is task\'s target labels (Exp 2 where measure is',
            'residuals), rather than where the predictor\'s predictions are',
            'the target (Exp 1).',
        ])
    )

    """
    parser.add_argument(
        '--loaded_bnn_outputs',
        default=None,
        action='store_true',
        help=' '.join([
            'Pass if the bnn_weights file is actually the bnn output, rather',
            'than the expected BNN weight sets. NOT IMPLEMNETED yet.',
        ])
    )
    """

    parser.add_argument(
        '--normalize',
        default=None,
        action='store_true',
        help=' '.join([
            'Pass if the Euclidean distances are to be normalized by the',
            'largest distance between two points possible within the',
            'probability simplex (aka distance between two vertices of the',
            'probability simplex).',
        ])
    )

    parser.add_argument(
        '--use_givens_distrib_samples',
        default=None,
        action='store_true',
        help=' '.join([
            'Pass if to use samples of the distirb that models givens, rather',
            'than just the givens data itself.',
        ])
    )

    parser.add_argument(
        '--quantiles_frac',
        default=5,
        type=int,
        help=' '.join([
            'The number of fractions 1 to serve as quantiles to be used. ie.',
            'quantile of 4 results in 0 (min), 0.25, 0.5 (median), 0.75  and',
            '1 (max). The min, max, and median are always calculated.',
            'Quantiles_frac less than or equal to 2 results in no other',
            'quantiles.',
        ])
    )

    parser.add_argument(
        '--do_not_save_raw',
        action='store_true',
        help='Pass if not to save the raw measurements.'
    )


if __name__ == '__main__':
    # TODO create argparser
    # Create argparser
    args = io.parse_args(
        ['sjd'],
        custom_args=add_custom_args,
        description=' '.join([
            'Runs KL Divergence on ouputs of euclidean BNN given the',
            'sampled weights. Expeirment 2 for KL Divergence completed with',
            'this script.',
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

    # TODO Load the data
    givens, pred, weights_sets, bnn = load_bnn_fwd(
        args.data.dataset_filepath,
        args.bnn_weights_file,
        bnn_mcmc_args,
    )

    # Do exp 2 using using KLDiv, euclid dist
    # in future consider confusion matrix?
    if args.use_givens_distrib_samples:
        # TODO specify if testing only the conditional (use givens as input to
        # BNN) or if using samples from some distrib modeling the givens (ie.
        # Dirichlet)
        # targets = load from file OR sample from given distirb
        raise NotImplementedError
    else:
        targets = givens

        # TODO use the chosen distribution of the givens and sample from it
        # (ie. Dirichlet)
    del pred

    measurements = measure.measure(
        #measure.kldiv_probs,
        partial(entropy, base=2.0, axis=1),
        targets,
        bnn.predict(givens, weights_sets),
    )

    save_measures(
        output_dir,
        'kldivergence',
        measurements,
        args.quantiles_frac,
        save_raw=not args.do_not_save_raw,
    )
