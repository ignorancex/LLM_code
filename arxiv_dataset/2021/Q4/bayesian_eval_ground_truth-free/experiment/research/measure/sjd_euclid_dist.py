"""Uses the Supervised Joint Distribution to analyze the specified human data
and the predictors trained on that data.
"""
import logging
import os

import numpy as np
import h5py

from psych_metric.distrib.supervised_joint_distrib import SupervisedJointDistrib

from experiment import io
import experiment.distrib
from experiment.research.sjd import src_candidates
from experiment.research.measure import kldiv
from experiment.research.bnn.bnn_mcmc_fwd import load_bnn_io_json


def mult_candidates_exp1(
    candidates,
    train,
    test=None,
    sample_size=100,
    normalize=False,
    output_dir=None,
    skip_train_eval=False,
    ):
    """
    Runs multiple instances of experiment 1 with the given set of candidates
    and data

    Parameters
    ----------
    candidates : dict
        Dictionary of str candidate identifiers keys to values of either.
    train : tuple(np.ndarray, np.ndarray)
        Data used for fitting the SJD and testing the typical Bayesian
        situation. Tuple of same shaped ndarrays where the first is the target
        and the second is the predictions of a predictor.
    test : tuple(np.ndarray, np.ndarray)
        Data used for testing the fitted SJD only in a cross validation
        scenerio, different from the typical Bayesian situation of testing
        fitted models. Tuple of same shaped ndarrays where the first is the
        target and the second is the predictions of a predictor. This optional
        and by default is assumed as not given.
    """
    # iterate through the candidate SJDs to obtain their results
    results = {}

    for key, kws in candidates.items():
        if not isinstance(kws, tuple):
            if isinstance(kws, SupervisedJointDistrib):
                candidate = kws
            else:
                # fit appropriate SJDs to train data.
                candidate = SupervisedJointDistrib(
                    target=train[0],
                    pred=train[1],
                    **kws,
                )

            # Save parameters
            results[key] = {'params': candidate.params}
        else:
            # params already saved, just load the conditional samples.
            candidate = kws
            results[key] = {}

        # Fitting done, now for sampling and eval:
        if not skip_train_eval:
            if isinstance(candidate, tuple):
                # Candidate is the str filepaths to the data to be loaded
                with h5py.File(candidate[0], 'r') as f:
                    conditional_samples = f['conditional_samples'][()]
            else:
                if isinstance(output_dir, str):
                    train_path = os.path.join(
                        output_dir,
                        key,
                        'train',
                        'conds.h5',
                    )
                else:
                    train_path = None

                # Get log prob exp results on in-sample data:
                conditional_samples = exp1_givens_data(
                    candidate,
                    train[0],
                    train[1],
                    sample_size,
                    normalize,
                    train_path,
                )

            results[key]['train'] = kldiv.get_l2dists(
                train[1],
                conditional_samples,
                normalize,
            )

        if test is not None:
            # Get out of sample log prob exp results
            if isinstance(candidate, tuple):
                # Candidate is the str filepaths to the data to be loaded
                with h5py.File(candidate[1], 'r') as f:
                    conditional_samples = f['conditional_samples'][()]
            else:
                if isinstance(output_dir, str):
                    test_path = os.path.join(
                        output_dir,
                        key,
                        'test',
                        'conds.h5',
                    )
                else:
                    test_path = None

                conditional_samples = exp1_givens_data(
                    candidate,
                    test[0],
                    test[1],
                    sample_size,
                    normalize,
                    test_path,
                )

            results[key]['test'] =  kldiv.get_l2dists(
                test[1],
                conditional_samples,
                normalize,
            )

    return results


def exp1_givens_data(
    candidate,
    targets,
    preds,
    sample_size=100,
    normalize=False,
    output_path=None,
):
    """Calculates the euclidean distance of the candidate
    SupervisedJointDistrib models' conditional distrib samples to the actual
    prediction value using the actual data of the givens. This is to test ONLY
    the conditional distrib, not the entire SupervisedJointDistrib (excluding
    the distrib of the givens).

    Parameters
    ----------
    candidates : SupervisedJointDistrib | np.ndarray
        instance of a SupervisedJointDistrib whose log prob is to be estimated.
        An array of the samples of the model of the conditional distribution.
    target : np.ndarray
        Used as input to the SJD to generate the conditoinal distribution to be
        evaluated against the actual predicitons via Euclidean distance
        (residuals)
    pred : np.ndarray
    """
    # Sample from given candidate sjd, the conditional distrib only.
    if candidate.independent:
        conditional_samples = candidate.sample(
            len(targets) * sample_size,
        )[1].reshape(len(targets), sample_size, targets.shape[1])
    else:
        conditional_samples = np.swapaxes(
            np.stack(
                [
                    candidate.transform_distrib.sample(targets)
                    for i in range(sample_size)
                ],
            ),
            0,
            1,
        )

    if isinstance(output_path, str):
        output_path = io.create_filepath(output_path)

        with h5py.File(output_path) as hdf5:
            hdf5.create_dataset(
                'conditional_samples',
                data=conditional_samples,
            )

    return conditional_samples


def add_custom_args(parser):
    # just adding a this script specific argument
    kldiv.add_custom_args(parser)

    parser.add_argument(
        '--src_candidates',
        default=None,
        nargs='+',
        type=str,
        help='The premade candidate SJDs to be tested.'
    )

    parser.add_argument(
        '--save_conds',
        action='store_true',
        help='Save the predictions of the SJD results to this filepath.'
    )

    parser.add_argument(
        '--test_datapath',
        default=None,
        type=str,
        help='The filepath to the JSON containing the test input and outputs.'
    )

    parser.add_argument(
        '--sample_size',
        default=100,
        type=int,
        help=' '.join([
            'The number of samples to draw from the conditional distribution',
            'per given sample.'
        ])
    )

    parser.add_argument(
        '--skip_train_eval',
        action='store_true',
        help='Skips evaluating training.',
    )

    parser.add_argument(
        '--load_conds',
        default=None,
        nargs='+',
        type=str,
        help='Loads the conditional samples. Expects only 1 src candidate. ' \
            + 'Expects 1 or 2 filepaths here.',
    )


if __name__ == '__main__':
    args = experiment.io.parse_args(['mle', 'sjd'], add_custom_args)

    output_dir = io.create_dirs(args.output_dir)

    # NOTE be aware that the defaults of SJD args will overwrite src candidates
    del args.sjd.target_distrib
    del args.sjd.transform_distrib
    del args.sjd.independent
    del args.sjd.mle_args

    logging.info('Loading data')
    train = load_bnn_io_json(args.data.dataset_filepath)

    if args.test_datapath is not None:
        test = load_bnn_io_json(args.data.dataset_filepath)
    else:
        test = None

    # Load the candidates as the classes or the filepath to precomputed samples
    if args.load_conds is not None:
        # list of filepaths to hdf5 files of conditional samples
        # Handle when candidate is a valid filepath: ready to load 1 or 2 conds
        # samples
        # if no test, then only load first and put in first tuple idx
        # if test, then load 2 and put in order train, test for tuple.
        if args.skip_train_eval:
            if not os.path.isfile(args.load_conds[0]):
                raise IOError('load_conds is not a valid file')
            if len(args.load_conds) > 1:
                raise ValueError(
                    'load_conds contains more entries than expected.',
                )
            candidate = (None, args.load_conds[0])
        else:
            if len(args.load_conds) > 2:
                raise ValueError(
                    'load_conds contains more entries than expected.',
                )
            if len(args.load_conds) == 2 and args.test_datapath is None:
                raise ValueError(
                    'load_conds contains more entries than expected. ' \
                    + 'No given test_datapath, but given test conditional '\
                    + 'samples.',
                )
            candidate = tuple(args.load_conds)

        # To make formats match as expected (dict)
        candidates = {args.src_candidates[0]: candidate}
    else:
        if args.src_candidates is None:
            args.src_candidates = [
                'iid_uniform_dirs',
                'iid_dirs_mean',
                #'iid_dirs_adam',
                'dir-mean_mvn-umvu-zero',
                #'dir-adam_mvn-umvu',
            ]

        logging.info('Getting candidates')
        # Get candidates
        candidates = src_candidates.get_sjd_candidates(
            args.src_candidates,
            train[0].shape[1],
            vars(args.mle),
            vars(args.sjd),
            n_jobs=args.cpu_cores,
        )

    logging.info('Candidates obtained. Beginning the experiment loop.')

    # Loop through candidates, fit givens and conds, sample conds given
    # data givens,
    if test is not None:
        results = mult_candidates_exp1(
            candidates,
            train,
            test,
            sample_size=args.sample_size,
            normalize=args.normalize,
            output_dir=output_dir if args.save_conds else None,
         )
    else:
        results = mult_candidates_exp1(
            candidates,
            train,
            sample_size=args.sample_size,
            normalize=args.normalize,
            output_dir=output_dir if args.save_conds else None,
         )

    logging.info('Looped through all candidates.')
    logging.info('Saving results.')

    # Save the results of multiple candidates.
    for key, result in results.items():
        tmp_out_dir = io.create_dirs(os.path.join(output_dir, key, 'train'))

        if 'params' in result:
            io.save_json(
                os.path.join(tmp_out_dir, 'params.json'),
                result['params'],
            )

        kldiv.save_measures(
            tmp_out_dir,
            'euclid_dists_train',
            result['train'],
            args.quantiles_frac,
            not args.do_not_save_raw,
        )

        if test is not None:
            tmp_out_dir = io.create_dirs(os.path.join(output_dir, key, 'test'))

            kldiv.save_measures(
                tmp_out_dir,
                'euclid_dists_test',
                result['test'],
                args.quantiles_frac,
                not args.do_not_save_raw,
            )
