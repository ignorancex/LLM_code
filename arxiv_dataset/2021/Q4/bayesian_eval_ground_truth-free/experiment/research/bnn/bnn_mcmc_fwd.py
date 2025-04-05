"""BNN MCMC Forward pass helper functions."""
import json
import logging
import os
from pathlib import Path

import numpy as np

from psych_metric.distrib.bnn.bnn_mcmc import BNNMCMC


def load_sample_weights(weights_dir, filename='.json', dtype=np.float):
    """Loads the sampled bnn weight sets from directory recursively into
    memory.
    """
    weights_sets = []

    # loop through json files, get weights, concatenate them
    # walk directory tree, grabbing all
    #for filepath in glob.iglob(os.path.join(weights_dir, '*', filename)):
    if os.path.isdir(weights_dir):
        for filepath in Path(weights_dir).rglob('*' + filename):
            with open(filepath, 'r') as f:
                weights = json.load(f)
                for vals in weights.values():
                    if weights_sets:
                        for i, w in enumerate(vals['weights']):
                            weights_sets[i] = np.vstack((
                                weights_sets[i],
                                np.array(w, dtype=dtype)[vals['is_accepted']],
                            ))
                    else:
                        weights_sets += [
                            np.array(w, dtype=dtype)[vals['is_accepted']]
                            for w in vals['weights']
                        ]
        return weights_sets

    # Handles the case where it is given a single json of accepted weights.
    elif os.path.isfile(weights_dir):
        with open(weights_dir, 'r') as f:
            contents = json.load(f)

            if isinstance(contents, list):
                weights_sets = [np.array(x, dtype=dtype) for x in contents]
            elif isinstance(contents, dict):
                if 'is_accepted' in  contents and 'weights' in contents:
                    weights_sets = [np.array(x, dtype=dtype)['is_accepted']
                        for x in contents['weights']
                    ]
                else:
                    for vals in weights.values():
                        if weights_sets:
                            for i, w in enumerate(vals['weights']):
                                weights_sets[i] = np.vstack((
                                    weights_sets[i],
                                    np.array(w, dtype=dtype)[vals['is_accepted']],
                                ))
                        else:
                            weights_sets += [
                                np.array(w, dtype=dtype)[vals['is_accepted']]
                                for w in vals['weights']
                            ]

        return weights_sets

    raise ValueError(' '.join([
        '`weights_dir` is an invalid filepath. Expected either a directory of',
        'JSONs to be traversed recursively and whose rejected weights sets',
        'are to be pruned, xor a single JSON file of aggregated accepted',
        'weights sets.',
    ]))


def load_bnn_io_json(dataset_filepath, load_simplex=False, dtype=np.float32):
    with open(dataset_filepath, 'r') as f:
        data = json.load(f)
        givens = np.array(data['givens'], dtype=dtype)
        conditionals = np.array(data['conditionals'], dtype=dtype)
        if load_simplex:
            change_of_basis = np.array(data['change_of_basis'], dtype=dtype)
            origin_adjust = np.array(data['origin_adjust'], dtype=dtype)
            return givens, conditionals, change_of_basis, origin_adjust
    return givens, conditionals


def load_bnn_fwd(
    dataset_filepath,
    bnn_weights_file=None,
    bnn_mcmc_args=None,
    dtype=np.float32,
    #load_simplex=False,
):
    """Convenience script function for loading everything to perform BNNMCMC
    fwd pass. Loading the data for training the BNNMCMC, loads the BNNMCMC,
    loads the BNNMCMC's weights.

    Parameters
    ----------
    dataset_filepath : str
        The filepath to the JSON file that contains the array of given labels
        (input to the BNNMCMC), the array of conditionals (expected output of
        BNNMCMC) and the simplex transformation attributes: change of basis
        matrix and origin adjust.
    bnn_weights_file : str, optional
        Filepath to the JSON file or directory of JSON files that contain the
        BNN weights sets. In case of a single JSON file, it should just contain
        a list of lists, the actual aggregated weights sets. If a directory,
        expected to contain a dict of multiple BNNMCMC chain run results that
        contain both the weights sets and the boolean of whether the specific
        weights sets were accepted or rejected by the MCMC algorithm. The
        accepted weights are extracted from these files recursively.

    Returns
    -------
    (np.ndarray, np.ndarray, list(np.ndarray), BNNMCMC)
        Returns the array of given labels (input to BNNMCMC), the array of
        predictor output labels (expected label output of BNNMCMC), the
        different weights sets of the BNNMCMC, and the BNNMCMC instance.
    """
    givens, conds = load_bnn_io_json(dataset_filepath)

    #sample_log_prob = partial(
    #    bnn_transform.mcmc_sample_log_prob,
    #    origin_adjust=origin_adjust,
    #    rotation_mat=change_of_basis,
    #    scale_identity_multiplier=bnn_args['scale_identity_multiplier'],
    #)

    if bnn_weights_file is None:
        logging.warning(' '.join([
            '`bnn_weights_file` was not given. Generating random initial',
            'weights set of BNN MCMC',
        ]))

        weights_sets = [
            np.random.normal(
                scale=12**0.5,
                size=(givens.shape[1] - 1, bnn_mcmc_args['num_hidden']),
            ).astype(dtype),
            np.zeros([bnn_mcmc_args['num_hidden']], dtype=dtype),
            np.random.normal(
                scale=0.48**0.5,
                size=(bnn_mcmc_args['num_hidden'], givens.shape[1] - 1),
            ).astype(dtype),
            np.zeros([givens.shape[1] - 1], dtype=dtype),
        ]
    else:
        weights_sets = load_sample_weights(bnn_weights_file)

    # Create instance of BNNMCMC
    if 'dim' in bnn_mcmc_args and bnn_mcmc_args['dim'] != givens.shape[1]:
        raise ValueError(' '.join([
            '`dim` was given as a BNNMCMC arg and equal to',
            f'{bnn_mcmc_args["dim"]}, but the input data loaded has a',
            'different sized dimension of givens.shape[1] =',
            f'{givens.shape[1]}',
        ]))

    if 'dim' not in bnn_mcmc_args:
        bnn_mcmc_args['dim'] = givens.shape[1]

    return givens, conds, weights_sets, BNNMCMC(**bnn_mcmc_args)
