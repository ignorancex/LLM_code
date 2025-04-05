import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd
import seaborn as sns

from experiment import io

from psych_metric.metrics import measure


def parse_args():
    parser = argparse.ArgumentParser(description='')

    # IO args
    parser.add_argument(
        'train_paths',
        default=None,
        nargs='+',
        type=str,
        help='CSV containing measurements on train.',
    )

    parser.add_argument(
        '--test_paths',
        default=None,
        nargs='+',
        type=str,
        help='CSV containing measurements on test.',
    )

    parser.add_argument(
        '-o',
        '--output_path',
        default=None,
        help='The output file path of converted json.',
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='If given, the script will overwrite pre-existing files.',
    )


    # Data  speccific args
    parser.add_argument(
        '--conditional_models',
        default='conditional_model',
        nargs='+',
        type=str,
        help='Names of the different conditional models.',
    )

    # Credible interval speccific args
    parser.add_argument(
        '--cred_interval',
        default=None,
        choices=['left', 'right', 'highest_density'],
        help='The type of credibility interval to calculate.',
    )

    parser.add_argument(
        '--credibility',
        default=0.95,
        type=float,
        help='The credibility percent to calculate.',
    )

    args = parser.parse_args()

    return args


def get_cred(train_csv, test_csv, cred_interval, credibility=.95):
    train = pd.read_csv(train_csv, header=None).values.flatten()
    test = pd.read_csv(test_csv, header=None).values.flatten()

    if cred_interval == 'highest_density':
        train_quantiles = measure.highest_density_credible_interval(
            train,
            credibility,
        )

        test_quantiles = measure.highest_density_credible_interval(
            test,
            credibility,
        )

        return {
            'train': {
                'lower_quantile': train_quantiles[0],
                'higher_quantile': train_quantiles[1],
            },
            'test': {
                'lower_quantile': test_quantiles[0],
                'higher_quantile': test_quantiles[1],
            },
        }

    if cred_interval == 'left' or cred_interval == 'right':
        train_interval = measure.one_tailed_credible_interval(
            train,
            credibility,
            cred_interval == 'left',
        )

        test_interval = measure.one_tailed_credible_interval(
            test,
            credibility,
            cred_interval == 'left',
        )

        return {'train': train_interval, 'test': test_interval}

    raise ValueError(f'Unexpected cred_interval value = {cred_interval}')


if __name__ == '__main__':
    args = parse_args()

    # Get intervals save intervals to model by json
    intervals = {}
    intervals['credibility'] = args.credibility
    intervals['credibility_interval'] = args.cred_interval

    if isinstance(args.train_paths, str) and isinstance(args.test_paths, str):
        intervals[args.conditional_models] = get_cred(
            args.train_paths,
            args.test_paths,
            args.cred_interval,
            args.credibility,
        )
    elif (
        isinstance(args.train_paths, list)
        and isinstance(args.test_paths, list)
        and len(args.train_paths) == len(args.test_paths)
        and args.conditional_models is not None
        and len(args.train_paths) == len(args.conditional_models)
    ):
        for i in range(len(args.train_paths)):
            intervals[args.conditional_models[i]] = get_cred(
                args.train_paths[i],
                args.test_paths[i],
                args.cred_interval,
                args.credibility,
            )
    else:
        raise TypeError('Expected either pair of strs, or pair of lists.')

    io.save_json(os.path.join(args.output_path), intervals)
