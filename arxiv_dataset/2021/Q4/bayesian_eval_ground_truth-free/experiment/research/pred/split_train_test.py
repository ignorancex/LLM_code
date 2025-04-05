import argparse
import json
import os

import numpy as np
from sklearn.model_selection import KFold

from experiment import io

def split(
    in_file,
    out_file=None,
    keys=['conditionals', 'givens'],
    shuffle=True,
    kfolds=3,
    random_seed=1234,
):
    """Converts format from `load_fold_save_pred` to given and conditional of
    a single data split (train XOR test)
    """
    if out_file is None:
        out_file = in_file

    with open(in_file, 'r') as f:
        in_content = json.load(f)

    for key in keys:
        in_content[key] = np.array(in_content[key])

    for fold_num, (train_idx, test_idx) in enumerate(
        KFold(kfolds, shuffle=shuffle, random_state=random_seed).split(
            in_content[keys[0]]
        )
    ):
        out_1 = in_content.copy()
        out_2 = in_content.copy()

        for key in keys:
            out_1[key] = out_1[key][train_idx]
            out_2[key] = out_2[key][test_idx]

        io.save_json(f'{out_file}_fold_{fold_num}_train.json', out_1)
        io.save_json(f'{out_file}_fold_{fold_num}_test.json', out_2)


def parse_args():
    parser = argparse.ArgumentParser(description='Splits given json into two.')

    parser.add_argument(
        'in_file',
        help='The JSON file in format of `save_pred.load_fold_save_pred()`.',
    )

    parser.add_argument(
        '-o',
        '--out_file',
        default=None,
        help='The output file path of converted json.',
    )

    parser.add_argument(
        '--keys',
        default=['conditionals', 'givens'],
        nargs='+',
        type=str,
        help='Keys in json to split in half.',
    )

    parser.add_argument(
        '--no_shuffle',
        action='store_false',
        help='Do not shuffle the data prior to splitting.',
    )

    parser.add_argument(
        '--kfolds',
        default=3,
        type=int,
        help='The number of folds to split the data into. Default is 3.',
    )

    # NOTE this random seed was not used in half split: all half 1 or 2 sets.
    parser.add_argument(
        '--random_seed',
        default=1234,
        type=int,
        help='The random seed used to shuffle the data if it is shuffled.',
    )

    args = parser.parse_args()

    # variable name change
    args.shuffle = args.no_shuffle
    del args.no_shuffle

    return  args


if __name__ == '__main__':
    split(**vars(parse_args()))
