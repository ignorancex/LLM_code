import argparse
import json

import numpy as np

from experiment import io

def combine(
    files,
    out_file,
    shuffle=False,
    keys=['conditionals', 'givens'],
    seed=1234,
):
    """Converts format from `load_fold_save_pred` to given and conditional of
    a single data split (train XOR test)
    """
    key_contents = {key:[] for key in keys}
    for in_file in files:
        with open(in_file, 'r') as f:
            file_contents = json.load(f)

            for key in keys:
                key_contents[key].append(file_contents[key])

    # Stack each key's contents
    for key in keys:
        key_contents[key] = np.vstack(key_contents[key])

        if shuffle:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError(
                    f'Expected `seed` to be of type `int` not `{type(seed)}`.'
                )
            np.random.shuffle(key_contents[key])

        # Delete key from file contents; keeping extra details from last JSON
        del file_contents[key]

    # Add the other details, if any into the new JSON file.
    # NOTE, expects that the last file to be opened contains all of the
    # extra content, thus the same content, as the prior files.
    if file_contents:
        key_contents.update(file_contents)

    io.save_json(out_file, key_contents)


def parse_args():
    parser = argparse.ArgumentParser(description='Combines given JSONs')

    parser.add_argument(
        'out_file',
        help='The output file path of converted json.',
    )

    parser.add_argument(
        'files',
        nargs='+',
        help='A list of JSON files in format of `save_pred.load_fold_save_pred()`.',
    )

    parser.add_argument(
        '--keys',
        default=['conditionals', 'givens'],
        nargs='+',
        type=str,
        help='Keys in json to split in half.',
    )

    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffles the data after combining.',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='Seed used to shuffle the contents together the same way.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    combine(**vars(parse_args()))
