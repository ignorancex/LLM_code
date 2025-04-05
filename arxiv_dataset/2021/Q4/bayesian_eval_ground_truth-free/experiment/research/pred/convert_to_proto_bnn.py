import argparse
import json
import os

import numpy as np

from psych_metric.distrib.simplex.euclidean import EuclideanSimplexTransform

from experiment import io

def convert_pred_file_to_proto_bnn(
    in_file_path,
    out_file_path=None,
    part='train',
    old_euclid_transform=True,
 ):
    """Converts format from `load_fold_save_pred` to given and conditional of
    a single data split (train XOR test)
    """
    if part != 'train' and part != 'test':
        raise ValueError('part must be the strings "train" or "test".')

    if out_file_path is None:
        if os.path.sep in in_file_path:
            out_file_path = in_file_path.rpartition(os.path.sep)[0]
        else:
            out_file_path = './'

        out_file_path = os.path.join(
            out_file_path,
            f'proto_bnn_format_{part}.json',
        )

    with open(in_file_path, 'r') as f:
        in_file = json.load(f)

    out_file = {
        'givens': in_file['labels'][part],
        'conditionals': in_file[part],
    }

    if old_euclid_transform:
        est = EuclideanSimplexTransform(np.array(out_file['givens']).shape[1])
        out_file['change_of_basis'] = est.change_of_basis_matrix
        out_file['origin_adjust'] = est.origin_adjust

    io.save_json(out_file_path, out_file)


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        'in_file_path',
        help='The JSON file in format of `save_pred.load_fold_save_pred()`.',
    )

    parser.add_argument(
        '-o',
        '--out_file_path',
        default=None,
        help='The output file path of converted json.',
    )

    parser.add_argument(
        '-p',
        '--part',
        default='train',
        help='The data part "train" xor "test".',
    )

    parser.add_argument(
        '--old_euclid_transform',
        action='store_true',
        help=' '.join([
            'If given, the change of basis and origin adjust are added to the',
            'output file',
        ]),
    )

    return parser.parse_args()


if __name__ == '__main__':
    convert_pred_file_to_proto_bnn(**vars(parse_args()))
