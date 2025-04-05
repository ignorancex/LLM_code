"""Convert first impressions predictions format into proto bnn's expected
format.
"""
import argparse
import csv
import os

import numpy as np

from psych_metric.distrib.simplex.euclidean import EuclideanSimplexTransform

from experiment import io

def convert_pred_file_to_proto_bnn(
    in_file_path,
    out_file_path=None,
    part='train',
    old_euclid_transform=True,
    givens_col_idx=5,
    conditionals_col_idx=4,
    header=True,
    delimiter=',',
    quotechar='"',
    dtype=np.float64,
    normalize_givens=True,
 ):
    """Converts format from first impresisons format to given and conditional
    of a single data split (train XOR test) """
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

    out_file = {}
    with open(in_file_path, 'r') as csv_f:
        reader = csv.reader(csv_f, delimiter=delimiter, quotechar=quotechar)
        if header:
            next(reader)

        first_data_row = next(reader)

        out_file['givens'] = np.fromstring(
            first_data_row[givens_col_idx][1:-1],
            dtype=dtype,
            sep=',',
        )
        if normalize_givens:
            out_file['givens'] /= out_file['givens'].sum()

        out_file['conditionals'] = np.fromstring(
            first_data_row[conditionals_col_idx][1:-1],
            dtype=dtype,
            sep=',',
        )

        for row in reader:
            # Need to normalize the historgram of target labels (givens)
            out_file['givens'] = np.vstack((
                out_file['givens'],
                np.fromstring(row[givens_col_idx][1:-1], dtype=dtype, sep=','),
            ))
            if normalize_givens:
                out_file['givens'][-1] /= out_file['givens'][-1].sum()

            out_file['conditionals'] = np.vstack((
                out_file['conditionals'],
                np.fromstring(
                    row[conditionals_col_idx][1:-1],
                    dtype=dtype,
                    sep=',',
                ),
            ))

    if old_euclid_transform:
        est = EuclideanSimplexTransform(out_file['givens'].shape[1])
        out_file['change_of_basis'] = est.change_of_basis_matrix
        out_file['origin_adjust'] = est.origin_adjust

    io.save_json(out_file_path, out_file)


def parse_args():
    parser = argparse.ArgumentParser(description='')

    # specify part first
    parser.add_argument(
        'part',
        default=None,
        choices=['train', 'test'],
        help='The data part "train" xor "test".',
    )

    parser.add_argument(
        'in_file_path',
        help='The file in format of `save_pred.load_fold_save_pred()`.',
    )

    parser.add_argument(
        '-o',
        '--out_file_path',
        default=None,
        help='The output file path of converted json.',
    )

    parser.add_argument(
        '--old_euclid_transform',
        action='store_true',
        help=' '.join([
            'If given, the change of basis and origin adjust are added to the',
            'output file',
        ]),
    )

    parser.add_argument(
        '--no_header',
        action='store_false',
        help='Use if no header exists in src csv file.',
    )

    parser.add_argument(
        '--no_normalize_givens',
        action='store_false',
        help='Use if the givens from src csv do NOT need normalized.',
    )

    parser.add_argument(
        '--delimiter',
        default=',',
        help='Use if no header exists in src csv file.',
    )

    parser.add_argument(
        '--quotechar',
        default='"',
        help='Use if no header exists in src csv file.',
    )

    parser.add_argument(
        '-g',
        '--givens_col_idx',
        default=5,
        type=int,
        help='Column index of givens data in src csv.',
    )

    parser.add_argument(
        '-c',
        '--conditionals_col_idx',
        default=4,
        type=int,
        help='Column index of conditionals data in src csv.',
    )

    args = parser.parse_args()

    args.header = args.no_header
    del args.no_header

    args.normalize_givens = args.no_normalize_givens
    del args.no_normalize_givens

    return args


if __name__ == '__main__':
    convert_pred_file_to_proto_bnn(**vars(parse_args()))
