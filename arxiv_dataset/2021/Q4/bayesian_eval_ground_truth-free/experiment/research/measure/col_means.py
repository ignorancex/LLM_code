"""Given csv, obtains column means."""
import argparse

import numpy as np

from experiment import io


def parse_args():
    parser = argparse.ArgumentParser(description='Column means')

    # add other args
    parser.add_argument(
        'input_file',
        help='Input file to the csv of the matrix',
    )

    parser.add_argument(
        'output_file',
        help='Output file of csv with resulting means',
    )

    parser.add_argument(
        '--axis',
        default=0,
        choices=[0, 1],
        type=int,
        help='axis to apply the mean. Default axis = 0',
    )

    parser.add_argument(
        '--keep_infinite',
        action='store_true',
        help='If given, inf and nans are kept and used in the mean',
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    matrix = np.loadtxt(args.input_file, delimiter=',')

    # remove nans and infs
    if not args.keep_infinite:
        matrix = np.ma.masked_array(matrix, mask=~np.isfinite(matrix))

    np.savetxt(
        io.create_filepath(args.output_file),
        matrix.mean(axis=args.axis)
    )
