import numpy as np
import argparse
import sys


def init_args():
    parser = argparse.ArgumentParser(
        prog='From point-cloud to distance matrix',
        description='''Loads a point cloud from space-separated-value file/stdin, one point per row.
Outputs space-separated-value lower-triangular distance matrix, i.e.,
list of the distance matrix entries below the diagonal, sorted lexicographically by row index, then column index.''')
    parser.add_argument('--input', '-i', type=str, required=False, default=None,
                        help='Path to the point-cloud file. By default standard input used.')
    parser.add_argument('--input_type', type=str, required=False, default='txt', choices=['txt', 'np'],
                        help='Type of the file. Either text file (txt) or numpy file (np) (default txt)')
    parser.add_argument('--output', '-o', type=str, required=False, default=None,
                        help='Path to the output. By default standard output used.')
    parser.add_argument('--metric', type=str, required=True, choices=['euclidean', 'cosine'])

    return parser.parse_args()


def load_vectors_text(file_name):
    return np.loadtxt(file_name if file_name else sys.stdin, delimiter=' ')


def load_vectors_numpy(file_name):
    return np.load(file_name)


def euclidean_distance(a, b):
    return np.linalg.norm(b - a)


def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_and_save_matrix(write_method, pts, metric_name='euclidean',
                            column_separator=' ', row_separator='\n', decimal_places=10):
    distance = {
        'euclidean': euclidean_distance,
        'cosine': cosine_distance
    }[metric_name]
    for i in range(1, len(pts)):
        for j in range(i):
            write_method(f'{distance(pts[i], pts[j]):.{decimal_places}f}')
            write_method(column_separator)
        write_method(row_separator)


def main():
    args = init_args()
    if args.input_type == 'txt':
        pts = load_vectors_text(args.input)
    elif args.input_type == 'np':
        pts = load_vectors_numpy(args.input)
    if args.output:
        with open(args.output, 'w') as file:
            compute_and_save_matrix(lambda text: file.write(text), pts, metric_name=args.metric)
    else:
        compute_and_save_matrix(lambda text: print(text, end=''), pts, metric_name=args.metric)


if __name__ == '__main__':
    main()
