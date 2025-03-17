#
#   For the folders used, search for comments PATH TO DATA and PATH RO OUTPUT
#

import numpy as np
import persim
import time
import gudhi
import argparse


def init_args():
    parser = argparse.ArgumentParser(
        prog='Compute persistence diagram distances')
    parser.add_argument('--name', type=str, required=True,
                        help='Experiment name.')
    parser.add_argument('--number', '-n', type=int, required=True,
                        help='Number of words used.')
    parser.add_argument('--dimension', '-d', type=int, required=True,
                        help='The maximal dimension used.')
    parser.add_argument('--distances', type=str, nargs='+', required=False,
                        default=('bottleneck', 'sliced_wasserstein', 'persistence_image', 'bars_statistics'),
                        help='List of the distances between persistence diagrams to use.')
    parser.add_argument('--metrics', type=str, nargs='+', required=False,
                        default=('euclidean', 'cosine'),
                        help='List of the point-cloud metrics used.')
    parser.add_argument('--languages', '-l', type=str, nargs='+', required=True,
                        help='List of the languages.')
    parser.add_argument('--task_id', type=int, required=False, default=-1,
                        help='If given, only perform one job, with the given id.')

    return parser.parse_args()


def load_bars(filename):
    bars = []
    bars_dim = []
    with open(filename, 'r') as file:
        for row in file:
            row = row.strip()
            if len(row) == 0:
                bars.append(np.array(bars_dim))
            elif len(row) == 1:
                bars_dim = []
            else:
                bars_dim.append(tuple(map(float, row.split())))
        bars.append(np.array(bars_dim))

    return bars


def save_matrix(filename, mat, labels, line_sep='\n', decimal_places=6):
    with open(filename, 'w') as file:
        file.write(' '.join(labels) + line_sep)
        for row in mat:
            file.write(' '.join([f'{x:.{decimal_places}f}' for x in row]) + line_sep)


def vectorise_persistence_image(pd, birth_range, pers_range, pixel_size, sigma):
    pi_transformer = persim.PersistenceImager(
        birth_range=birth_range,
        pers_range=pers_range,
        pixel_size=pixel_size,
        weight='persistence',
        weight_params={},
        kernel_params={'sigma': [[sigma, 0], [0, sigma]]}
    )

    return pi_transformer.transform([pd])[0]


def entropy(values):
    total = sum(values)
    if total <= 0:
        return 0
    entropy_not_normalized = sum(x * np.log(x) for x in values if x > 0)
    return np.log(sum(values)) - (entropy_not_normalized / sum(values))


def vectorise_bars_statistics(pd, only_death=False):
    vector = []
    if only_death:
        quantities = {'deaths': lambda birth, death: death}
    else:
        quantities = {'deaths': lambda birth, death: death,
                      'births': lambda birth, death: birth,
                      'lifespans': lambda birth, death: max(0, death - birth),
                      'midpoints': lambda birth, death: (birth + death) / 2}
    for quantity_label, quantity in sorted(quantities.items()):
        values = np.fromiter((quantity(birth, death) for birth, death in pd), dtype='float64')
        if len(values) == 0:
            values = np.zeros(1, dtype='float64')
        statistics = {
            'mean': np.mean(values),
            'standard deviation': np.std(values),
            'median': np.median(values),
            'interquartile range': np.subtract(*np.percentile(values, [75, 25])),
            'full range': np.ptp(values),
            '10th percentile': np.percentile(values, 10),
            '25th percentile': np.percentile(values, 25),
            '75th percentile': np.percentile(values, 75),
            '90th percentile': np.percentile(values, 90),
            'entropy': entropy(values)
        }
        for statistic_label in ['mean', 'standard deviation', 'median', 'interquartile range', 'full range',
                                '10th percentile', '25th percentile', '75th percentile', '90th percentile', 'entropy']:
            vector.append(statistics[statistic_label])
    return np.array(vector)


def compare_pds(pd1, pd2, distance):
    if distance == 'bottleneck':
        return gudhi.bottleneck_distance(pd1, pd2)  # third argument leads to approximating
    elif distance == 'sliced_wasserstein':
        return persim.sliced_wasserstein(pd1, pd2, M=50)


def main():
    args = init_args()
    experiment_name = args.name
    out_folder = f'outputs/pd_distances_{experiment_name}'
    langs = args.languages
    n = args.number
    maxdim = args.dimension
    metrics = ('euclidean', 'cosine')
    distances = args.distances

    print(f"Experiment {experiment_name}: n={n}, #langs={len(langs)}, maxdim={maxdim}")
    print("==================================================")

    parameter_list = [(metric, dim, distance) for metric in metrics
                      for dim in range(maxdim + 1)
                      for distance in distances]
    if args.task_id >= 0:
        parameter_indices = [args.task_id]
    else:
        parameter_indices = range(len(parameter_list))

    for parameter_index in parameter_indices:
        metric, dim, distance = parameter_list[parameter_index]

        pds = {lang: load_bars(f'data/bars/bars.{lang}.300.n{n}.{metric}.d{maxdim}.txt') for lang in langs}  # PATH TO DATA
        print(f'Computing matrix for {metric} metric, with {distance} distance, dimension {dim}')
        ts = time.perf_counter()

        if distance == 'persistence_image':
            if metric == 'euclidean' and dim == 0:
                pds_vec = {
                    lang: vectorise_persistence_image(pd[dim], birth_range=(0, 1), pers_range=(0, 10), pixel_size=1,
                                                      sigma=1) for lang, pd in pds.items()}
            elif metric == 'euclidean' and dim > 0:
                pds_vec = {
                    lang: vectorise_persistence_image(pd[dim], birth_range=(0, 10), pers_range=(0, 10), pixel_size=1,
                                                      sigma=1) for lang, pd in pds.items()}
            elif metric == 'cosine' and dim == 0:
                pds_vec = {
                    lang: vectorise_persistence_image(pd[dim], birth_range=(0, .1), pers_range=(0, 1), pixel_size=.1,
                                                      sigma=.1) for lang, pd in pds.items()}
            elif metric == 'cosine' and dim > 0:
                pds_vec = {
                    lang: vectorise_persistence_image(pd[dim], birth_range=(0, 1), pers_range=(0, 1), pixel_size=.1,
                                                      sigma=.1) for lang, pd in pds.items()}
            else:  # Should not occur
                raise ValueError('invalid metric-dimension combination for persistence_image distance [check code]')

        if distance == 'bars_statistics':
            pds_vec = {lang: vectorise_bars_statistics(pd[dim], only_death=True if dim == 0 else False) for lang, pd in
                       pds.items()}

        distances_lower_triangular_matrix = []
        for i in range(1, len(langs)):
            row = []
            for j in range(i):
                if distance in ('bottleneck', 'sliced_wasserstein'):
                    row.append(compare_pds(pds[langs[i]][dim], pds[langs[j]][dim], distance))
                else:
                    row.append(np.linalg.norm(pds_vec[langs[i]] - pds_vec[langs[j]]))
            distances_lower_triangular_matrix.append(row)
        filename = f'{out_folder}/pddmat.{experiment_name}.n{n}.{metric}.{distance}.d{dim}.txt'  # PATH TO OUTPUT
        save_matrix(filename, distances_lower_triangular_matrix, langs)
        print(f'Time: {time.perf_counter() - ts:.1f} s, matrix saved to: {filename}')
        print()


if __name__ == '__main__':
    main()
