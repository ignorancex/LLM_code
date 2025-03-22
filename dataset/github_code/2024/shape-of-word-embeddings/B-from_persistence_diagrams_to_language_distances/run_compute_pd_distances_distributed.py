#
#   For the folders used, search for comments PATH TO DATA and PATH RO OUTPUT
#

import numpy as np
import persim
import time
import gudhi
import argparse
import json
from pathlib import Path


def init_args():
    parser = argparse.ArgumentParser(
        prog='Compute persistence diagram distances')
    parser.add_argument('--name', type=str, required=True,
                        help='Experiment name.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Output folder path.')
    parser.add_argument('--number', type=int, required=True,
                        help='Number of words used.')
    parser.add_argument('--maxdim', type=int, required=False, default=-1,
                        help='The maxdim used for ripser computation -- used only for loading bars from the correct file.')
    parser.add_argument('--dimensions', type=int, nargs='+', required=True,
                        help='The maximal dimension used.')
    parser.add_argument('--distances', type=str, nargs='+', required=True,
                        default=('bottleneck', 'sliced_wasserstein', 'persistence_image', 'bars_statistics'),
                        help='List of the distances between persistance diagrams to use.')
    parser.add_argument('--metrics', type=str, nargs='+', required=True,
                        help='List of the point-cloud metrics used.')
    parser.add_argument('--languages', type=str, nargs='+', required=True,
                        help='List of the languages.')
    parser.add_argument('--task_id', type=int, required=True,
                        help='Based on the task_id, a batch of jobs (distances) is computed.')
    parser.add_argument('--batch_size', type=int, required=False, default=100,
                        help='Size of one batch')

    return parser.parse_args()


def pair_to_number(i, j):
    return i * (i - 1) // 2 + j


def number_to_pair(index):
    i = int((1 + (1 + 8 * index) ** (1 / 2)) / 2)
    j = index - (i * (i - 1) // 2)
    return i, j


def job_id_to_parameter_indices(job_id, number_of_parameters, number_of_languages):
    k = job_id % number_of_parameters
    number = job_id // number_of_parameters
    i, j = number_to_pair(number)
    return k, (i, j)


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
    print("=== Script started: Compute pd distances distributed ===")
    args = init_args()
    experiment_name = args.name
    out_folder = Path(args.output_folder)
    n = args.number

    languages = args.languages

    metrics = args.metrics
    dimensions = args.dimensions
    maxdim = args.maxdim if args.maxdim > -1 else max(dimensions)
    distances = args.distances
    parameters = [(metric, dimension, distance) for metric in metrics for dimension in dimensions for distance in
                  distances]
    number_of_jobs = len(parameters) * len(languages) * (len(languages) - 1) // 2

    task_id = args.task_id
    batch_size = args.batch_size

    print(
        f"Experiment {experiment_name}: words={n}, #langs={len(languages)}, maxdim={maxdim}, task_id={task_id} --> jobs {task_id * batch_size}-{min((task_id + 1) * batch_size, number_of_jobs)}")

    results = []
    ts_all = time.perf_counter()
    for job_id in range(task_id * batch_size, min((task_id + 1) * batch_size, number_of_jobs)):

        k, (i, j) = job_id_to_parameter_indices(job_id, len(parameters), len(languages))
        metric, dimension, distance = parameters[k]
        language_1 = languages[i]
        language_2 = languages[j]

        print(f"    job {job_id : 6d}: {metric}, {dimension}, {distance} for {language_1} vs {language_2} ... ", end="")

        pds = {lang: load_bars(f'data/bars/bars.{lang}.300.n{n}.{metric}.d{maxdim}.txt') for lang in  # PATH TO DATA
               [language_1, language_2]}
        ts = time.perf_counter()

        if distance == 'persistence_image':
            if metric == 'euclidean' and dimension == 0:
                pds_vec = {lang: vectorise_persistence_image(pd[dimension], birth_range=(0, 1), pers_range=(0, 10),
                                                             pixel_size=1, sigma=1) for lang, pd in pds.items()}
            elif metric == 'euclidean' and dimension > 0:
                pds_vec = {lang: vectorise_persistence_image(pd[dimension], birth_range=(0, 10), pers_range=(0, 10),
                                                             pixel_size=1, sigma=1) for lang, pd in pds.items()}
            elif metric == 'cosine' and dimension == 0:
                pds_vec = {lang: vectorise_persistence_image(pd[dimension], birth_range=(0, .1), pers_range=(0, 1),
                                                             pixel_size=.1, sigma=.1) for lang, pd in pds.items()}
            elif metric == 'cosine' and dimension > 0:
                pds_vec = {lang: vectorise_persistence_image(pd[dimension], birth_range=(0, 1), pers_range=(0, 1),
                                                             pixel_size=.1, sigma=.1) for lang, pd in pds.items()}
            else:  # Should not occur
                raise ValueError('invalid metric-dimension combination for persistence_image distance [check code]')

        if distance == 'bars_statistics':
            pds_vec = {lang: vectorise_bars_statistics(pd[dimension], only_death=True if dimension == 0 else False) for
                       lang, pd in pds.items()}

        if distance in ('bottleneck', 'sliced_wasserstein'):
            value = compare_pds(pds[language_1][dimension], pds[language_2][dimension], distance)
        else:
            value = np.linalg.norm(pds_vec[language_1] - pds_vec[language_2])

        results.append({
            'experiment_name': experiment_name,
            'words': n,
            'embedding_dimension': 300,
            'job_id': job_id,
            'metric': metric,
            'persistent_diagram_dim': dimension,
            'pd_metric': distance,
            'languages': sorted([language_1, language_2]),
            'value': value
        })

        print(f'{time.perf_counter() - ts : .3f} s')

    filename = out_folder / f'distances.{experiment_name}.part{task_id:05d}.json'  # PATH TO OUTPUT
    with open(filename, 'w') as file:
        json.dump(results, file)

    print(f'Time: {time.perf_counter() - ts_all : .3f} s, distances saved to: {filename}')


if __name__ == '__main__':
    main()
