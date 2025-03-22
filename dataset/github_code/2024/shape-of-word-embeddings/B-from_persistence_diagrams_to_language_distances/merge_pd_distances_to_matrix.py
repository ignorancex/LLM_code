import numpy as np
import time
import argparse
import json
from pathlib import Path
import os

def init_args():
    parser = argparse.ArgumentParser(
        prog='Compute persistence diagram distances')
    parser.add_argument('--name', type=str, required=True,
        help='Experiment name.')
    parser.add_argument('--data_folder', type=str, required=True,
        help='Single distances folder path.')
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

    return parser.parse_args()

def save_matrix(filename, mat, labels, line_sep='\n', decimal_places=10):
    with open(filename, 'w') as file:
        file.write(' '.join(labels) + line_sep)
        for i, row in enumerate(mat):
            if i > 0:
                file.write(' '.join([f'{x : .{decimal_places}f}' for x in row[:i]]) + line_sep)

def main():
    args = init_args()
    experiment_name = args.name
    data_folder = Path(args.data_folder)
    output_folder = Path(args.output_folder)
    n = args.number

    languages = sorted(args.languages)
    language_to_index = {lang: i for i, lang in enumerate(languages)}

    metrics = args.metrics
    dimensions = args.dimensions
    distances = args.distances
    parameters = [(metric, dimension, distance) for metric in metrics for dimension in dimensions for distance in distances]

    matrices = {
        parameter_tuple : np.zeros((len(languages), len(languages)), dtype=float)
        for parameter_tuple in parameters
    }
    matrices_check = {
        parameter_tuple : np.triu(np.ones((len(languages), len(languages)), dtype=bool), 0)
        for parameter_tuple in parameters
    }
    
    filename_list = [filename for filename in os.listdir(data_folder) if filename.startswith(f'distances.{experiment_name}') and filename.endswith('.json')]
    print(f'Collected {len(filename_list)} data files in {data_folder}.')
    for filename in filename_list:
        with open(data_folder / filename, 'r') as file:
            data = json.load(file)
        for data_entry in data:
            # === CHECK ===
            if not data_entry['words'] == n:
               raise ValueError(f'The number of words is inconsistent for {filename}')
            if not data_entry['experiment_name'] == experiment_name:
               raise ValueError(f'The experiment_name is inconsistent for {filename}')
            # =============
            metric = data_entry['metric']
            dimension = data_entry['persistent_diagram_dim']
            distance = data_entry['pd_metric']
            if (metric, dimension, distance) in parameters:
                i, j = sorted([language_to_index[lang] for lang in data_entry['languages']], reverse=True)
                matrices[(metric, dimension, distance)][i,j] = data_entry['value']
                matrices_check[(metric, dimension, distance)][i,j] = True

    for metric, dimension, distance in parameters:
        if matrices_check[(metric, dimension, distance)].all():
            print(f'Matrix for {(metric, dimension, distance)} succesfully constructed.')
            output_matrix_filename = output_folder / f"pddmat.{experiment_name}.n{n}.{metric}.{distance}.d{dimension}.txt"
            save_matrix(output_matrix_filename, matrices[(metric, dimension, distance)], languages)
            print(f'    Saved to > {output_matrix_filename}')
        else:
            print(f'Matrix for {(metric, dimension, distance)} HAS INCOMPLETE DATA! Missing {(1 - matrices_check[(metric, dimension, distance)]).sum()} values.')

if __name__ == '__main__':
    main()