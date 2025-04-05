import gc

from copy import deepcopy

from src.util import get_many_random_hyperparams, get_smoothness, run_experiment

import numpy as np
import pandas as pd

from raise_utils.data import Data
from sklearn.model_selection import train_test_split
from tqdm import tqdm


datasets = ['ant', 'cassandra', 'commons', 'derby',
            'jmeter', 'lucene-solr', 'maven', 'tomcat']
base_path = './data/static-code/'

hpo_space = {
    'n_units': (2, 6),
    'n_layers': (2, 5),
    'preprocessor': ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
    'wfo': [False, True],
    'smote': [False, True],
    'ultrasample': [False, True],
    'smooth': [False, True],
}


def load_data(dataset: str) -> Data:
    train_file = base_path + 'train/' + dataset + '_B_features.csv'
    test_file = base_path + 'test/' + dataset + '_C_features.csv'

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    df = pd.concat((train_df, test_df), join='inner')

    del train_df, test_df
    gc.collect()

    X = df.drop('category', axis=1)
    y = df['category']

    del df
    gc.collect()

    y[y == 'close'] = 1
    y[y == 'open'] = 0

    y = np.array(y, dtype=np.float32)

    X = X.select_dtypes(
        exclude=['object']).astype(np.float32)

    if dataset == 'maven':
        data = Data(*train_test_split(X, y, test_size=.5, shuffle=False))
    else:
        data = Data(*train_test_split(X, y, test_size=.2, shuffle=False))

    data.x_train = np.array(data.x_train)
    data.y_train = np.array(data.y_train)

    del X, y
    gc.collect()

    return data


def run_all_experiments():
    """
    Runs all experiments
    """
    file = open('sensitivity_results.txt', 'a')

    for dataset in datasets[2:]:
        print(f'Running {dataset}:', file=file, flush=True)
        data_orig = load_data(dataset)

        num_configs = 100

        configs = get_many_random_hyperparams(hpo_space, num_configs)

        for config in tqdm(configs):
            try:
                data = deepcopy(data_orig)
                smoothness = get_smoothness(data, 2, **config)
                results = run_experiment(data, 2, **config)

                del data
                gc.collect()

                print(f"Config: {config} | Smoothness: {smoothness} | Results: {results}", file=file, flush=True)
            except ValueError:
                print("failed")

    print('Done.')


if __name__ == '__main__':
    run_all_experiments()
