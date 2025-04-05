import gc
import os
import random
import time

from copy import deepcopy
from functools import partial

import neps
import numpy as np
import pandas as pd

from raise_utils.data import Data
from raise_utils.learners import Autoencoder, FeedforwardDL
from raise_utils.metrics import ClassificationMetrics
from raise_utils.transforms import Transform
from raise_utils.transforms.remove_labels import remove_labels
from sklearn.model_selection import train_test_split


datasets = ['ant', 'cassandra', 'commons', 'derby',
            'jmeter', 'lucene-solr', 'maven', 'tomcat']
base_path = '../../DODGE Data/static_code/'

hpo_space = {
    'n_units': neps.Integer(2, 6, default=4, default_confidence="low"),
    'n_layers': neps.Integer(2, 5, default=3, default_confidence="low"),
    'preprocessor': neps.Categorical(
        ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
        default="normalize",
        default_confidence="low"
    ),
    'wfo': neps.Categorical([False, True], default=False, default_confidence="low"),
    'smote': neps.Categorical([False, True], default=False, default_confidence="low"),
    'ultrasample': neps.Categorical([False, True], default=False, default_confidence="low"),
    'smooth': neps.Categorical([False, True], default=False, default_confidence="low"),
}


def load_data(dataset: str) -> Data:
    train_file = base_path + 'train/' + dataset + '_B_features.csv'
    test_file = base_path + 'test/' + dataset + '_C_features.csv'

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    df = pd.concat((train_df, test_df), join='inner')

    X = df.drop('category', axis=1)
    y = df['category']

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

    del train_df, test_df, X, y

    return data


def run(data: Data, config: dict):
    """
    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {Config} config - The config to use. Must be one in the format used in `process_configs`.
    """
    if config['smooth']:
        data.x_train = np.array(data.x_train)
        data.y_train = np.array(data.y_train)
        data.x_train, data.y_train = remove_labels(data.x_train, data.y_train)

    if config['ultrasample']:
        # Apply WFO
        transform = Transform('wfo')
        transform.apply(data)

        # Reverse labels
        data.y_train = 1. - data.y_train
        data.y_test = 1. - data.y_test

        # Autoencode the inputs
        ae = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5)
        ae.set_data(*data)
        ae.fit()

        data.x_train = ae.encode(np.array(data.x_train))
        data.x_test = ae.encode(np.array(data.x_test))

        del ae

    learner = FeedforwardDL(n_layers=config['n_layers'], n_units=config['n_units'],
                            weighted=True, wfo=config['wfo'],
                            smote=config['smote'], n_epochs=100)

    learner.set_data(*data)
    learner.fit()

    # Get the results.
    preds = learner.predict(data.x_test)
    m = ClassificationMetrics(data.y_test, preds)
    m.add_metrics(['pd-pf', 'pd', 'pf', 'prec', 'auc'])
    results = m.get_metrics()
    print(f'Results: {results}')

    del learner
    del data
    return results


def run_all_experiments():
    """
    Runs all experiments 10 times each.
    """
    file_number = os.getenv('SLURM_JOB_ID') or random.randint(1, 10000)
    file = open(f'runs-{file_number}.txt', 'a')

    for dataset in datasets[6:]:
        print(f'{dataset}:', file=file)
        data_orig = load_data(dataset)

        def objective(data_orig, **config) -> dict:
            start = time.time()
            data = deepcopy(data_orig)

            try:
                results = run(data, config)
            except:
                results = [0., 0., 0., 0., 0.]

            pdpf = results[0]
            end = time.time()

            print(f"Result: {results} | Time: {end - start}", file=file, flush=True)

            del data
            del results
            gc.collect()

            return {
                "loss": -pdpf,
                "info_dict": {
                    "test_score": pdpf,
                    "train_time": end - start
                }
            }

        # 20 repeats
        for i in range(5):
            neps.run(
                run_pipeline=partial(objective, data_orig),
                pipeline_space=hpo_space,
                root_directory=f"priorband_{dataset}_{i}",
                max_evaluations_total=50
            )

    print('Done.')


if __name__ == '__main__':
    run_all_experiments()
