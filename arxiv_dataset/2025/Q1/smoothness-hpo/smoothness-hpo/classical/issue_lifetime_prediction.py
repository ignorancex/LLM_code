import time

from typing import Tuple

import numpy as np

from imblearn.over_sampling import SMOTE
from raise_utils.data import Data
from raise_utils.learners import Autoencoder
from raise_utils.transforms import Transform
from raise_utils.transforms.remove_labels import remove_labels
from raise_utils.transforms.wfo import fuzz_data
from sklearn.metrics import accuracy_score
from smoothness.configs import learner_configs
from smoothness.data import load_issue_lifetime_prediction_data, remove_labels_legacy
from smoothness.hpo.smoothness import SmoothnessHPO
from smoothness.hpo.util import get_learner


config_space = {
    'preprocessor': ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
    'wfo': [False, True],
    'smote': [False, True],
    'ultrasample': [False, True],
    'smooth': [False, True],
}
clf = 'ff'


def data_fn(config: dict) -> Tuple[np.array, np.array, np.array, np.array]:
    n_class = 2
    x_train, x_test, y_train, y_test = load_issue_lifetime_prediction_data('chromium', n_class)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    transform = Transform(config['preprocessor'])
    data = Data(x_train, x_test, y_train, y_test)
    transform.apply(data)
    x_train, x_test, y_train, y_test = data.x_train, data.x_test, data.y_train, data.y_test

    if config['smooth']:
        print('[get_model] Running smooth')
        if n_class == 2:
            x_train, y_train = remove_labels(x_train, y_train)
        else:
            x_train, y_train = remove_labels_legacy(x_train, y_train)
        print('[get_model] Finished running smooth')

    if config['ultrasample']:
        # Apply WFO
        print('[get_model] Running ultrasample:wfo')
        x_train, y_train = fuzz_data(x_train, y_train)
        print('[get_model] Finished running ultrasample:wfo')

        # Reverse labels
        y_train = 1. - y_train
        y_test = 1. - y_test

        # Autoencode the inputs
        loss = 1e4
        while loss > 1e3:
            ae = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5)
            ae.set_data(x_train, y_train, x_test, y_test)
            print('[get_model] Fitting autoencoder')
            ae.fit()
            print('[get_model] Fit autoencoder')

            loss = ae.model.history.history['loss'][-1]

        x_train = np.array(ae.encode(x_train))
        x_test = np.array(ae.encode(x_test))

    if config['wfo']:
        print('[get_model] Running wfo')
        x_train, y_train = fuzz_data(x_train, y_train)
        print('[get_model] Finished running wfo')

    if config['smote']:
        if n_class > 2 and len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)

        smote = SMOTE()
        x_train, y_train = smote.fit_resample(x_train, y_train)

    return x_train, x_test, y_train, y_test


def query_fn(config: dict, seed: int = 42, budget: int = 100, **kwargs):
    start = time.time()
    x_train, x_test, y_train, y_test = data_fn(config)

    # Comment the below if statements for MulticlassDL.
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)

    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)

    learner = get_learner(clf, config)
    if clf == 'ff':
        learner.set_data(x_train, y_train, x_test, y_test)
        learner.fit()
    else:
        learner.fit(x_train, y_train)
    preds = learner.predict(x_test)

    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)

    if len(preds.shape) == 2:
        preds = np.argmax(preds, axis=1)

    end = time.time()
    # NOTE: Only for DEHB: for all others, return the commented out version.
    return {
        "fitness": -accuracy_score(y_test, preds),
        "cost": end - start
    }

    # return accuracy_score(y_test, preds)


for learner in [clf]:
    hpo_space = {**config_space, **learner_configs[learner]}

    hpo = SmoothnessHPO(hpo_space, learner, query_fn, data_fn)

    scores, _time = hpo.run(1, 30)
    print(f'Accuracy: {np.median(scores)}\nTime: {_time}')
