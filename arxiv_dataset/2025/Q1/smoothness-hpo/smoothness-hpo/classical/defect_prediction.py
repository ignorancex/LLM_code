from typing import Tuple
import os

import numpy as np
from raise_utils.transforms.wfo import fuzz_data
from raise_utils.transforms import Transform
from raise_utils.learners import Autoencoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

from smoothness.configs import learner_configs
from smoothness.hpo.bohb import BohbHPO
from smoothness.data import load_defect_prediction_data, remove_labels
from smoothness.hpo.util import get_learner

import ses

config_space = {
    'preprocessor': ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
    'wfo': [False, True],
    'smote': [False, True],
    'ultrasample': [False, True],
    'smooth': [False, True],
}
learner_name = 'ff'
dataset = 'camel1'


def data_fn(config: dict) -> Tuple[np.array, np.array, np.array, np.array]:
    data = load_defect_prediction_data(dataset)

    transform = Transform(config['preprocessor'])
    transform.apply(data)
    x_train, x_test, y_train, y_test = data.x_train, data.x_test, data.y_train, data.y_test

    if config['smooth']:
        print('[get_model] Running smooth')
        x_train, y_train = remove_labels(x_train, y_train)
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
        smote = SMOTE()
        x_train, y_train = smote.fit_resample(x_train, y_train)

    return x_train, x_test, y_train, y_test


def query_fn(config: dict, seed: int = 42, budget: int = 100):
    x_train, x_test, y_train, y_test = data_fn(config)

    # Comment the below if statements for MulticlassDL.
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)

    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)

    learner = get_learner(learner_name, config)
    learner.fit(x_train, y_train)
    preds = learner.predict(x_test)

    return accuracy_score(y_test, preds)


n_jobs = 20
if __name__ == '__main__':
    hpo_space = {**config_space, **learner_configs[learner_name]}

    hpo = BohbHPO(hpo_space, learner_name, query_fn)

    try:
        scores, time = hpo.run(1, 30)

        # Notify me
        with open('.status', 'r') as f:
            lines = int(f.readline())

        if lines + 1 >= n_jobs:
            ses.send_email('ARC Success Notification', 'All jobs completed.')
        else:
            with open('.status', 'w') as f:
                f.write(str(lines + 1))

        print(f'Accuracy: {np.median(scores)}\nTime: {time}')
    except:
        ses.send_email('ARC Failure Notification', f'Run {os.getenv("SLURM_JOB_ID")} failed.')
        n_jobs -= 1
