import gc
import random

import numpy as np
import pandas as pd

from ConfigSpace import ConfigurationSpace
from keras import backend as K
from raise_utils.data import Data, DataLoader
from raise_utils.hooks import Hook
from raise_utils.learners import Autoencoder
from raise_utils.metrics import ClassificationMetrics
from raise_utils.transforms import Transform
from raise_utils.transforms.remove_labels import remove_labels
from scipy.spatial import KDTree
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical


# Dataset filenames
defect_file_dic = {'ivy':     ['ivy-1.1.csv', 'ivy-1.4.csv', 'ivy-2.0.csv'],
            'lucene':  ['lucene-2.0.csv', 'lucene-2.2.csv', 'lucene-2.4.csv'],
            'poi':     ['poi-1.5.csv', 'poi-2.0.csv', 'poi-2.5.csv', 'poi-3.0.csv'],
            'synapse': ['synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv'],
            'velocity': ['velocity-1.4.csv', 'velocity-1.5.csv', 'velocity-1.6.csv'],
            'camel': ['camel-1.0.csv', 'camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv'],
            'jedit': ['jedit-3.2.csv', 'jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv', 'jedit-4.3.csv'],
            'log4j': ['log4j-1.0.csv', 'log4j-1.1.csv', 'log4j-1.2.csv'],
            'xalan': ['xalan-2.4.csv', 'xalan-2.5.csv', 'xalan-2.6.csv', 'xalan-2.7.csv'],
            'xerces': ['xerces-1.2.csv', 'xerces-1.3.csv', 'xerces-1.4.csv']
            }


def load_defect_data(dataset: str) -> Data:
    def _binarize(x, y): y[y > 1] = 1
    base_path = '../DODGE Data/defect/'
    dataset = DataLoader.from_files(
        base_path=base_path, files=defect_file_dic[dataset], hooks=[Hook('binarize', _binarize)])

    return dataset


def run_experiment(data: Data, n_class: int, wfo: bool, smote: bool, ultrasample: bool, smooth: bool, n_units: int,
    n_layers: int, preprocessor: str) -> list[float]:
    print('[run_experiment] Getting model')
    model, data = get_model(data, n_class, wfo, smote, ultrasample, smooth, n_units, n_layers, preprocessor)

    if n_class > 2 and len(data.y_train) == 1:
        data.y_train = to_categorical(data.y_train, n_class)
        data.y_test = to_categorical(data.y_test, n_class)
    print('[run_experiment] Got model')
    model.fit(data.x_train, data.y_train, epochs=100, verbose=1, batch_size=128)
    print('[run_experiment] Fit model')

    if n_class == 2:
        y_pred = (model.predict(data.x_test) > 0.5).astype('int32')
    else:
        y_pred = np.argmax(model.predict(data.x_test), axis=-1)

    if n_class > 2 and len(data.y_test) > 1:
        data.y_test = np.argmax(data.y_test, axis=1)

    del model
    gc.collect()
    K.clear_session()

    metrics = ClassificationMetrics(data.y_test, y_pred)
    metrics.add_metrics(['pd-pf', 'pd', 'pf', 'prec', 'auc'])
    return metrics.get_metrics()


def get_smoothness(data: Data, n_class: int, wfo: bool, smote: bool, ultrasample: bool, smooth: bool, n_units: int,
    n_layers: int, preprocessor: str) -> float:
    print('[get_smoothness] Getting model')
    model, data = get_model(data, n_class, wfo, smote, ultrasample, smooth, n_units, n_layers, preprocessor)

    if n_class > 2 and len(data.y_train) == 1:
        data.y_train = to_categorical(data.y_train, n_class)
        data.y_test = to_categorical(data.y_test, n_class)
    print('[get_smoothness] Got model')

    # Fit for one epoch before computing smoothness
    model.fit(data.x_train, data.y_train, epochs=1, verbose=1, batch_size=128)
    print('[get_smoothness] Fit model')

    def func(xb):
        _model = Model(inputs=[model.layers[0].input], outputs=[model.layers[-2].output])
        return _model(xb)

    batch_size = 128
    Kz = 0.
    Kw = 0.
    for i in range((len(data.x_train) - 1) // batch_size + 1):
        start_i = i * batch_size
        end_i = start_i + batch_size
        xb = data.x_train[start_i:end_i]

        activ = np.linalg.norm(func([xb]))
        if activ > Kz:
            Kz = activ

        assert len(model.layers[-1].weights[0].shape) == 2
        W = np.linalg.norm(model.layers[-1].weights[0])
        if W > Kw:
            Kw = W

    if Kw == 0:
        return 0.

    del model
    gc.collect()
    K.clear_session()

    return Kz / Kw


def get_random_hyperparams(options: dict) -> dict:
    """
    Get hyperparameters from options.
    """
    hyperparams = {}
    for key, value in options.items():
        if isinstance(value, list):
            hyperparams[key] = random.choice(value)
        elif isinstance(value, tuple):
            hyperparams[key] = random.randint(value[0], value[1])
    return hyperparams


def get_many_random_hyperparams(options: dict, n: int) -> list:
    """
    Get n hyperparameters from options.
    """
    hyperparams = []
    for _ in range(n):
        hyperparams.append(get_random_hyperparams(options))
    return hyperparams


def hp_space_to_configspace(hp_space: dict) -> ConfigurationSpace:
    return ConfigurationSpace(hp_space)


def remove_labels_legacy(data: Data) -> Data:
    # "Remove" labels
    lost_idx = np.random.choice(
        len(data.y_train), size=int(len(data.y_train) - np.sqrt(len(data.y_train))), replace=False)

    x_lost = data.x_train[lost_idx]
    x_rest = np.delete(data.x_train, lost_idx, axis=0)
    y_lost = data.y_train[lost_idx]
    y_rest = np.delete(data.y_train, lost_idx, axis=0)

    if len(x_lost.shape) == 1:
        x_lost = x_lost.reshape(1, -1)
    if len(x_rest.shape) == 1:
        x_rest = x_rest.reshape(1, -1)

    # Impute data
    tree = KDTree(x_rest)
    _, idx = tree.query(x_lost, k=int(np.sqrt(np.sqrt(len(x_rest)))), p=1)
    y_lost = mode(y_rest[idx], axis=1)[0]
    y_lost = y_lost.reshape((y_lost.shape[0], y_lost.shape[-1]))

    assert len(x_lost) == len(y_lost)

    data.x_train = np.concatenate((x_lost, x_rest), axis=0)
    data.y_train = np.concatenate((y_lost, y_rest), axis=0)
    return data


def get_model(data: Data, n_class: int, wfo: bool, smote: bool, ultrasample: bool, smooth: bool,
    n_units: int, n_layers: int, preprocessor: str) -> tuple:
    """
    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {str} name - The name of the experiment.
    :param {dict} config - The config to use. Must be one in the format used in `process_configs`.
    """
    transform = Transform(preprocessor)
    transform.apply(data)

    if smooth:
        data.x_train = np.array(data.x_train)
        data.y_train = np.array(data.y_train)

        print('[get_model] Running smooth')
        if n_class == 2:
            data.x_train, data.y_train = remove_labels(data.x_train, data.y_train)
        else:
            data = remove_labels_legacy(data)
        print('[get_model] Finished running smooth')

    if ultrasample:
        # Apply WFO
        print('[get_model] Running ultrasample:wfo')
        transform = Transform('wfo')
        transform.apply(data)
        print('[get_model] Finished running ultrasample:wfo')

        # Reverse labels
        data.y_train = 1. - data.y_train
        data.y_test = 1. - data.y_test

        # Autoencode the inputs
        loss = 1e4
        count = 0
        while loss > 1e3 and count < 5:
            ae = Autoencoder(n_layers=2, n_units=[10, 7], n_out=5)
            ae.set_data(*data)
            print('[get_model] Fitting autoencoder')
            ae.fit()
            print('[get_model] Fit autoencoder')

            loss = ae.model.history.history['loss'][-1]
            count += 1

        data.x_train = ae.encode(np.array(data.x_train))
        data.x_test = ae.encode(np.array(data.x_test))

    learner = Sequential()
    data.y_train = data.y_train.astype('float32')

    data.x_train, data.x_test, data.y_train, data.y_test = \
        map(np.array, (data.x_train, data.x_test, data.y_train, data.y_test))
    data.y_train = data.y_train.squeeze()
    data.y_test = data.y_test.squeeze()

    if wfo:
        print('[get_model] Running wfo')
        transform = Transform('wfo')
        transform.apply(data)
        print('[get_model] Finished running wfo')

    if smote:
        print('[get_model] Running smote')
        transform = Transform('smote')

        if n_class > 2 and len(data.y_train.shape) > 1:
            data.y_train = np.argmax(data.y_train, axis=1)

        transform.apply(data)

        if n_class > 2 and len(data.y_train.shape) == 1:
            data.y_train = to_categorical(data.y_train, n_class)

        print('[get_model] Finished running smote')

    for _ in range(n_layers):
        learner.add(Dense(n_units, activation='relu'))

    if n_class == 2:
        learner.add(Dense(1, activation='sigmoid'))
    else:
        learner.add(Dense(n_class, activation='softmax'))

    learner.compile(loss='binary_crossentropy' if n_class == 2 else 'categorical_crossentropy', optimizer='sgd')

    return learner, data


def split_data(filename: str, data: Data, n_classes: int) -> Data:
    if n_classes == 2:
        if filename == 'firefox.csv':
            data.y_train = data.y_train < 4
            data.y_test = data.y_test < 4
        elif filename == 'chromium.csv':
            data.y_train = data.y_train < 5
            data.y_test = data.y_test < 5
        else:
            data.y_train = data.y_train < 6
            data.y_test = data.y_test < 6
    elif n_classes == 3:
        data.y_train = np.where(data.y_train < 2, 0,
                                np.where(data.y_train < 6, 1, 2))
        data.y_test = np.where(
            data.y_test < 2, 0, np.where(data.y_test < 6, 1, 2))
    elif n_classes == 5:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 3, 1, np.where(
            data.y_train < 6, 2, np.where(data.y_train < 21, 3, 4))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 3, 1, np.where(
            data.y_test < 6, 2, np.where(data.y_test < 21, 3, 4))))
    elif n_classes == 7:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 2, 1, np.where(data.y_train < 3, 2, np.where(
            data.y_train < 6, 3, np.where(data.y_train < 11, 4, np.where(data.y_train < 21, 5, 6))))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 2, 1, np.where(data.y_test < 3, 2, np.where(
            data.y_test < 6, 3, np.where(data.y_test < 11, 4, np.where(data.y_test < 21, 5, 6))))))
    else:
        data.y_train = np.where(data.y_train < 1, 0, np.where(data.y_train < 2, 1, np.where(data.y_train < 3, 2, np.where(data.y_train < 4, 3, np.where(
            data.y_train < 6, 4, np.where(data.y_train < 8, 5, np.where(data.y_train < 11, 6, np.where(data.y_train < 21, 7, 8))))))))
        data.y_test = np.where(data.y_test < 1, 0, np.where(data.y_test < 2, 1, np.where(data.y_test < 3, 2, np.where(data.y_test < 4, 3, np.where(
            data.y_test < 6, 4, np.where(data.y_test < 8, 5, np.where(data.y_test < 11, 6, np.where(data.y_test < 21, 7, 8))))))))

    if n_classes > 2:
        data.y_train = to_categorical(data.y_train, num_classes=n_classes).astype(int)
        data.y_test = to_categorical(data.y_test, num_classes=n_classes).astype(int)

    return data


def load_issue_lifetime_prediction_data(filename: str, n_classes: int) -> Data:
    df = pd.read_csv(f'./data/{filename}.csv')
    df.drop(['Unnamed: 0', 'bugID'], axis=1, inplace=True)
    _df = df[['s1', 's2', 's3', 's4', 's5', 's6', 's8', 'y']]
    _df['s70'] = df['s7'].apply(lambda x: eval(x)[0])
    _df['s71'] = df['s7'].apply(lambda x: eval(x)[1])
    _df['s72'] = df['s7'].apply(lambda x: eval(x)[2])
    _df['s90'] = df['s9'].apply(lambda x: eval(x)[0])
    _df['s91'] = df['s9'].apply(lambda x: eval(x)[1])

    if filename == 'firefox':
        _df['s92'] = df['s9'].apply(lambda x: eval(x)[2])

    x = _df.drop('y', axis=1)
    y = _df['y']

    data = Data(*train_test_split(x, y))
    data = split_data(filename, data, n_classes)
    return data
