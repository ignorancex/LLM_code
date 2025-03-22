from pathlib import Path
from pickle import Unpickler
from typing import List

import numpy as np
from sklearn.metrics import classification_report

import globalconfig as g_conf
from evaluation.postprocessing import chewing2meals
from evaluation.result import Result
from utilities.numpyutils import ensure_numpy_1d_vector
from utilities.typingutils import is_typed_list


# Experiment configuration
g_conf.set_experiment_name('exp_simclr_2')


class ExperimentArgs:
    def __init__(self):
        self.temperature: float = 0.0
        self.projection_head: str = ''
        self.g_keep: int = 0

    def parse(self, args: List[str]):
        assert is_typed_list(args, str)

        for arg in args:
            if arg[:8] == '--temper':
                self.temperature = float(arg[14:-1])
            elif arg[:8] == '--projec':
                self.projection_head = arg[18:-1]
            elif arg[:8] == '--g_keep':
                self.g_keep = int(arg[9:-1])


def main1(timestamp: str):
    result_flags: Path = g_conf.get_results_path() / ('flags ' + timestamp + '.txt')
    result_pickle: Path = g_conf.get_results_path() / ('ts_lopo_results ' + timestamp + '.pickle')

    if not result_flags.exists():
        raise FileNotFoundError('File not found: ' + str(result_flags))
    if not result_pickle.exists():
        raise FileNotFoundError('File not found: ' + str(result_pickle))

    args = ExperimentArgs()
    with open(result_flags, 'r') as fh:
        args.parse(fh.readlines()[:6])

    result: Result = None
    with open(result_pickle, 'rb') as fh:
        result = Unpickler(fh).load()

    # print('Loaded results from: ' + str(result_pickle))
    # print('timestamp: ' + timestamp + '\n')

    # Main part
    result._compute_predicted()

    fs_Hz: float = 1.0  # Equal to wstep_sec of the experiment
    col_idx: int = 1  # Corresponds to second column, where label is '1'
    y_true: np.ndarray = ensure_numpy_1d_vector(result._y)
    y_pred: np.ndarray = ensure_numpy_1d_vector(result._y_pred)
    y_pred = chewing2meals(y_pred, fs_Hz)

    tmp = classification_report(y_true, y_pred, output_dict=True)
    precision: float = tmp['1']['precision']
    recall: float = tmp['1']['recall']
    f1score: float = tmp['1']['f1score']
    accuracy: float = tmp['accuracy']

    print(classification_report(y_true, y_pred))

    print('done')


if __name__ == '__main__':
    # Create a list of timestamps
    timestamps: List[str] = []
    for file in g_conf.get_results_path().glob('ts_lopo_results *'):
        timestamps.append(file.name[16:35])
    timestamps.sort()

    timestamps = timestamps[-21:]

    for timestamp in timestamps:
        main1(timestamp)

    print('done')
