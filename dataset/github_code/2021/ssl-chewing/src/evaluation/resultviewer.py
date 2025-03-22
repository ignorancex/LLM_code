import sys
from pathlib import Path
from pickle import Unpickler
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

import globalconfig as g_conf
from evaluation.postprocessing import chewing2meals
from evaluation.result import Result
#
# General configuration
from utilities.numpyutils import ensure_numpy_1d_vector

g_conf.set_experiment_name('exp_simclr_2')

#
# Create a list of timestamps
timestamps: List[str] = []
for file in g_conf.get_results_path().glob('ts_lopo_results *'):
    timestamps.append(file.name[16:35])
timestamps.sort()

timestamps = timestamps[-21:]
idx: int = int(sys.argv[1])

#
# Load the pickle file
timestamp: str = timestamps[idx]
result_pickle: Path = g_conf.get_results_path() / ('ts_lopo_results ' + timestamp + '.pickle')
result_flags: Path = g_conf.get_results_path() / ('flags ' + timestamp + '.txt')

if not result_pickle.exists():
    raise FileNotFoundError('File not found: ' + str(result_pickle))
with open(result_pickle, 'rb') as fh:
    result: Result = Unpickler(fh).load()
print('Loaded results from: ' + str(result_pickle))
print('timestamp: ' + timestamp + '\n')

if not result_flags.exists():
    raise FileNotFoundError('File not found: ' + str(result_flags))
print('FLAGS')
with open(result_flags, 'r') as fh:
    for i in range(6):
        print(fh.readline()[:-1])

#
result._compute_predicted()

fs_Hz: float = 1.0  # Equal to wstep_sec of the experiment
col_idx: int = 1  # Corresponds to second column, where label is '1'
y_true: np.ndarray = ensure_numpy_1d_vector(result._y)
y_pred: np.ndarray = ensure_numpy_1d_vector(result._y_pred)
y_pred_2: np.ndarray = chewing2meals(y_pred, fs_Hz)

#
# View the results
# print('--- PER-WINDOW RESULT ---')
# print(classification_report(y_true, y_pred))
print('\nCUMULATIVE MEAL DURATION RESULT')
print(classification_report(y_true, y_pred_2))

exit(0)
#
# Plot some figures
plt.figure()
plt.plot(y_true)
plt.plot(y_pred * 0.9)
plt.plot(y_pred_2 * 0.8)
plt.grid(True)
plt.legend(['y_true', 'y_pred', 'y_pred_2'])
plt.show()
