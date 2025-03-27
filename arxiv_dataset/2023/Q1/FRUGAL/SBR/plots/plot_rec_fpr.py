import os
import sys
import numpy as np
import pandas as pd

cwd = os.getcwd()
root = cwd[:os.getcwd().rfind('e-dom/') + len('e-dom/') - 1]
sys.path.append(f'{root}/src')
from model import ML, hp

import pdb


def read_txt(model, alg):
    metrics_for_all_repeats = list()
    records = list()
    with open(f'{root}/results/{model}_{alg}.txt', 'r') as infile:
        for line in infile:
            if line.startswith('## END'):
                metrics = pd.DataFrame(
                    records, columns=['rec', 'fpr', 'g_measure'])
                metrics.iloc[:, 1] = 1 - metrics.iloc[:, 1]
                metrics_for_all_repeats.append(metrics)
                records.clear()
                continue
            if not line.startswith('V'):
                continue
            line = line[2:-1]
            record = np.fromstring(line, dtype=float, sep=' ')
            records.append(record)

    return metrics_for_all_repeats


metrics_for_all_repeats = read_txt('ambari', 'random100')