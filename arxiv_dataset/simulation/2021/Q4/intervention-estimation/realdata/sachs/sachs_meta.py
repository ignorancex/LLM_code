import causaldag as cd
import os
from config import PROJECT_FOLDER, REALDATA_FOLDER
import numpy as np
import pandas as pd
import random

nnodes = 11
# this is the more recent version
true_edges_recent = [(0, 1),
    (1, 5),
    (2, 3),
    (2, 4),
    (4, 3),
    (7, 0),
    (7, 1),
    (7, 5),
    (7, 6),
    (7, 9),
    (7, 10),
    (8, 0),
    (8, 1),
    (8, 7),
    (8, 9),
    (8, 10)]

true_dag_recent = np.zeros((nnodes,nnodes))
for edge in true_edges_recent:
    true_dag_recent[edge] = 1



# this is what UT-IGSP paper uses
true_edges_old = [(0, 1),
    (1, 5),
    (2, 3),
    (2, 8),
    (3, 8),
    (4, 2),
    (4, 3),
    (4, 6),
    (7, 0),
    (7, 1),
    (7, 5),
    (7, 6),
    (7, 9),
    (7, 10),
    (8, 0),
    (8, 1),
    (8, 9),
    (8, 10)]

true_dag_old = np.zeros((nnodes,nnodes))
for edge in true_edges_old:
    true_dag_old[edge] = 1


SACHS_FOLDER = os.path.join(REALDATA_FOLDER, 'sachs')
SACHS_DATA_FOLDER = os.path.join(SACHS_FOLDER, 'data')
SACHS_ESTIMATED_FOLDER = os.path.join(SACHS_FOLDER, 'estimated')
SACHS_FIGURES_FOLDER = os.path.join(SACHS_FOLDER, 'figures')

os.makedirs(SACHS_FIGURES_FOLDER, exist_ok=True)
os.makedirs(SACHS_ESTIMATED_FOLDER, exist_ok=True)


def sachs_get_samples():
    sample_dict = dict()
    for file in os.listdir(SACHS_DATA_FOLDER):
        samples = pd.read_csv(os.path.join(SACHS_DATA_FOLDER, file), sep=',')
        iv_str = file.split('=')[1][:-4]
        ivs = frozenset({int(iv_str)}) if iv_str != '' else frozenset()
        sample_dict[ivs] = samples.values    

    obs_samples = sample_dict[frozenset()]
    all_samples = np.concatenate(tuple(sample_dict.values()), axis=0)

    setting_list = [
        {'known_interventions': iv_nodes}
        for iv_nodes, samples in sample_dict.items()
        if iv_nodes != frozenset()
    ]

    iv_samples_list = [sample_dict[setting['known_interventions']] for setting in setting_list]

    return obs_samples, iv_samples_list, setting_list
