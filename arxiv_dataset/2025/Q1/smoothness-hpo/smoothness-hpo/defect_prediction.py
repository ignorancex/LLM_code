import os
import random

from copy import deepcopy
from src.util import load_defect_data, get_many_random_hyperparams, get_smoothness, run_experiment, defect_file_dic


hpo_space = {
    'n_units': (2, 6),
    'n_layers': (2, 5),
    'preprocessor': ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
    'wfo': [False, True],
    'smote': [False, True],
    'ultrasample': [False, True],
    'smooth': [False, True],
}

file_number = os.getenv('SLURM_JOB_ID') or random.randint(1, 10000)
for filename in defect_file_dic.keys():
    file = open(f'runs-{file_number}.txt', 'a')
    print(f'{filename}:', file=file)

    data_orig = load_defect_data(filename)

    best_betas = []
    best_configs = []
    keep_configs = 5
    num_configs = 30

    configs = get_many_random_hyperparams(hpo_space, num_configs)

    for config in configs:
        data = deepcopy(data_orig)
        print('[main] Computing smoothness for config:', config)
        smoothness = get_smoothness(data, 2, **config)
        print(f'Config: {config}\nSmoothness: {smoothness}', file=file)
        file.flush()

        if len(best_betas) < keep_configs or smoothness > min(best_betas):
            best_betas.append(smoothness)
            best_configs.append(config)

            best_betas, best_configs = zip(*sorted(zip(best_betas, best_configs), reverse=True, key=lambda x: x[0]))
            best_betas = list(best_betas[:keep_configs])
            best_configs = list(best_configs[:keep_configs])
    
    for beta, config in zip(best_betas, best_configs):
        data = deepcopy(data_orig)
        print(f'Config: {config}\nbeta: {beta}', file=file)
        print('[main] Accuracy:', run_experiment(data, 2, **config), file=file)
        file.flush()
