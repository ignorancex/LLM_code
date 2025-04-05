import os
import random
from copy import deepcopy

from src.util import load_issue_lifetime_prediction_data, get_many_random_hyperparams, get_smoothness, run_experiment


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
for filename in ['chromium', 'firefox', 'eclipse']:
    n_class = 5
    file = open(f'runs-{file_number}.txt', 'a')
    print(f'{filename}-{n_class}:', file=file)

    data_orig = load_issue_lifetime_prediction_data(filename, n_class)

    best_betas = []
    best_configs = []
    keep_configs = 5
    num_configs = 30

    configs = get_many_random_hyperparams(hpo_space, num_configs)

    for config in configs:
        try:
            data = deepcopy(data_orig)
            smoothness = get_smoothness(data, n_class, **config)
            print(f'Config: {config}\nSmoothness: {smoothness}', file=file)
            if len(best_betas) < keep_configs or smoothness > min(best_betas):
                best_betas.append(smoothness)
                best_configs.append(config)
                best_betas, best_configs = zip(*sorted(zip(best_betas, best_configs), reverse=True, key=lambda x: x[0]))
                best_betas = list(best_betas[:keep_configs])
                best_configs = list(best_configs[:keep_configs])
        except:
            print(f'Failed to run config {config}')
    
    for config in best_configs:
        data = deepcopy(data_orig)
        print(config, file=file)
        print(run_experiment(data, n_class, **config), file=file)
