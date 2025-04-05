import os
import random
from copy import deepcopy

from src.util import load_issue_lifetime_prediction_data, get_many_random_hyperparams, get_smoothness, get_approx_smoothness, run_experiment


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

    best_betas_exact = []
    best_configs_exact = []
    best_betas_approx = []
    best_configs_approx = []
    keep_configs = 5
    num_configs = 30

    configs = get_many_random_hyperparams(hpo_space, num_configs)

    for config in configs:
        try:
            data = deepcopy(data_orig)
            smoothness_exact = get_smoothness(data, n_class, **config)
            smoothness_approx = get_approx_smoothness(data, n_class, **config)
            print(
                f'Config: {config}\nExact smoothness: {smoothness_exact}\nApproximate smoothness: {smoothness_approx}', file=file)

            if len(best_betas_exact) < keep_configs or smoothness_exact > min(best_betas_exact):
                best_betas_exact.append(smoothness_exact)
                best_configs_exact.append(config)
                best_betas_exact, best_configs_exact = zip(
                    *sorted(zip(best_betas_exact, best_configs_exact), reverse=True, key=lambda x: x[0]))
                best_betas_exact = list(best_betas_exact[:keep_configs])
                best_configs_exact = list(best_configs_exact[:keep_configs])

            if len(best_betas_approx) < keep_configs or smoothness_approx > min(best_betas_approx):
                best_betas_approx.append(smoothness_approx)
                best_configs_approx.append(config)
                best_betas_approx, best_configs_approx = zip(
                    *sorted(zip(best_betas_approx, best_configs_approx), reverse=True, key=lambda x: x[0]))
                best_betas_approx = list(best_betas_approx[:keep_configs])
                best_configs_approx = list(best_configs_approx[:keep_configs])
        except:
            print(f'Failed to run config {config}')

    count = 0
    for exact_config in best_configs_exact:
        if exact_config in best_configs_approx:
            count += 1

    print(f'Score: {count / 5}')
