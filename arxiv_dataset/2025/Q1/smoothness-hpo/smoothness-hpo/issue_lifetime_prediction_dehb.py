import os
import gc
from dehb import DEHB
import time
import random

from copy import deepcopy
from ConfigSpace import Configuration
from src.util import load_issue_lifetime_prediction_data, run_experiment, hp_space_to_configspace


hpo_space = hp_space_to_configspace({
    'n_units': (2, 6),
    'n_layers': (2, 5),
    'preprocessor': ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
    'wfo': [False, True],
    'smote': [False, True],
    'ultrasample': [False, True],
    'smooth': [False, True],
})


file_number = os.getenv('SLURM_JOB_ID') or random.randint(1, 10000)
for filename in ['chromium', 'firefox', 'eclipse']:
    file = open(f'runs-{file_number}.txt', 'a')
    n_class = 3
    file = open(f'runs-{file_number}.txt', 'a')
    print(f'Running: {filename}-{n_class}:', file=file)

    data_orig = load_issue_lifetime_prediction_data(filename, n_class)
     
    def objective(config: Configuration, fidelity: float, **_) -> dict:
        start = time.time()
        data = deepcopy(data_orig)
        acc = run_experiment(data, n_class, **config)[0]
        end = time.time()

        print(f"Result: {acc} | {end - start}")

        del data
        gc.collect()

        return {
            "fitness": -acc,
            "cost": end - start
        }
    
    dehb = DEHB(
        f=objective,
        cs=hpo_space,
        min_fidelity=1,
        max_fidelity=10,
        n_workers=1,
        output_path="./tmp"
    )
    trajectory, runtime, history = dehb.run(fevals=30)
