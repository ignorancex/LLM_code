import os
import random
import time

from copy import deepcopy

from src.util import defect_file_dic, hp_space_to_configspace, load_defect_data, run_experiment

from ConfigSpace import Configuration
from dehb import DEHB


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
for filename in defect_file_dic:
    file = open(f'runs-{file_number}.txt', 'a')
    print(f'{filename}:', file=file, end=' ')

    data_orig = load_defect_data(filename)

    def objective(config: Configuration, fidelity: float, **_) -> dict:
        start = time.time()
        data = deepcopy(data_orig)
        results = run_experiment(data, 2, **config)
        f1 = results[0]
        end = time.time()

        print(f"{results} | {end - start}")

        return {
            "fitness": -f1,
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
