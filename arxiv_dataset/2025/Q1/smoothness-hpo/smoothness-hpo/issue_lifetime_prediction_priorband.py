import gc
import os
import random
import time

from copy import deepcopy

from src.util import load_issue_lifetime_prediction_data, run_experiment

import neps


hpo_space = {
    'n_units': neps.Integer(2, 6, default=4, default_confidence="low"),
    'n_layers': neps.Integer(2, 5, default=3, default_confidence="low"),
    'preprocessor': neps.Categorical(
        ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
        default="normalize",
        default_confidence="low"
    ),
    'wfo': neps.Categorical([False, True], default=False, default_confidence="low"),
    'smote': neps.Categorical([False, True], default=False, default_confidence="low"),
    'ultrasample': neps.Categorical([False, True], default=False, default_confidence="low"),
    'smooth': neps.Categorical([False, True], default=False, default_confidence="low"),
}


file_number = os.getenv('SLURM_JOB_ID') or random.randint(1, 10000)
file = open(f'runs-{file_number}.txt', 'a')
n_class = 3
for filename in ['chromium', 'firefox', 'eclipse']:
    print(f'Running: {filename}-{n_class}:', file=file)

    data_orig = load_issue_lifetime_prediction_data(filename, n_class)

    def objective(**config) -> dict:
        start = time.time()
        data = deepcopy(data_orig)
        acc = run_experiment(data, n_class, **config)[0]
        end = time.time()

        print(f"Result: {acc} | {end - start}", file=file)

        del data
        gc.collect()

        return {
            "loss": -acc,
            "info_dict": {
                "test_score": acc,
                "train_time": end - start
            }
        }

    neps.run(
        run_pipeline=objective,
        pipeline_space=hpo_space,
        root_directory=f"priorband_{filename}_{n_class}",
        max_evaluations_total=30
    )
