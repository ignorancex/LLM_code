'''
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-19 15:03:45
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-10-24 02:39:58
'''
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.config import configs
from pyutils.general import ensure_dir, logger

dataset = "mnist"
model = "cnn"
exp_name = "search_8_multiobj"
root = f"log/{dataset}/{model}/{exp_name}"
script = "search_multiobj.py"
config_file = f"configs/{dataset}/{model}/search_8.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]

    n_iteration, population_size, batch_size, train_ratio, valid_ratio, \
    area_lower_bound, area_upper_bound, power_lower_bound, power_upper_bound, \
    phase_noise_std, sigma_noise_std, dc_noise_std, cr_tr_noise_std, cr_phase_noise_std, \
    checkpoint_dir, id = args

    with open(
        os.path.join(root, f"Multiobj-area_bound-{area_lower_bound}-{area_upper_bound}_noisy_run-{id}.log"),
        "w",
    ) as wfid:
        exp = [
            f"--evo_search.n_iterations={n_iteration}",
            f"--evo_search.population_size={population_size}",
            f"--run.batch_size={batch_size}",
            f"--dataset.train_valid_split_ratio=[{train_ratio},{valid_ratio}]",
            f"--super_layer.arch.device_cost.area_lower_bound={area_lower_bound}",
            f"--super_layer.arch.device_cost.area_upper_bound={area_upper_bound}",
            f"--evo_search.constr.area=[{area_lower_bound},{area_upper_bound}]",
            f"--evo_search.constr.power=[{power_lower_bound},{power_upper_bound}]",
            f"--run.random_state=42",
            f"--checkpoint.model_comment=adeptzero_bound-{area_upper_bound}-valid_ratio-{valid_ratio}-run-{id}",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
            f"--evo_search.robustness.phase_noise_std={phase_noise_std}",
            f"--evo_search.robustness.sigma_noise_std={sigma_noise_std}",
            f"--evo_search.robustness.dc_noise_std={dc_noise_std}",
            f"--evo_search.robustness.cr_tr_noise_std={cr_tr_noise_std}",
            f"--evo_search.robustness.cr_phase_noise_std={cr_phase_noise_std}",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)

if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (80, 100, 256, 0.90, 0.10, 3114853.47, 1000000000, 10, 10000, 0.02, 0, 0.01, 0.01, 1, "mnist/cnn/search_results_rf", 1),
    ]
        
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
