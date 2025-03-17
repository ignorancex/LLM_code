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
exp_name = "retrain_solutions_16"
root = f"log/{dataset}/{model}/{exp_name}"
script = "retrain_solutions.py"
config_file = f"configs/{dataset}/{model}/train_baseline.yml"
configs.load(config_file, recursive=True)

def task_launcher(args):
    pres = ["python3", script, config_file]

    optimizer, lr, experiment_name, n_epochs, random_state,\
    checkpoint_dir, save_best_model_k, model_name, kernel_list, block_list, \
    super_layer_name, n_waveguides, n_front_share_waveguides, n_front_share_ops, n_blocks, \
    area_lower_bound, area_upper_bound, power_lower_bound, power_upper_bound, \
    phase_noise_std, sigma_noise_std, dc_noise_std, cr_tr_noise_std, cr_phase_noise_std, id = args

    with open(
        os.path.join(root, f"Train_sampled_solutions-PTC_size-{n_waveguides}_run-{id}.log"),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.name={optimizer}",
            f"--optimizer.lr={lr}",
            f"--run.experiment={experiment_name}",
            f"--run.n_epochs={n_epochs}",
            f"--run.random_state={random_state}",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
            f"--checkpoint.save_best_model_k={save_best_model_k}",
            f"--model.name={model_name}",
            f"--model.kernel_list={kernel_list}",
            f"--model.block_list={block_list}",
            f"--super_layer.name={super_layer_name}",
            f"--super_layer.arch.n_waveguides={n_waveguides}",
            f"--super_layer.arch.n_front_share_waveguides={n_front_share_waveguides}",
            f"--super_layer.arch.n_front_share_ops={n_front_share_ops}",
            f"--super_layer.arch.n_blocks={n_blocks}",
            f"--super_layer.arch.device_cost.area_lower_bound={area_lower_bound}",
            f"--super_layer.arch.device_cost.area_upper_bound={area_upper_bound}",
            f"--evo_search.constr.area=[{area_lower_bound},{area_upper_bound}]",
            f"--evo_search.constr.power=[{power_lower_bound},{power_upper_bound}]",
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
        (
        "sgd", # optimizer
        0.02, # learning rate
        "mnist_cnn_retrain_sampled_solutions_16", # experiment name
        30, # number of training epochs
        42, # random state
        "mnist/cnn/retrain_sampled_solutions_16/set8", # checkpoint directory
        1, # number of best models saved
        "SuperOCNN", # model name
        [32,32], # model kernel list
        [16,16,16], # model block list
        "ps_dc_cr_adeptzero", # super layer name
        16, # number of waveguides in one PS array
        16, # n_front_share_waveguides
        16, # n_front_share_ops
        64, # number of blocks in the solution
        1000, # lower bound of area
        1000000000, # upper bound of area
        10, # lower bound of power
        100000, # upper bound of power
        0.02, # phase noise std
        0, # sigma noise std
        0.01, # dc noise std
        0.01, # cr_tr noise std
        1, # cr_phase noise std
        8 # run id
        )
    ]
    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")