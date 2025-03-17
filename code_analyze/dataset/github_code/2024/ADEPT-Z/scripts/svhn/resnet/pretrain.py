'''
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-19 15:03:45
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-12-17 22:56:22
'''
import os
import subprocess
from multiprocessing import Pool
import numpy as np

import mlflow
from pyutils.config import configs
from pyutils.general import ensure_dir, logger

dataset = "svhn"
model = "resnet20"
exp_name = "pretrain"
root = f"log/{dataset}/{model}/{exp_name}"
script = "pretrain.py"
config_file = f"configs/{dataset}/{model}/pretrain.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]

    optimizer, lr, epoch, id = args

    with open(
        os.path.join(root, f"ResNet20_optimizer-{optimizer}_lr-{lr:.4f}_epoch-{epoch}_run-{id}.log"),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.name={optimizer}",
            f"--optimizer.lr={lr}",
            f"--run.n_epochs={epoch}",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [("sgd", 0.02, 100, 1)]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
