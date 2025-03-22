'''
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-19 15:03:45
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-12-17 22:25:44
'''
"""
Description: 
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 1969-12-31 18:00:00
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2023-05-30 00:13:47
"""
"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-22 02:32:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-06-24 04:05:00
"""
import os
import subprocess
from multiprocessing import Pool
import numpy as np

import mlflow
from pyutils.config import configs
from pyutils.general import ensure_dir, logger

dataset = "cifar10"
model = "vgg8"
exp_name = "pretrain"
root = f"log/{dataset}/{model}/{exp_name}"
script = "pretrain_MZI_16.py"
config_file = f"configs/{dataset}/{model}/pretrain.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]

    optimizer, lr, epoch, weight_decay, id = args

    with open(
        os.path.join(root, f"VGG_optimizer-{optimizer}_lr-{lr:.4f}_epoch-{epoch}_weight_decay-{weight_decay}_run-{id}.log"),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.name={optimizer}",
            f"--optimizer.lr={lr}",
            f"--run.n_epochs={epoch}",
            f"--optimizer.weight_decay={weight_decay}",
            f"--super_layer.arch.n_blocks=64",
            f"--super_layer.name=ps_dc_cr_adeptzero",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [("sgd", 0.02, 200, 0.0001, 4)]

    with Pool(1) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
