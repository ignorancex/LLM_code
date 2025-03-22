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

dataset = "svhn"
model = "resnet20"
exp_name = "train_8_baseline"
root = f"log/{dataset}/{model}/{exp_name}"
script = "train_baseline.py"
config_file = f"configs/{dataset}/{model}/train_baseline.yml"
configs.load(config_file, recursive=True)


def task_launcher(args):
    pres = ["python3", script, config_file]

    Design_name, experiment_name, checkpoint_dir, model_name, model_kernel_list, model_block_list, k, n_blocks, \
    teacher_name, teacher_checkpoint, file_path, id = args

    with open(
        os.path.join(root, f"{Design_name}_design_evaluation-PTC_size-{k}_run-{id}.log"),
        "w",
    ) as wfid:
        exp = [
            f"--run.experiment={experiment_name}",
            f"--run.random_state=42",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
            f"--model.name={model_name}",
            f"--model.kernel_list={model_kernel_list}",
            f"--model.block_list={model_block_list}",
            f"--super_layer.name=ps_dc_cr_adeptzero",
            f"--super_layer.arch.n_waveguides={k}",
            f"--super_layer.arch.n_front_share_waveguides={k}",
            f"--super_layer.arch.n_front_share_ops={k}",
            f"--super_layer.arch.n_blocks={n_blocks}",
            f"--teacher.name={teacher_name}",
            f"--teacher.checkpoint={teacher_checkpoint}",
            f"--gene.file_path={file_path}",
        ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [("MZI",
              f"{dataset}_{model}_train_MZI_8",
              f"{dataset}/{model}/train_8_MZI",
              "SuperResNet20",
              [6,16],
              [8,8,8,8,8,8],
              8,
              32,
              "",
              None,
              f"{dataset}/genes/MZI_gene.yaml",
              1
              ),

              ("Butterfly",
              f"{dataset}_{model}_train_butterfly_8",
              f"{dataset}/{model}/train_8_butterfly",
              "SuperResNet20",
              [6,16],
              [8,8,8,8,8,8],
              8,
              8,
              None,
              "",
              f"{dataset}/genes/butterfly_gene.yaml",
              1
              ),

              ("kxk_MMI",
              f"{dataset}_{model}_train_kxk_MMI_8",
              f"{dataset}/{model}/train_8_kxk_MMI",
              "SuperResNet20",
              [6,16],
              [8,8,8,8,8,8],
              8,
              8,
              None,
              "",
              f"{dataset}/genes/kxk_MMI_gene.yaml",
              1
              ),
              ]

    with Pool(3) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
