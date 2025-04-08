#!/usr/bin/env python3
#sbatch -p GPU hipp_AEs.sh

import os
import re
from argparse import ArgumentParser

import torch.nn as nn
from pytorch_lightning import seed_everything

import Models
from lightning import getModules
from utils import yaml


def train(config, fast_dev_run=False, accelerator="gpu", checkpoint=None):
    seed_everything(42, workers=True)
    trainer, modelTrainer, dataModule = getModules(config, fast_dev_run, accelerator)    
    trainer.fit(modelTrainer, dataModule, ckpt_path=checkpoint) # train the model


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--config',type=str, help='Config file full path')
    parser.add_argument('--fast-dev-run', default=False, action='store_true', help='if set, only run a single batch')
    parser.add_argument('--accelerator', type=str, default="gpu", help='accelerator to use')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    params = parser.parse_args()

    config_name = os.path.basename(params.config)
    config_name = re.sub(r'.ya?ml$', '', config_name)
    config = {'name': config_name, **yaml.load(params.config)}

    train(config, params.fast_dev_run, params.accelerator, params.checkpoint)
