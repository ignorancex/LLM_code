import argparse
import logging
import os
from pathlib import Path

import torch
import torch_geometric
from tqdm import tqdm

from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_config, setup_logging
from ocpmodels.datasets import LmdbDataset
from ocpmodels.trainers import EnergyTrainer, ForcesTrainer

setup_logging()

parser = argparse.ArgumentParser()

parser.add_argument('--cfg_root', default='configs_oe62', type=str, help='The configuration file root.')
parser.add_argument('--cfg', default='schnet_oe62_na.yml', type=str, required=True, help='The configuration file.')

args = parser.parse_args()



config_dir = args.cfg_root
#-----------Put your model variant here-----------
config_path = os.path.join(config_dir, args.cfg)
# schnet_oe62_baseline -- Params. 2753025
# schnet_oe62_ewald -- Params. 12206977
# schnet_oe62_na

torch.cuda.empty_cache()
conf = load_config(config_path)[0]
task = conf["fixed"]["task"]
model = conf["fixed"]["model"]
optimizer = conf["fixed"]["optimizer"]
name = conf["fixed"]["name"]
logger = conf["fixed"]["logger"]
dataset = conf["fixed"]["dataset"]
trainer = EnergyTrainer(
    task=task,
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    identifier=name,
    run_dir="./",
    is_debug=False,  # if True, do not save checkpoint, logs, or results
    print_every=5000,
    seed=0,  # random seed to use
    logger=logger,  # logger of choice (tensorboard and wandb supported)
    local_rank=0,
    amp=False,  # whether to use PyTorch Automatic Mixed Precision
)

trainer.train()

checkpoint_path = os.path.join(
    trainer.config["cmd"]["checkpoint_dir"], "best_checkpoint.pt"
)
trainer = EnergyTrainer(
    task=task,
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    identifier=name,
    run_dir="./",
    # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
    is_debug=True,  # if True, do not save checkpoint, logs, or results
    print_every=5000,
    seed=0,  # random seed to use
    logger=logger,  # logger of choice (tensorboard and wandb supported)
    local_rank=0,
    amp=False,  # use PyTorch Automatic Mixed Precision (faster training and less memory usage)
)
trainer.load_checkpoint(checkpoint_path=checkpoint_path)

metrics = trainer.validate(split="test")
results = {key: val["metric"] for key, val in metrics.items()}
print(f"Results for configuration {name}: {results}")



