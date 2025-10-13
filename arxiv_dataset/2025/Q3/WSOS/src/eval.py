from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
OmegaConf.register_new_resolver("eval", eval)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


import torch
import os

def fix_checkpoint_prefix(original_checkpoint_path):
    """
    Remove '_orig_mod' prefix from state dict keys in a checkpoint and save the fixed checkpoint.

    Args:
    original_checkpoint_path (str): Path to the original checkpoint file.

    Returns:
    str: Path to the fixed checkpoint file.
    """
    # Load the original checkpoint
    checkpoint = torch.load(original_checkpoint_path)
    # Get the state dict from the checkpoint
    state_dict = checkpoint['state_dict']

    # Remove the '_orig_mod.' prefix from the state dict keys
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value

    # Update the checkpoint with the new state dict
    checkpoint['state_dict'] = new_state_dict

    # Construct the fixed checkpoint path
    dir_name, base_name = os.path.split(original_checkpoint_path)
    fixed_base_name = f"fixed_{base_name}"
    fixed_checkpoint_path = os.path.join(dir_name, fixed_base_name)

    # Save the updated checkpoint
    torch.save(checkpoint, fixed_checkpoint_path)

    log.info(f"Updated checkpoint saved to {fixed_checkpoint_path}")
    return fixed_checkpoint_path


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    if len(logger) == 0:
        logger = [TensorBoardLogger(cfg.paths.get('output_dir'), cfg.task_name, log_graph=True)]
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.set_logger(logger)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
    cfg.ckpt_path = fix_checkpoint_prefix(cfg.ckpt_path)
    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
