from typing import Optional

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models import AbstractTrainer
from src.utils import LOGGER, seed_everything

if len(LOGGER.handlers) > 1:
    LOGGER.handlers = []


def train(
    cfg: DictConfig,
):
    """
    Train the model.
    Parameters
    ----------
    cfg: DictConfig
        The configuration to use for training.
    """

    if cfg.get("seed"):
        LOGGER.info(f"Seed set to {cfg.seed}")
        seed_everything(cfg.seed)

    LOGGER.info(f"Instantiating datamoudle <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_loader = datamodule.get_dataloader("train")
    val_loader = datamodule.get_dataloader("val")

    LOGGER.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    LOGGER.info(f"Instantiating optimizer <{cfg.optimizer._target_}>")
    optimizer_partial = hydra.utils.instantiate(cfg.optimizer)
    optimizer = optimizer_partial(model.parameters())

    scheduler = None
    if cfg.get("scheduler"):
        LOGGER.info(f"Instantiating scheduler <{cfg.scheduler._target_}>")
        scheduler_partial = hydra.utils.instantiate(cfg.scheduler)
        scheduler = scheduler_partial(optimizer)

    LOGGER.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    triner_partial = hydra.utils.instantiate(cfg.trainer, _recursive_=True)
    trainer: AbstractTrainer = triner_partial(
        model=model, optimizer=optimizer, scheduler=scheduler
    )

    LOGGER.info("Starting training!")
    trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=cfg.base.epochs)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name=None,
)
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point for training.
    Parameters
    ----------
    cfg: DictConfig
        The configuration composed by Hydra.

    Returns
    -------
    Optional[float] with optimized metric value.
    """

    # train the model
    train(cfg)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    import gc

    gc.collect()
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    main()
