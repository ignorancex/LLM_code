
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningDataModule
from omegaconf import DictConfig, OmegaConf
import clip

from src.utils import utils, convert_32
from src.video_module import VideoModel
from src.datamodules.video_datamodule import VideoDataModule
from src.models.trn import RelationModuleMultiScale

# Initialize logger
log = utils.get_pylogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Instantiate the data module
    log.info(f"Instantiating datamodule <{cfg.experiment.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.experiment.datamodule)

    # Load the CLIP model (ResNet-50)
    log.info("Loading pretrained CLIP model")
    clip_model, preprocess = clip.load("RN50")
    for param in clip_model.parameters():
        param.requires_grad = False

    # Instantiate the TRN model with the CLIP visual encoder
    log.info("Instantiating TRN model with the CLIP visual encoder")
    trn_model = RelationModuleMultiScale(
        clip_model=clip_model,
        num_bottleneck=cfg.model.trn_model.num_bottleneck,
        num_frames=cfg.model.trn_model.num_frames,
        rand_relation_sample=False,
    )

    video_model = VideoModel(
        teacher_model=trn_model,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        extra_args={
            "dataset": cfg.experiment.extra.dataset,
            "train_file": cfg.experiment.datamodule.train_file,
            "classes_limit": cfg.experiment.extra.class_limit,
            "data_folder": cfg.experiment.extra.data_folder,
        }
    )

    # Convert model to 32-bit
    convert_32(video_model)

    # Load the checkpoint and update the model's state dict
    checkpoint_path = "/home/ubuntu/Amir/sfvda/trn/src/lightning_logs/version_p0k5a2w9/best_model.ckpt"  # Update this path
    log.info(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    video_model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Instantiate the trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=False,  # Disable logging for validation
        enable_checkpointing=False  # Disable checkpointing for validation
    )

    # Start validation
    log.info("Starting validation!")
    result = trainer.validate(model=video_model, datamodule=datamodule)

    log.info(f"Validation results: {result}")

if __name__ == "__main__":
    main()
