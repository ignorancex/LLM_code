

import sys
import os
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from copy import deepcopy
import torch.nn as nn

import sys
sys.path.append(root)  # Ensure that the root is added to sys.path
import hydra
import lovely_tensors as lt
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningDataModule, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

from omegaconf import DictConfig, OmegaConf
import clip
from src.utils import utils, convert_32
from src.video_module import VideoModel
from src.datamodules.video_datamodule import VideoDataModule
from src.utils.best_val_accuracy_callback import BestValAccuracyCallback


from src.models.trn import RelationModuleMultiScale  


log = utils.get_pylogger(__name__)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    wandb.init(project="M_H", name=cfg.run_name, settings=wandb.Settings(_disable_stats=True))

  
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

   
    if cfg.get("lovely_tensors"):
        lt.monkey_patch()

   
    log.info(f"Instantiating datamodule <{cfg.experiment.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.experiment.datamodule)


    log.info("Loading CLIP model (ViT-L/14 for training/student model)...")
    clip_model_train, preprocess = clip.load("ViT-L/14")
    for param in clip_model_train.parameters():
        param.requires_grad = False  

  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_train = clip_model_train.to(device)

    log.info("Loading CLIP model (ViT-L/14) for zero-shot learning...")
    clip_model_zeroshot = clip_model_train  # Use the same model for zero-shot predictions

    log.info("Instantiating TRN model with the CLIP ViT-L/14 visual encoder (for training)")
    trn_model = RelationModuleMultiScale(
        clip_model=clip_model_train,
        num_bottleneck=cfg.model.trn_model.num_bottleneck,
        num_frames=cfg.model.trn_model.num_frames,
        rand_relation_sample=False,
    ).to(device)

    trn_model.classifier = nn.Linear(256, cfg.model.num_classes).to(device)

   # checkpoint_path = "/home/amir/My_DA/trn-2080_no_tacher_student-vit14/src/lightning_logs/mit_hmdb51/best_model.ckpt"  # Update this path
    checkpoint_path ="/home/amir/My_DA/trn-2080_no_tacher_student-vit14/src/lightning_logs/version_hpyt3x18/best_model_epoch_17.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    # Load the model checkpoint
    state_dict = {k.replace("teacher_model.", ""): v for k, v in checkpoint['state_dict'].items() if "classifier" not in k}
    trn_model.load_state_dict(state_dict, strict=False)

    # Load the classifier weights
    if 'classifier.weight' in checkpoint['state_dict'] and 'classifier.bias' in checkpoint['state_dict']:
        trn_model.classifier.weight.data = checkpoint['state_dict']['classifier.weight']
        trn_model.classifier.bias.data = checkpoint['state_dict']['classifier.bias']
        log.info("Classifier loaded from checkpoint.")
    else:
        log.info("Classifier weights not found in checkpoint, initializing randomly.")

    log.info("Creating student model for prediction")
    student_model = deepcopy(trn_model)


    video_model = VideoModel(
        student_model=student_model,
        teacher_model=trn_model,
        clip_model=clip_model_zeroshot,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.model.learning_rate,
        temperature=cfg.extra.temperature,
        scale=cfg.extra.scale,
        prob_invert=cfg.extra.prob_invert,
        temperature_clip_reliability=cfg.extra.temperature_clip_reliability,
        temperature_clip_collaborative=cfg.extra.temperature_clip_collaborative,
        ema_decay=cfg.extra.ema_decay,
        scr=cfg.extra.scr, 
        min_weight=cfg.extra.min_weight, 
        max_weight=cfg.extra.max_weight,
        mu_temp=cfg.extra.mu_temp,
        beta_temp=cfg.extra.beta_temp,
        imp_fac=cfg.extra.imp_fac,    
        extra_args={
            "dataset": cfg.experiment.extra.dataset,
            "train_file": cfg.experiment.datamodule.train_file,
            "classes_limit": cfg.experiment.extra.class_limit,
            "data_folder": cfg.experiment.extra.data_folder
        },
    )

    convert_32(video_model)

    # Instantiate callbacks
    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    best_val_accuracy_callback = BestValAccuracyCallback()
    callbacks.append(best_val_accuracy_callback)

    # Instantiate loggers
    log.info("Instantiating loggers...")
    wandb_logger = WandbLogger(log_model=True)

    # Instantiate the trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=wandb_logger
    )

    # Start training
    log.info("Starting training!")
    trainer.fit(model=video_model, datamodule=datamodule)



if __name__ == "__main__":
    main()