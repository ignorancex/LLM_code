# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import inspect
import logging
import os
import torchvision
import time
import wandb
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from solo.args.linear import parse_cfg
from solo.data.classification_dataloader_kfold import prepare_data
from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous

import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

try:
    from solo.data.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_avaliable = False

@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    K = cfg.K_fold

    print(f"K-fold cross validation: {K}")

    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]

    # initialize backbone
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    if cfg.backbone.name.startswith("resnet"):
        # remove fc layer
        backbone.fc = nn.Identity()

    ckpt_path = cfg.pretrained_feature_extractor
    assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")

    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
            logging.warn(
                "You are using an older checkpoint. Use a new one as some issues might arrise."
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    


    targ_dir = cfg.data.train_path
    # To remove, or parametrize
    # Alert! questo viene sovrascritto in dataset_preparation in classification_dataloader_kfold
    """
    if cfg.data.dataset == "robin":
        #datalist = "info_full_downsampled_min.json"
        datalist = "info_full_balanced_1650_down_and_oversampling.json"
    elif cfg.data.dataset == "rgz":
        datalist = "info_full_downsampled_min_wo_nan.json"
    """
    # "info_full_downsampled_min.json"

    info_file = os.path.join(targ_dir, cfg.data.datalist)
    info_pd = pd.read_json(info_file, orient="index")

    print(info_pd.describe())

    target = np.array(info_pd["source_type"])

    accuracy_values = []
    columns=["accuracy_values"]
    summary_table = wandb.Table(columns= columns, data=[])
    for _ in range(cfg.repetitions):
        for k in range(K):
            train_idxs = np.load(os.path.join(targ_dir, f"info_full_balanced_{cfg.data.sample_size}_down_and_oversampling_train_K{K}_{k}.npy"))
            val_idxs = np.load(os.path.join(targ_dir, f"info_full_balanced_{cfg.data.sample_size}_down_and_oversampling_val_K{K}_{k}.npy"))

            backbone.load_state_dict(state, strict=False)
            logging.info(f"Loaded {ckpt_path}")

            # check if mixup or cutmix is enabled
            if cfg.label_smoothing > 0:
                loss_func = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)
            else:
                loss_func = torch.nn.CrossEntropyLoss()

            model = LinearModel(backbone, loss_func=loss_func, mixup_func=None, cfg=cfg)
            make_contiguous(model)

            
            # can provide up to ~20% speed up
            if not cfg.performance.disable_channel_last:
                model = model.to(memory_format=torch.channels_last)

            train_loader, val_loader = prepare_data(
                dataset=cfg.data.dataset,
                data_dict=info_pd,
                balancing_strategy=cfg.data.balancing_strategy,
                batch_size=cfg.optimizer.batch_size,
                num_workers=cfg.data.num_workers,
                cfg = cfg,
                train_idxs = train_idxs,
                val_idxs = val_idxs,
                subsampler=False
            )

            ckpt_path, wandb_run_id = None, None
            if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
                auto_resumer = AutoResumer(
                    checkpoint_dir=os.path.join(cfg.checkpoint.dir, "linear"),
                    max_hours=cfg.auto_resume.max_hours,
                )
                resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
                if resume_from_checkpoint is not None:
                    print(
                        "Resuming from previous checkpoint that matches specifications:",
                        f"'{resume_from_checkpoint}'",
                    )
                    ckpt_path = resume_from_checkpoint
            elif cfg.resume_from_checkpoint is not None:
                ckpt_path = cfg.resume_from_checkpoint
                del cfg.resume_from_checkpoint

            callbacks = []

            if cfg.checkpoint.enabled:
                # save checkpoint on last epoch only
                ckpt = Checkpointer(
                    cfg,
                    logdir=os.path.join(cfg.checkpoint.dir, "linear"),
                    frequency=cfg.checkpoint.frequency,
                    keep_prev=cfg.checkpoint.keep_prev,
                )
                callbacks.append(ckpt)

            # wandb logging
            if cfg.wandb.enabled:
                wandb_logger = WandbLogger(
                    name=cfg.name,
                    project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    offline=cfg.wandb.offline,
                    resume="allow" if wandb_run_id else None,
                    id=wandb_run_id,
                )
                #wandb_logger.watch(model, log="gradients", log_freq=100)
                wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

                # lr logging
                lr_monitor = LearningRateMonitor(logging_interval="epoch") # epoch?
                callbacks.append(lr_monitor)

            trainer_kwargs = OmegaConf.to_container(cfg)
            # we only want to pass in valid Trainer args, the rest may be user specific
            valid_kwargs = inspect.signature(Trainer.__init__).parameters
            trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
            trainer_kwargs.update(
                {
                    "logger": wandb_logger if cfg.wandb.enabled else None,
                    "callbacks": callbacks,
                    "enable_checkpointing": False,
                    "strategy": DDPStrategy(find_unused_parameters=False)
                    if cfg.strategy == "ddp"
                    else cfg.strategy,
                }
            )

            trainer = Trainer(**trainer_kwargs)

            print(f"Mini-batches train: {len(train_loader)}")
            print(f"Mini-batches val: {len(val_loader)}")
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
            accuracy_values.append(trainer.logged_metrics["val_acc-top1"].item())
            

            summary_table.add_row(trainer.logged_metrics["val_acc-top1"].item())
            #
            #table_data = {'mean_accuracy': mean_accuracy, 'std_accuracy': std_accuracy}
    
    #table_data = accuracy_values

    mean_accuracy = np.mean(accuracy_values)
    std_accuracy = np.std(accuracy_values)

    mean_std_table = wandb.Table(columns= ["mean", "std"], data=[])
    mean_std_table.add_row(mean_accuracy, std_accuracy)
    wandb.log({'mean_std_table': mean_std_table})
    wandb.log({'accuracies_table': summary_table})


if __name__ == "__main__":
    main()
