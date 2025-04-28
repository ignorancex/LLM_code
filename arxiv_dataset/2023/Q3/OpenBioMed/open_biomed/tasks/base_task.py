# Three major responsibilities of a Task module:
# Corresponding Datasets
# TaskModel Wrapper: defining training and validation step, call appropreiate Model functions
# Callbacks: sampling / validation

from abc import ABC, abstractmethod, abstractstaticmethod
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from typing import Dict, Tuple, Optional
from typing_extensions import Any

import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from open_biomed.models import MODEL_REGISTRY
from open_biomed.datasets import DATASET_REGISTRY
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class ModelWrapper(pl.LightningModule, ABC):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_featurizer(self) -> Tuple[Featurizer, Collator]:
        raise NotImplementedError

class BaseTask(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractstaticmethod
    def print_usage():
        raise NotImplementedError

    @abstractstaticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        raise NotImplementedError

    @abstractstaticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> ModelWrapper:
        raise NotImplementedError

    @abstractstaticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        raise NotImplementedError

    @abstractstaticmethod
    def get_monitor_cfg() -> Struct:
        raise NotImplementedError

class DefaultDataModule(pl.LightningDataModule):
    def __init__(self, task: str,dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> None:
        super(DefaultDataModule, self).__init__()
        dataset = DATASET_REGISTRY[task][dataset_cfg.name](dataset_cfg, featurizer)
        train, valid, test = dataset.split()
        self.train_loader = DataLoader(
            dataset=train, 
            batch_size=dataset_cfg.batch_size_train, 
            shuffle=True, 
            num_workers=dataset_cfg.num_workers,
            collate_fn=collator
        )
        if valid is not None:
            self.valid_loader = DataLoader(
                dataset=valid,
                batch_size=dataset_cfg.batch_size_eval,
                shuffle=False,
                num_workers=dataset_cfg.num_workers,
                collate_fn=collator,
            )
        self.test_loader = DataLoader(
            dataset=test,
            batch_size=dataset_cfg.batch_size_eval,
            shuffle=False,
            num_workers=dataset_cfg.num_workers,
            collate_fn=collator,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.valid_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.test_loader

class DefaultModelWrapper(ModelWrapper):
    def __init__(self, task: str, model_cfg: Config, train_cfg: Config) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        if getattr(model_cfg, 'pretrained', ''):
            self.model = MODEL_REGISTRY[task][model_cfg.name].from_pretrained(model_cfg)
        else:
            self.model = MODEL_REGISTRY[task][model_cfg.name](model_cfg)
        self.model.configure_task(task)

    def get_featurizer(self) -> Tuple[Featurizer, Collator]:
        return self.model.get_featurizer()

    def forward(self, batch: Dict[str, Any]):
        return self.model(**batch)

    def predict(self, batch: Dict[str, Any]):
        return self.model.predict(**batch)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        loss_dict = self.model(**batch)
        loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, batch_size=self.train_cfg.batch_size, sync_dist=True, prog_bar=True, logger=True)
        return loss_dict["train/loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.model.predict(**batch)

    def test_step(self, batch: Dict[str, Any], batch_idx: int, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.model.predict(**batch)

    def configure_optimizers(self) -> Any:
        if self.train_cfg.optimizer.name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.train_cfg.optimizer.lr,
                weight_decay=self.train_cfg.optimizer.weight_decay,
                betas=(
                    getattr(self.train_cfg.optimizer, "beta1", 0.9), 
                    getattr(self.train_cfg.optimizer, "beta2", 0.999),
                )
            )
        else:
            raise NotImplementedError(f'Optimizer not supported:{self.train_cfg.optimizer.name}')

        if self.train_cfg.scheduler.name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.train_cfg.max_iters,
                eta_min=getattr(self.train_cfg.optimizer, "min_lr", self.train_cfg.optimizer.lr * 0.05),
            )
            self.scheduler = {
                "scheduler": scheduler,
                "interval": "step",
            }
        elif self.train_cfg.schedular == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=getattr(self.train_cfg.scheduler, "factor", 0.6),
                patience=getattr(self.train_cfg.scheduler, "patience", 10),
                min_lr=getattr(self.train_cfg.scheduler, "min_lr", self.train_cfg.optimizer.lr * 0.05)
            )
            self.scheduler = {
                "scheduler": scheduler,
                "monitor": self.train_cfg.monitor.name,
                "interval": "step",
                "frequency": getattr(self.cfg, "val_freq", 1000)
            }
        
        return {
            "optimizer": self.optimizer,
            "lr_schedular": self.scheduler,
        }