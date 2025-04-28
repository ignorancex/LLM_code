from typing import Dict, List, Tuple, Optional
from typing_extensions import Any

import json
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from open_biomed.data import molecule_fingerprint_similarity, check_identical_molecules
from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class TextGuidedMoleculeGeneration(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage() -> str:
        return "\n".join([
            'Text-based molecule generation.',
            'Inputs: {"text": textual descriptions of the desired molecule.}',
            "Outputs: A new molecule that best fits the textual instruction."
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("text_guided_molecule_generation", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("text_guided_molecule_generation", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return TextGuidedMoleculeGenerationEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/accuracy",
            output_str="-accuracy_{val/accuracy:.4f}-validity_{val/validity:.4f}",
            mode="max",
        )
    
class TextGuidedMoleculeGenerationEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = []
        self.eval_dataset = None

    def on_validation_batch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Optional[STEP_OUTPUT], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.outputs.extend(outputs)
        if batch_idx == 0:
            for i in range(2):
                out_labels = str(self.eval_dataset.labels[i])
                logging.info(f"Original: {self.eval_dataset.texts[i]}")
                logging.info(f"Predict: {self.outputs[i]}")
                logging.info(f"Ground Truth: {out_labels}")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.status = "val"
        self.outputs = []
        self.eval_dataset = trainer.val_dataloaders.dataset

    def on_validation_epoch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        gts = self.eval_dataset.labels
        
        cnt_valid, cnt_accurate = 0, 0
        output_str = ""
        for i in range(len(self.outputs)):
            text, label = self.eval_dataset.texts[i], self.eval_dataset.labels[i]
            cur_sim = molecule_fingerprint_similarity(self.outputs[i], gts[i])
                
            output_str += "\t".join([str(text), str(label), str(self.outputs[i]), f"{cur_sim:.4f}"]) + "\n"
            if self.outputs[i].rdmol is not None:
                cnt_valid += 1
            if check_identical_molecules(self.outputs[i], gts[i]):
                cnt_accurate += 1

        output_path = os.path.join(trainer.default_root_dir, f"{self.status}_outputs", f"epoch{pl_module.current_epoch}")
        out_metrics = {
            f"{self.status}/validity": cnt_valid / len(self.outputs),
            f"{self.status}/accuracy": cnt_accurate / len(self.outputs),
        }
        pl_module.log_dict(out_metrics)
        print(json.dumps(out_metrics, indent=4))
        json.dump(out_metrics, open(output_path + "_metrics.json", "w"))
        
        with open(output_path + "_outputs.txt", "w") as f:
            f.write(output_str)
        
        
    def on_test_batch_end(self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.status = "test"
        self.outputs = []
        self.eval_dataset = trainer.test_dataloaders.dataset

    def on_test_epoch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        self.on_validation_epoch_end(trainer, pl_module)