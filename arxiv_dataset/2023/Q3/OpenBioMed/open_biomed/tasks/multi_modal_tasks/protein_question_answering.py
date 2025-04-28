from typing import Dict, List, Tuple, Optional
from typing_extensions import Any

import json
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.callbacks import TextOverlapEvalCallback
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class ProteinQA(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage() -> str:
        return "\n".join([
            'Protein question answering.',
            'Inputs: {"protein": a protein you are interested in. "text": a question about the protein.}',
            "Outputs: An answer to the question."
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("protein_question_answering", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("protein_question_answering", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return ProteinQAEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/ROUGE-1",
            output_str="-BLUE-2_{val/BLEU-2:.4f}-BLUE-4_{val/BLEU-4:.4f}-ROUGE-1_{val/ROUGE-1:.4f}-ROUGE-2_{val/ROUGE-2:.4f}",
            mode="max",
        )

class ProteinQAEvaluationCallback(TextOverlapEvalCallback):
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
        if batch_idx == 0:
            for i in range(2):
                out_labels = str(self.eval_dataset.labels[i])
                logging.info(f"Question: {self.eval_dataset.texts[i]}")
                logging.info(f"Protein: {self.eval_dataset.proteins[i]}")
                logging.info(f"Predict: {self.outputs[i]}")
                logging.info(f"Ground Truth: {out_labels}")