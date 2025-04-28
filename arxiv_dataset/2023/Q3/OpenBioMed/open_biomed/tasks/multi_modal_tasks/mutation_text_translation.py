from typing import Optional, Any

import logging
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.callbacks import TextOverlapEvalCallback
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class MutationExplanation(BaseTask):
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def print_usage() -> str:
        return "\n".join([
            'Mutation explanation.',
            'Inputs: {"protein": a protein (the amino acid sequence is required). "mutation": a string that describes a single-point substitution mutation. (e.g. A51F indicates that the 51st amino acid changes from A to F)}',
            "Outputs: Textual descriptions of how this mutation affects protein function and further biological significance."
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("mutation_explanation", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("mutation_explanation", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return MutationExplanationEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/ROUGE-1",
            output_str="-BLUE-2_{val/BLUE-2:.4f}-BLUE-4_{val/BLUE-4:.4f}-BLUE-L_{val/BLUE-L:.4f}-ROUGE-1_{val/ROUGE-1:.4f}-ROUGE-2_{val/ROUGE-2:.4f}",
            mode="max",
        )

class MutationExplanationEvaluationCallback(TextOverlapEvalCallback):
    def __init__(self) -> None:
        super().__init__(tokenizer_type=None)

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
                logging.info(f"Protein: {self.eval_dataset.protein[i]}")
                logging.info(f"Predict: {self.outputs[i]}")
                logging.info(f"Ground Truth: {out_labels}")

class MutationEngineering(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage() -> str:
        return "\n".join([
            'Mutation engineering.',
            'Inputs: {"protein": a protein (the amino acid sequence is required). "prompt": A textual prompt indicating the desired property of the mutant.}',
            "Outputs: A string that describes a single-point substitution mutation. (e.g. A51F indicates that the 51st amino acid changes from A to F)."
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("mutation_engineering", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("mutation_engineering", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return MutationEngineeringEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/recall@50",
            output_str="-Recall@50_{val/recall@50:.4f}-Accuracy_{val/accuracy:.4f}",
            mode="max",
        )

# TODO: implement mutation engineering
class MutationEngineeringEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()