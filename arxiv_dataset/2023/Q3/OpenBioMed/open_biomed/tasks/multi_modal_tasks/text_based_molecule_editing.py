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

class TextMoleculeEditing(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage() -> str:
        return "\n".join([
            'Text-based molecule editing.',
            'Inputs: {"molecule": a small molecule, "text": the desired property of the updated molecule}',
            "Outputs: A new molecule that is structurally similar to the original molecule but exhibit improved property described in text."
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("text_based_molecule_editing", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("text_based_molecule_editing", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return TextMoleculeEditingEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/ratio_improved",
            output_str="-validity_{val/validity:.4f}-ratio_improved_{val/ratio_improved:.4f}-accuracy_{val/accuracy:.4f}",
            mode="max",
        )

class TextMoleculeEditingEvaluationCallback(pl.Callback):
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
                out_labels = '\n'.join([str(x) for x in self.eval_dataset.labels[i]])
                logging.info(f"Original: {self.eval_dataset.molecules[i]}")
                logging.info(f"Prompt: {self.eval_dataset.texts[i]}")
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
        cnt_valid, cnt_improved, cnt_accurate = 0, 0, 0
        output_str = "\t".join(["Original", "Prompt", "Predicted", "Best_Reference", "All_Reference", "FP_Similarity", "FP_Similarity_orig"]) + "\n"
        for i in range(len(self.outputs)):
            best_sim, best_sim_orig = 0.0, 0.0
            best_ref = None

            orig_mol, prompt, label = self.eval_dataset.molecules[i], self.eval_dataset.texts[i], self.eval_dataset.labels[i]
            all_ref = []
            for ref_mol in label:
                cur_sim = molecule_fingerprint_similarity(self.outputs[i], ref_mol)
                if cur_sim > best_sim:
                    best_sim = cur_sim
                    best_ref = ref_mol
                all_ref.append(str(ref_mol))
                cur_sim = molecule_fingerprint_similarity(orig_mol, ref_mol)
                if cur_sim > best_sim_orig:
                    best_sim_orig = cur_sim
                
            output_str += "\t".join([str(orig_mol), str(prompt), str(self.outputs[i]), str(best_ref), ",".join(all_ref), f"{best_sim:.4f}", f"{best_sim_orig:.4f}"]) + "\n"
            if self.outputs[i].rdmol is not None:
                cnt_valid += 1
            if best_sim > best_sim_orig:
                cnt_improved += 1
            if check_identical_molecules(self.outputs[i], best_ref):
                cnt_accurate += 1

        output_path = os.path.join(trainer.default_root_dir, f"{self.status}_outputs", f"epoch{pl_module.current_epoch}")
        out_metrics = {
            f"{self.status}/validity": cnt_valid / len(self.outputs),
            f"{self.status}/ratio_improved": cnt_improved / len(self.outputs),
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