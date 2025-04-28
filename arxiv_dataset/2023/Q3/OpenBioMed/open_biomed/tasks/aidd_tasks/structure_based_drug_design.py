from typing import Optional

import pytorch_lightning as pl

from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

class StructureBasedDrugDesign(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def print_usage() -> str:
        return "\n".join([
            'Structure-based drug design.',
            'Inputs: {"pocket": a protein pocket}',
            "Outputs: A small molecule that is likely to bind with the pocket."
        ])

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("structure_based_drug_design", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("structure_based_drug_design", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return StructureBasedDrugDesignEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/vina_score",
            output_str="-{val_vina_score:.4f}",
            mode="min",
        )

class StructureBasedDrugDesignEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    # TODO: implement molecular propery eval and DockVina evaluation