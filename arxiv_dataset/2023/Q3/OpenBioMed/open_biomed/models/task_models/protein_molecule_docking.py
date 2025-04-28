from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch

from open_biomed.data import Molecule, Pocket
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import EnsembleFeaturizer, Featurized
from open_biomed.utils.misc import sub_dict

class PocketMolDockModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        self.supported_tasks["pocket_molecule_docking"] = {
            "forward_fn": self.forward_pocket_molecule_docking,
            "predict_fn": self.predict_pocket_molecule_docking,
            "featurizer": EnsembleFeaturizer({
                **sub_dict(self.featurizers, ["molecule", "pocket"]),
                "label": self.featurizers["molecule"]
            }),
            "collator": EnsembleCollator({
                **sub_dict(self.collators, ["molecule", "pocket"]),
                "label": self.collators["molecule"]
            })
        }
    
    @abstractmethod
    def forward_pocket_molecule_docking(self,
        molecule: Featurized[Molecule], 
        pocket: Featurized[Pocket], 
        label: Featurized[Molecule],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_pocket_molecule_docking(self,
        molecule: Featurized[Molecule],
        pocket: Featurized[Pocket], 
    ) -> List[Molecule]:
        raise NotImplementedError

#TODO: implement blind docking
class BlindMolDockModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)