from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch

from open_biomed.data import Molecule, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import EnsembleFeaturizer
from open_biomed.utils.misc import sub_dict

class TextGuidedMoleculeGenerationModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        self.supported_tasks["text_guided_molecule_generation"] = {
            "forward_fn": self.forward_text_guided_molecule_generation,
            "predict_fn": self.predict_text_guided_molecule_generation,
            "featurizer": EnsembleFeaturizer({
                **sub_dict(self.featurizers, ["text"]),
                "label": self.featurizers["molecule"]
            }),
            "collator": EnsembleCollator({
                **sub_dict(self.collators, ["text"]),
                "label": self.collators["molecule"]
            })
        }
    
    @abstractmethod
    def forward_text_guided_molecule_generation(self, 
        text: List[Text], 
        label: List[Molecule],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def predict_text_guided_molecule_generation(self,
        text: List[Text], 
    ) -> List[Molecule]:
        raise NotImplementedError