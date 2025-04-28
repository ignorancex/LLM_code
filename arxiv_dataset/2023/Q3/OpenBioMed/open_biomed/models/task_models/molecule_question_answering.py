from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch

from open_biomed.data import Molecule, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import EnsembleFeaturizer, Featurized
from open_biomed.utils.misc import sub_dict

class MoleculeQAModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        self.supported_tasks["molecule_question_answering"] = {
            "forward_fn": self.forward_molecule_question_answering,
            "predict_fn": self.predict_molecule_question_answering,
            "featurizer": EnsembleFeaturizer({
                **sub_dict(self.featurizers, ["molecule", "text"]),
                "label": self.featurizers["text"]
            }),
            "collator": EnsembleCollator({
                **sub_dict(self.collators, ["molecule", "text"]),
                "label": self.collators["text"]
            })
        }
    
    @abstractmethod
    def forward_molecule_question_answering(self,
        molecule: Featurized[Molecule],
        text: Featurized[Text], 
        label: Featurized[Text],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_molecule_question_answering(self,
        molecule: Featurized[Molecule],
        text: Featurized[Text], 
    ) -> List[Text]:
        raise NotImplementedError