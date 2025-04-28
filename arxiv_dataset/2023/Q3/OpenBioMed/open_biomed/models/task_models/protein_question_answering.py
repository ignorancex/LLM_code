from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch

from open_biomed.data import Protein, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurized, EnsembleFeaturizer
from open_biomed.utils.misc import sub_dict

class ProteinQAModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        self.supported_tasks["protein_question_answering"] = {
            "forward_fn": self.forward_protein_question_answering,
            "predict_fn": self.predict_protein_question_answering,
            "featurizer": EnsembleFeaturizer({
                **sub_dict(self.featurizers, ["protein", "text"]),
                "label": self.featurizers["text"]
            }),
            "collator": EnsembleCollator({
                **sub_dict(self.collators, ["protein", "text"]),
                "label": self.collators["text"]
            })
        }
    
    @abstractmethod
    def forward_protein_question_answering(self, 
        protein: Featurized[Protein],
        text: Featurized[Text], 
        label: Featurized[Text],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_protein_question_answering(self,
        protein: Featurized[Protein],
        text: Featurized[Text], 
    ) -> List[Text]:
        raise NotImplementedError