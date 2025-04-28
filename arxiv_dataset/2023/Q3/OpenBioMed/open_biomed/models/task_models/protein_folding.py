from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Tuple

import torch

from open_biomed.data import Protein
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import Collator, EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, EnsembleFeaturizer, Featurized
from open_biomed.utils.misc import sub_dict


class ProteinFoldingModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        featurizer, collator = self.featurizer_protein_folding()
        self.supported_tasks["protein_folding"] = {
            "forward_fn": self.forward_protein_folding,
            "predict_fn": self.predict_protein_folding,
            "featurizer": featurizer,
            "collator": collator,
        }
    
    @abstractmethod
    def forward_protein_folding(self, 
        protein: Featurized[Protein], 
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_protein_folding(self,
        protein: Featurized[Protein],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    @abstractmethod
    def featurizer_protein_folding(self) -> Tuple[Featurizer, Collator]:
        raise NotImplementedError