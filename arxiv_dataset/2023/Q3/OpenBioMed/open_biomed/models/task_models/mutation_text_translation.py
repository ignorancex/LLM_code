from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch

from open_biomed.data import Protein, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized

class MutationExplanationModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        featurizer, collator = self.featurizer_mutation_explanation()
        self.supported_tasks["mutation_explanation"] = {
            "forward_fn": self.forward_mutation_explanation,
            "predict_fn": self.predict_mutation_explanation,
            "featurizer": featurizer,
            "collator": collator,
        }
    
    @abstractmethod
    def forward_mutation_explanation(self, 
        wild_type: Featurized[Protein],
        mutant: Featurized[Any], 
        label: Featurized[Text],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_mutation_explanation(self,
        wild_type: Featurized[Protein],
        mutant: Featurized[Any],
        **kwargs,
    ) -> List[Text]:
        raise NotImplementedError

    @abstractmethod
    def featurizer_mutation_explanation(self) -> Tuple[Featurizer, Collator]:
        raise NotImplementedError

class MutationEngineeringModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        featurizer, collator = self.featurizer_mutation_engineering()
        self.supported_tasks["mutation_engineering"] = {
            "forward_fn": self.forward_mutation_engineering,
            "predict_fn": self.predict_mutation_engineering,
            "featurizer": featurizer,
            "collator": collator,
        }
    
    @abstractmethod
    def forward_mutation_engineering(self, 
        wild_type: Featurized[Protein],
        prompt: Featurized[Text],
        label: Featurized[Any],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_mutation_engineering(self,
        wild_type: Featurized[Protein],
        prompt: Featurized[Text],
        position: Optional[torch.LongTensor]=None,
        **kwargs,
    ) -> List[List[Tuple[Protein, str]]]:
        raise NotImplementedError

    @abstractmethod
    def featurizer_mutation_engineering(self) -> Tuple[Featurizer, Collator]:
        raise NotImplementedError    