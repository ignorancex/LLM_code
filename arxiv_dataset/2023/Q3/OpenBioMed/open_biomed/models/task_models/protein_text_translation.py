from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_biomed.data import Protein, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.models.functional_model_registry import TEXT_ENCODER_REGISTRY, PROTEIN_DECODER_REGISTRY
from open_biomed.models.misc import MLP
from open_biomed.utils.collator import EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurized, EnsembleFeaturizer

class TextBasedProteinGenerationModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        self.supported_tasks["text_based_protein_generation"] = {
            "forward_fn": self.forward_text_based_protein_generation,
            "predict_fn": self.predict_text_based_protein_generation,
            "featurizer": EnsembleFeaturizer({
                "text": self.featurizers["text"],
                "label": self.featurizers["protein"],
            }),
            "collator": EnsembleCollator({
                "text": self.collators["text"],
                "label": self.collators["protein"]
            })
        }

    @abstractmethod
    def forward_text_based_protein_generation(self,
        text: Featurized[Text],
        label: Featurized[Protein],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_text_based_protein_generation(self,
        text: Featurized[Text]
    ) -> List[Protein]:
        raise NotImplementedError

class EnsembleTextBasedProteinGenerationModel(TextBasedProteinGenerationModel):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)
        self.text_config = Config(config_file=model_cfg.text.config_file).model
        self.text_model = TEXT_ENCODER_REGISTRY[self.text_config.name](self.text_config)
        if hasattr(model_cfg.text, "model_checkpoint"):
            self.text_model.load_from_checkpoint(model_cfg.text.model_checkpoint)
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.protein_config = Config(config_file=model_cfg.protein.config_file).model
        self.protein_model = PROTEIN_DECODER_REGISTRY[self.protein_config.name](self.protein_config)
        if hasattr(model_cfg.protein, "model_checkpoint"):
            self.protein_model.load_from_checkpoint(model_cfg.protein.model_checkpoint)
        if getattr(model_cfg.protein, "freeze", False):
            self.protein_model.freeze_parameters()
        self.connector = MLP(
            **model_cfg.connector.todict()
        )
        self.dropout = nn.Dropout(0.2)

        text_featurizer, text_collator = self.text_model.get_text_processor()
        protein_featurizer, protein_collator = self.protein_model.get_protein_processor()
        self.featurizers = {
            "text": text_featurizer,
            "protein": protein_featurizer,
        }
        self.collators = {
            "text": text_collator,
            "protein": protein_collator,
        }

        self._add_task()

    def forward_text_based_protein_generation(self, 
        text: Featurized[Text], 
        label: Featurized[Protein]
    ) -> Dict[str, torch.Tensor]:
        text_embeds = self.text_model.encode_text(text)
        text_embeds = self.dropout(self.connector(text_embeds))
        return self.protein_model.generate_loss(label, add_embeds=text_embeds[:, 0, :].unsqueeze(1), add_attention_mask=text.attention_mask[:, 0].unsqueeze(1))
        # return self.protein_model.generate_loss(label, add_embeds=text_embeds, add_attention_mask=text.attention_mask)

    def predict_text_based_protein_generation(self, 
        text: Featurized[Text]
    ) -> List[Protein]:
        text_embeds = self.text_model.encode_text(text)
        text_embeds = self.dropout(self.connector(text_embeds))
        return self.protein_model.generate_protein(add_embeds=text_embeds[:, 0, :].unsqueeze(1), add_attention_mask=text.attention_mask[:, 0].unsqueeze(1))
        # return self.protein_model.generate_protein(add_embeds=text_embeds, add_attention_mask=text.attention_mask)