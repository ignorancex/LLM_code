from typing import Tuple
from typing_extensions import Self
from abc import ABC, abstractmethod, abstractclassmethod

import torch
import torch.nn as nn

from open_biomed.utils.config import Config
from open_biomed.utils.collator import Collator
from open_biomed.utils.featurizer import Featurizer

class BaseModel(nn.Module):
    def __init__(self, model_cfg: Config) -> None:
        super(BaseModel, self).__init__()
        self.config = model_cfg
        self.supported_tasks = {}
        self.featurizers = {}
        self.collators = {}

    def get_featurizer(self) -> Tuple[Featurizer, Collator]:
        return self.featurizer, self.collator

    def load_from_checkpoint(self, checkpoint_file: str) -> None:
        # Load model parameters from a saved checkpoint with OpenBioMed format
        # NOTE: This method should not be overrided. To load the checkpoint provided by other works, please use the from_pretrained method
        state_dict = torch.load(open(checkpoint_file, "rb"), map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if hasattr(self, "load_ckpt"):
            self.load_ckpt(state_dict)
        self.load_state_dict(state_dict, strict=True)

    def configure_task(self, task: str) -> None:
        # Forward function, producing loss for training
        self.forward = self.supported_tasks[task]["forward_fn"]
        # Predict function, for evaluation and inference
        self.predict = self.supported_tasks[task]["predict_fn"]
        # Featurizer, for transforming input data into tensor
        self.featurizer = self.supported_tasks[task]["featurizer"]
        # Collator, for organizing multiple data into batch
        self.collator = self.supported_tasks[task]["collator"]

class PretrainedModel(BaseModel):
    def __init__(self, model_cfg: Config) -> None:
        super(PretrainedModel, self).__init__(model_cfg)

    @abstractclassmethod
    def from_pretrained(self, model_cfg: Config) -> Self:
        # Load model configurations and parameters from a pre-trained checkpoint (model_cfg.pretrained)
        raise NotImplementedError