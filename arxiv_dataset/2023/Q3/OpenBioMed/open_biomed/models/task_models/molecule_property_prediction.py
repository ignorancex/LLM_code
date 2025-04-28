from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import torch

from open_biomed.data import Molecule, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import EnsembleFeaturizer
from open_biomed.utils.misc import sub_dict


def get_num_task(dataset):
    dataset = dataset.lower()
    """ used in molecule_finetune.py """
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'donor']:
        return 1
    elif dataset == 'pcba':
        return 92
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    raise ValueError('Invalid dataset name.')


class MoleculePropertyPredictionModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)
        self.num_tasks = get_num_task(model_cfg.dataset_name)

    def _add_task(self) -> None:
        self.supported_tasks["molecule_property_prediction"] = {
            "forward_fn": self.forward_molecule_property_prediction,
            "predict_fn": self.predict_molecule_property_prediction,
            # TODO: 这里label的featurize和collator如何定义呢
            "featurizer": EnsembleFeaturizer({
                **sub_dict(self.featurizers, ["molecule"]),
                "label": self.featurizers["classlabel"]
            }),
            "collator": EnsembleCollator({
                **sub_dict(self.collators, ["molecule"]),
                "label": self.collators["classlabel"]
            })
        }
    
    # TODO: Add type hints to the parameters.
    @abstractmethod
    def forward_molecule_property_prediction(self, 
        molecule,
        label
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_molecule_property_prediction(self,
        molecule
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError