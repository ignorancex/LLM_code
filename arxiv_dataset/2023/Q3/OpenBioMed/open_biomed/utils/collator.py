from abc import ABC, abstractmethod
from typing import Any, Dict, List
from torch_geometric.data import Data, Batch
import torch

from transformers import AutoTokenizer, DataCollatorWithPadding

class Collator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, inputs: List[Any]) -> Any:
        raise NotImplementedError

    def _collate_single(self, data):
        if isinstance(data[0], Data):
            return Batch.from_data_list(data)
        elif torch.is_tensor(data[0]):
            return torch.stack([x.squeeze() for x in data])
        elif isinstance(data[0], int):
            return torch.tensor(data).view((-1, 1))

class PygCollator(Collator):
    def __init__(self, follow_batch: List[str]=[], exclude_keys: List[str]=[]) -> None:
        super().__init__()
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, inputs: List[Data]) -> Batch:
        return Batch.from_data_list(inputs, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)

class ListCollator(Collator):
    def __call__(self, inputs: List[Any]) -> Any:
        return inputs

class ClassLabelCollator(Collator):
    def __call__(self, inputs: List[Any]) -> Any:
        batch = torch.stack(inputs)
        return batch


class DPCollator(Collator):
    def __init__(self):
        super(DPCollator, self).__init__()

    def __call__(self, mols):
        batch = self._collate_single(mols)
        return batch


class EnsembleCollator(Collator):
    def __init__(self, to_ensemble: Dict[str, Collator]) -> None:
        super().__init__()
        self.collators = {}
        for k, v in to_ensemble.items():
            self.collators[k] = v

    def __call__(self, inputs: List[Dict[str, Any]]) -> Dict[Any, Any]:
        collated = {}
        for k in inputs[0]:
            collated[k] = self.collators[k]([item[k] for item in inputs])
        return collated

    def get_attrs(self) -> List[str]:
        return list(self.collators.keys())