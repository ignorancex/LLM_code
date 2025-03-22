from torch.utils.data import Dataset
from typing import Sequence, TypeVar
T_co = TypeVar('T_co', covariant=True)

class SubsetWIdx(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]] + (idx,)

    def __len__(self):
        return len(self.indices)