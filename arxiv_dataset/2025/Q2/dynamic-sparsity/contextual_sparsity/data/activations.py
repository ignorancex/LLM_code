# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class ActivationDictDataset(Dataset):
    """
    Wrapper class used to make a dataset out of the model activations.
    Each element is a dictionary {activation_id: activation_values}
    """

    def __init__(
        self,
        activations: Dict[str, Union[np.ndarray, torch.Tensor]],
        flatten: bool = False,
        **keys,
    ):
        super().__init__()
        self.activations = activations
        self.flatten = flatten

        for k, activation_name in keys.items():
            if activation_name not in activations:
                raise KeyError(f"Key {k} not found in activations")
        self.keys = keys
        self.batch_shape = self.activations[next(iter(self.keys.values()))].shape[:-1]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.flatten:
            multi_index = []

            for dim in self.batch_shape[::-1]:
                multi_index.append(index % dim)
                index //= dim

            index = multi_index[::-1]
        else:
            index = [index]

        return {k: self.activations[layer][tuple(index)] for k, layer in self.keys.items()}

    def __len__(self) -> int:
        if self.flatten:
            n_datapoints = np.prod(self.batch_shape)
        else:
            n_datapoints = self.batch_shape[0]
        return n_datapoints
