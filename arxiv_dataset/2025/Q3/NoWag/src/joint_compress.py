import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import numpy as np
import os
from typing import Tuple, Optional, Union, List
from src.compression_parent import CompressedLinear
from src.quantize_compress import LinearVQ, LinearVQ_Halving
from src.sparse_compress import SparseLinear
from src.utils.normalizer import Normalizer


class JointLinear(CompressedLinear):
    """Jointly compresses the weight of a linear layer with multiple compression modules"""

    name = "JointLinear"

    def compress(
        self,
        compression_kwargs: dict[str, dict],
        shared_normalizer: bool = False,
        normalizer_kwargs: dict = {},
    ):
        """
        Compress the model with multiple compression modules
        Args:
            compression_kwargs: dictionary containing the compression module names and their respective kwargs
        """

        modules = []
        weight = self.weight
        if shared_normalizer:
            normalizer, _ = Normalizer.normalize_init(weight, **normalizer_kwargs)
            self.normalizer_bits = normalizer.get_n_bits()
        else:
            normalizer = None
            self.normalizer_bits = 0
        for module_name, kwargs in compression_kwargs.items():
            if module_name == "LinearVQ":
                new_module = LinearVQ(weight, None, False)
            elif module_name == "LinearVQ_Halving":
                new_module = LinearVQ_Halving(weight, None, False)
            elif module_name == "SparseLinear":
                new_module = SparseLinear(weight, None, False)
            else:
                raise ValueError(f"Unknown compression module {module_name}")
            new_module.compress(normalizer=normalizer, **kwargs)
            modules.append(new_module)
            weight = weight - new_module.reconstruct()

        self.modules = nn.ModuleList(modules)

    def _no_checkpoint_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        Args:
            x: input tensor
        Returns:
            output tensor
        """
        y = self.modules[0](x)
        for module in self.modules[1:]:
            y += module(x)

        if self.bias is not None:
            y += self.bias
        return y

    def reconstruct_(self, **kwargs):
        """
        Reconstruct the compressed weight
        """
        weight = self.modules[0].reconstruct()
        for module in self.modules[1:]:
            weight += module.reconstruct()
        return weight

    def blank_recreate(
        self,
        compression_kwargs: dict[str, dict],
        shared_normalizer: bool = False,
        normalizer_kwargs: dict = {},
    ):
        """recreates the compressed model with that of the same structure as the original model
        allows for loading of the model from a checkpoint"""
        if shared_normalizer:
            normalizer, _ = Normalizer.blank_recreate(weight, **normalizer_kwargs)
            self.normalizer_bits = normalizer.get_n_bits()
        else:
            normalizer = None
            self.normalizer_bits = 0

        weight = self.original_weight
        modules = []
        for module_name, kwargs in compression_kwargs.items():
            if module_name == "LinearVQ":
                new_module = LinearVQ(weight, None, False)
            elif module_name == "LinearVQ_Halving":
                new_module = LinearVQ_Halving(weight, None, False)
            elif module_name == "SparseLinear":
                new_module = SparseLinear(weight, None, False)
            else:
                raise ValueError(f"Unknown compression module {module_name}")
            new_module.blank_recreate(normalizer=normalizer, **kwargs)
            modules.append(new_module)
            weight = weight - new_module.reconstruct()
        self.modules = nn.ModuleList(modules)

    def get_n_bits(self):
        n_bits = 0
        for module in self.modules:
            n_bits += module.get_n_bits()
        n_bits -= self.normalizer_bits * (len(self.modules) - 1)
        return n_bits

    def change_denormalization_method(self, new_method):
        for module in self.modules:
            module.change_denormalization_method(new_method)

    def change_forward_method(self, new_method):
        for module in self.modules:
            module.change_forward_method(new_method)
