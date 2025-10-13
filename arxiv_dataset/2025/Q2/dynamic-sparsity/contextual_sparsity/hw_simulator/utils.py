# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig

from contextual_sparsity.hw_simulator.constants import MODEL_ID_TO_DIMS
from contextual_sparsity.mask.hooks import MaskingHook

logger = logging.getLogger(__name__)


def convert_memory_unit(value, from_u, to_u):
    unit_enum = {
        "bit": 1.0 / 8.0,
        "B": 1.0,
        "KB": float(1e3),
        "MB": float(1e6),
        "GB": float(1e9),
    }
    # We use powers of 10 instead of powers of 2 for memory occupancy units
    return value * (unit_enum[from_u] / unit_enum[to_u])


def precision_to_bytes(precision: Union[DictConfig, Dict[str, int]]) -> Dict[str, float]:
    # Cast bit precision to bytes
    return {k: convert_memory_unit(v, "bit", "B") for k, v in precision.items()}


def calculate_footprint(
    layer_types: Union[Dict, list],
    precision: Dict,
    dimensions: Dict,
    seq_len: Optional[Union[int, float]] = None,
    verbose: Optional[bool] = False,
) -> float:
    """
    The function iterates over the specified layer types and calculates the memory footprint of each layer based on its
    precision and dimensions.
    It then adds up the memory footprints of all layers to get the total memory footprint.

    Args:
        layer_types (list): The layers of the model.
        precision (dict): A dictionary mapping layer names to their precision in bytes.
        dimensions (ModelStruct): A namedtuple mapping dimension names to their values.
        seq_len (float, int, optional): The sequence length. Required if 'kv_cache' is in layers. Defaults to None.
        verbose (bool, optional): If True, prints the memory footprint of each layer in GB. Defaults to False.

    Returns:
        float: The total memory footprint of the model in bytes.
    """
    total = 0
    gqa_ratio = (
        float(dimensions["num_key_value_heads"]) / float(dimensions["num_attention_heads"])
        if float(dimensions["num_attention_heads"]) > 0
        else 0.0
    )
    for layer in layer_types:
        current = precision[layer]  # Bytes per value (arithmetic precision)
        if layer in {"lm_head", "embedding"}:
            current *= dimensions["hidden_size"] * dimensions["vocab_size"]
        elif layer == "mlp":
            n_linears = 3.0 if dimensions["has_gate_proj"] else 2.0
            current *= (
                dimensions["hidden_size"]
                * dimensions["intermediate_size"]
                * n_linears
                * dimensions["num_hidden_layers"]
            )
        elif layer == "attention":
            n_linears = 2.0 + 2.0 * gqa_ratio
            current *= (
                dimensions["hidden_size"]
                * dimensions["hidden_size"]
                * n_linears
                * dimensions["num_hidden_layers"]
            )
        elif layer == "kv_cache":
            assert seq_len is not None
            current *= (
                seq_len
                * dimensions["hidden_size"]
                * 2.0
                * dimensions["num_hidden_layers"]
                * gqa_ratio
            )
        elif layer == "predictors":
            current *= dimensions["predictors"]
        else:
            raise ValueError(layer)

        if verbose:
            logger.info(f"\t\t{layer}: {convert_memory_unit(current, 'B', 'GB') :.3f} GB")

        total += current
    return total


def get_layer_type_from_layer_key(layer_key: str, model_id: Optional[str] = None) -> str:
    """
    Args:
        layer_key: (str) torch layer key. For example, observed layer keys for down projections in MLPs:
             - model.decoder.layers.0.fc2 (OPT)
             - model.layers.0.mlp.down_proj (Llama and Phi)

    Returns:
        (str): layer type in ('mlp', 'attention'), to identifies what type of layer is being sparsified
    """
    if model_id == "dummy":
        return "mlp"  # the dummy model only has an MLP at this point, but the layer names are not informative.
    if "fc" in layer_key or "mlp" in layer_key:
        return "mlp"
    else:
        raise ValueError(f"Unrecognized layer key {layer_key}!")


def get_dimensions_from_model(
    model_id: str,
    model: Optional[torch.nn.Module] = None,
    masking_hooks: Optional[List[MaskingHook]] = None,
) -> Dict[str, Union[int, bool]]:
    """
    This function tries to infer the model dimensions from a model config instance if the model is provided.
    If no model is provided, the dimensions are loaded from the default values described in the dictionary
    contextual_sparsity.hw_simulator.constants.MODEL_ID_TO_DIMS.
    If a specific key is not found in the model config, the default values is loaded from the same dictionary.

    Args:
        model_id: str, model identifier as specified in its config file.
        model: torch.nn.Module or None, an optional instance of the model from which to infer dimensions.
        masking_hooks: list of MaskingHook or None, an optional list of masking hooks from which to parse the
        dimensionality of the predictors.

    Returns:
        dict: mapping from model dimension keys to its value.
    """
    dimensions = MODEL_ID_TO_DIMS[model_id]
    dimensions["predictors"] = 0

    if model is None:
        logger.warning(
            "Model instance was not provided to compute model dimensions. "
            'Using default values stored in "contextual_sparsity.hw_simulator.constants.MODEL_ID_TO_DIMS"!'
        )
        return dimensions

    if not hasattr(model, "config"):
        logger.warning(
            "Model instance does not have a config attribute. "
            'Using default values stored in "contextual_sparsity.hw_simulator.constants.MODEL_ID_TO_DIMS"!'
        )
        return dimensions

    for k in dimensions.keys():
        if k in {"has_gate_proj", "predictors"}:
            continue

        if hasattr(model.config, k):
            dimensions[k] = getattr(model.config, k)
        else:
            logger.warning(
                f'Value for key "{k}" not found in model.config. Using default value stored in '
                f'"contextual_sparsity.hw_simulator.constants.MODEL_ID_TO_DIMS"!'
            )

    dimensions["predictors"] = sum(
        p.numel() for hook in masking_hooks for p in hook.masking_func.parameters()
    )

    return dimensions


def get_layer_key_to_hook_targets(
    model: Optional[torch.nn.Module] = None, masking_hooks: Optional[List[MaskingHook]] = None
) -> Dict[str, int]:
    """
    Args:
        masking_hooks: masking_hooks
        model: model

    Returns:
        Mapping from layer_key (taken from hook.mask_cols_of) to 'n_linears' it targets, and linear shapes
        (size of the dimension to be masked, and size of the features per masked element)
    """
    if masking_hooks is None:
        return dict()

    assert model is not None
    layer_key_to_hook_targets = dict()
    for hook in masking_hooks:
        layer_key = hook.mask_cols_of[0]
        n_linears = len(hook.mask_cols_of) + (
            0 if hook.mask_rows_of is None else len(hook.mask_rows_of)
        )
        module_shape = model.get_submodule(layer_key).weight.shape
        layer_key_to_hook_targets[layer_key] = {
            "n_linears": n_linears,
            "size_mask": module_shape[1],
            "size_per_idx": module_shape[0],
        }

    return layer_key_to_hook_targets


class HardwareClock:
    """
    A class used to represent a Hardware Clock to

    Attributes:
        _ttft (list): A list to store the time taken for prompt encoding.
        _time_per_token (list): A list to store the time taken per token generation.
        _cache_hit_rate (list): A list to store cache hitrates averaged across layer during token generation.
        _cache_hit_rate_per_layer (dict(layer_key -> list)): A dict to store cache hit rates per layer.
    """

    def __init__(self):
        self._ttft = []
        self._time_per_token = []
        self._cache_hit_rate = []
        self._cache_hit_rate_per_layer = defaultdict(list)

    def update(
        self,
        current_prompt_encoding: float,
        current_token_generation_fixed: float,
        current_token_generation_dynamic: float,
        current_cache_hit_rate: List[float],
        current_cache_hit_rates_per_layer: Dict[str, List[float]],
    ):
        self._ttft.append(current_prompt_encoding)
        token_generation_time = current_token_generation_fixed + current_token_generation_dynamic
        self._time_per_token.append(token_generation_time)
        self._cache_hit_rate.append(np.mean(current_cache_hit_rate))
        for layer_key, values in current_cache_hit_rates_per_layer.items():
            self._cache_hit_rate_per_layer[layer_key].append(np.mean(values))

    def get_ttft(self, return_mean: bool = False) -> float:
        ttft = np.array(self._ttft)
        if return_mean:
            return ttft.mean()
        return ttft

    def get_throughput(self, return_mean: bool = False) -> float:
        throughput = np.array(self._time_per_token) ** -1.0
        if return_mean:
            return throughput.mean()
        return throughput

    def get_cache_hit_rate(self, return_mean: bool = False) -> float:
        cache_hit_rate = np.array(self._cache_hit_rate)
        if return_mean:
            return cache_hit_rate.mean()
        return cache_hit_rate

    def get_cache_hit_rate_per_layer(self, return_mean: bool = False) -> float:
        cache_hit_rate_per_layer = copy.deepcopy(self._cache_hit_rate_per_layer)
        for layer_key in cache_hit_rate_per_layer.keys():
            cache_hit_rate_per_layer[layer_key] = np.array(cache_hit_rate_per_layer[layer_key])
            if return_mean:
                cache_hit_rate_per_layer[layer_key] = np.mean(cache_hit_rate_per_layer[layer_key])
        return cache_hit_rate_per_layer
