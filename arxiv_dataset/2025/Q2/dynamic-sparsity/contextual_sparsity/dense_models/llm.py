# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from typing import Optional, Type, Union

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from contextual_sparsity.utils.layer_names import LAYERS_CONTAINER, MODEL_MAPS, N_LAYERS
from contextual_sparsity.utils.misc import parse_dtype
from contextual_sparsity.utils.phi import split_upgate
from contextual_sparsity.utils.sparsify import set_submodule
from contextual_sparsity.utils.turbosparse import remove_turbosparse_predictors

# A logger for this file
log = logging.getLogger(__name__)


def trim_layers(model: PreTrainedModel, model_id: str) -> PreTrainedModel:
    """
    Remove all layers but the first one from the model
    """
    layer_container = MODEL_MAPS[model_id][LAYERS_CONTAINER]
    layers = model.get_submodule(layer_container)
    set_submodule(model, layer_container, layers[:1])
    MODEL_MAPS[model_id][N_LAYERS] = 1

    return model


def load_hf_model(
    pretrained_model_path: str,
    model_id: str,
    dtype: Optional[Union[str, torch.dtype]] = None,
    model_type: Type[PreTrainedModel] = AutoModelForCausalLM,
    test_mode: bool = False,
    device: Union[str, torch.device] = "cpu",
    local_files_only: bool = True,
    remove_predictors: bool = True,
) -> torch.nn.Module:
    """
    Load a pre-trained model from Huggingface given a specified path.
    """
    # Parse the data type
    torch_dtype = parse_dtype(dtype)

    if model_id not in MODEL_MAPS:
        raise ValueError(f"Model {model_id} not found in {MODEL_MAPS.keys()}")

    # The net is stored locally
    log.info(
        f"Loading the {model_id} pretrained model from {pretrained_model_path} using {model_type}"
    )
    model = model_type.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_path,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    if model_id == "turbosparse-mistral" and remove_predictors:
        remove_turbosparse_predictors(model, model_id)

    # Split up and gate into separate matrices
    if "phi-3" in model_id:
        split_upgate(model, model_id)

    # Trim all layers but the first if in test mode
    if test_mode:
        model = trim_layers(model, model_id)

    return model.to(device)
