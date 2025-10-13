# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
import os
from copy import deepcopy
from typing import List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn
from transformers import PreTrainedTokenizer

from contextual_sparsity.mask import MaskingHook
from contextual_sparsity.masking_hooks.binarization import (
    BinarizationType,
    build_binarization,
)
from contextual_sparsity.utils.layer_names import (
    FC_DOWN,
    block_id_to_layer_ids,
    get_block_ids,
    get_layer_ids,
)

log = logging.getLogger(__name__)


def build_predictor_masking_hooks(
    layers_to_sparsify: Union[str, List[int], int],
    model_id: str,
    dense_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    predictor_cache_dir: Optional[str] = None,
    k: Optional[int] = None,
    keep: Optional[float] = None,
    threshold: Optional[float] = None,
    force_retrain: bool = False,
    base_predictor_conf: Optional[DictConfig] = None,
) -> List[MaskingHook]:
    """
    Factory function for building predictive masking hooks (based on a trained masking function).
    This includes methods based on DejaVU
    """
    from contextual_sparsity.masking_hooks.trained.train_predictor import get_predictor

    # Determine the name of the layers to mask
    block_ids = get_block_ids(model_id=model_id, block_names=layers_to_sparsify)
    down_layer_ids = get_layer_ids(
        model_id=model_id, layer_type=FC_DOWN, layer_names=layers_to_sparsify
    )
    down_activation_ids = [".".join([layer_id, "input"]) for layer_id in down_layer_ids]

    # Build the layer responsible for making the activations binary
    activation_binarization = build_binarization(
        activation_ids=down_activation_ids,
        model_id=model_id,
        dense_model=dense_model,
        data_id=None,
        calibration_data=None,
        binarization_type=(
            BinarizationType.topk.value if threshold is None else BinarizationType.threshold.value
        ),
        preprocess_batch=None,
        keep=keep,
        k=k,
        threshold=threshold,
    )

    # Load the predictor conf if not provided
    if base_predictor_conf is None:
        config_file = os.path.join(".hydra", "config.yaml")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                f"No config file found at {config_file}. Please specify predictor_conf"
            )
        base_predictor_conf = OmegaConf.load(config_file).predictor

        OmegaConf.resolve(base_predictor_conf)

    # Determine which device to use
    device = next(dense_model.parameters()).device

    # Wrap each one in a corresponding masking hook
    masking_hooks = []
    for i, block_id in enumerate(block_ids):
        up_layer_id, down_layer_id, gate_layer_id = block_id_to_layer_ids(
            block_id=block_id, model_id=model_id
        )

        log.info(f"Preparing the predictor masking hook for layer {block_id}")
        mask_cols_of = [down_layer_id]
        mask_rows_of = [up_layer_id]
        if gate_layer_id is not None:
            mask_rows_of.append(gate_layer_id)
        input_from = up_layer_id
        input_activation_id = ".".join([input_from, "input"])

        # Make a layer-specific predictor configuration from the base and the layer ids
        predictor_conf = deepcopy(base_predictor_conf)
        with open_dict(predictor_conf):
            predictor_conf.layer_to_mask = block_id
            predictor_conf.input_activation = input_activation_id

        masking_func = get_predictor(
            predictor_conf=predictor_conf,
            predictor_cache_dir=predictor_cache_dir,
            force_retrain=force_retrain,
            model_id=model_id,
            dense_model=dense_model,
            tokenizer=tokenizer,
        ).to(device)

        masking_hook = MaskingHook(
            masking_func=nn.Sequential(
                masking_func,
                activation_binarization[".".join([down_layer_id, "input"])],
            ),
            input_from=input_activation_id.replace(".input", ""),
            mask_cols_of=mask_cols_of,
            mask_rows_of=mask_rows_of,
        )

        # When training sequentially,
        # add the hook to the model with the new predictor before computing the new activations

        masking_hooks.append(masking_hook)

    return masking_hooks
