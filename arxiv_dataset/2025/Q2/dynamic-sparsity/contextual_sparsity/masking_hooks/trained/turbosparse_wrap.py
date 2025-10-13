# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from typing import List, Optional, Union

from torch import nn

from contextual_sparsity.mask import MaskingHook
from contextual_sparsity.masking_hooks.binarization import (
    BinarizationType,
    build_binarization,
)
from contextual_sparsity.utils.layer_names import (
    FC_DOWN,
    LAYERS_CONTAINER,
    MODEL_MAPS,
    block_id_to_layer_ids,
    block_number_from_id,
    get_block_ids,
    get_layer_ids,
)
from contextual_sparsity.utils.turbosparse import remove_turbosparse_predictors

log = logging.getLogger(__name__)


def build_original_turbosparse_hooks(
    model_id: str,
    dense_model: nn.Module,
    layers_to_sparsify: Union[str, List[int], int],
    k: Optional[Union[int, List[int]]] = None,
    keep: Optional[Union[float, List[float]]] = None,
    threshold: Optional[Union[float, List[float]]] = None,
) -> List[MaskingHook]:
    """
    Factory function for wrapping the original pre-trained TurboSparse predictors into MaskingHooks
    """

    assert model_id == "turbosparse-mistral"
    block_ids = get_block_ids(
        model_id=model_id,
        block_names=layers_to_sparsify,
    )
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
            BinarizationType.topk if threshold is None else BinarizationType.threshold
        ),
        preprocess_batch=None,
        keep=keep,
        k=k,
        threshold=threshold,
    )

    predictors = remove_turbosparse_predictors(dense_model, model_id)
    assert len(predictors) == len(dense_model.model.layers)

    masking_hooks: List[MaskingHook] = []
    for i, block_id in enumerate(block_ids):
        up_layer_id, down_layer_id, gate_layer_id = block_id_to_layer_ids(
            block_id=block_id, model_id=model_id
        )

        # Determine which predictor to use
        block_number = block_number_from_id(model_id=model_id, submodule_id=block_id)
        predictor = predictors[block_number]

        # The input comes from before the post_attention_layernom module
        input_from = ".".join(
            [
                MODEL_MAPS[model_id][LAYERS_CONTAINER],
                str(block_number),
                "post_attention_layernorm",
            ]
        )

        masking_hook = MaskingHook(
            masking_func=nn.Sequential(
                predictor,
                activation_binarization[".".join([down_layer_id, "input"])],
            ),
            mask_rows_of=[up_layer_id, gate_layer_id],
            mask_cols_of=[down_layer_id],
            input_from=input_from,
        )
        masking_hooks.append(masking_hook)

    return masking_hooks
