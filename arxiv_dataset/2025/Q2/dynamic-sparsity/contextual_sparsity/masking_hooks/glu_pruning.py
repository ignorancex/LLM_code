# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Callable, List, Optional, Union

from torch import nn
from torch.utils.data import DataLoader

from contextual_sparsity.mask import MaskingHook
from contextual_sparsity.masking_hooks.binarization import (
    BinarizationType,
    build_binarization,
)
from contextual_sparsity.nn import Abs
from contextual_sparsity.utils.layer_names import FC_DOWN, get_layer_ids


def build_glu_pruning_masking_hooks(
    model_id: str,
    dense_model: nn.Module,
    layers_to_sparsify: Union[str, List[int], int],
    binarization_type: str = BinarizationType.topk.value,
    data_id: Optional[str] = None,
    calibration_data: Optional[DataLoader] = None,
    preprocess_batch: Optional[Callable] = None,
    keep: Optional[Union[int, List[int]]] = None,
    k: Optional[Union[int, List[int]]] = None,
    threshold: Optional[Union[int, List[int]]] = None,
) -> List[MaskingHook]:
    """
    Factory function for building GLU pruning masking hooks.
    """

    down_layer_ids = get_layer_ids(
        model_id=model_id, layer_type=FC_DOWN, layer_names=layers_to_sparsify
    )
    down_activation_ids = [".".join([down_layer_id, "input"]) for down_layer_id in down_layer_ids]

    # Build the layer responsible for making the activations binary
    activation_binarization = build_binarization(
        activation_ids=down_activation_ids,
        model_id=model_id,
        dense_model=dense_model,
        data_id=data_id,
        calibration_data=calibration_data,
        binarization_type=binarization_type,
        preprocess_batch=preprocess_batch,
        keep=keep,
        k=k,
        threshold=threshold,
    )

    masking_hooks = []
    for i, layer_id in enumerate(down_layer_ids):
        # Make the masking function
        activation_id = ".".join([layer_id, "input"])

        masking_hook = MaskingHook(
            masking_func=nn.Sequential(
                Abs(),
                activation_binarization[activation_id],
            ),
            input_from=layer_id,
            mask_rows_of=[],
            mask_cols_of=[layer_id],
        )

        masking_hooks.append(masking_hook)

    return masking_hooks
