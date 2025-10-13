# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from enum import Enum
from typing import Callable, List, Optional, Union

from torch import nn
from torch.utils.data import DataLoader

from contextual_sparsity.mask import MaskingHook
from contextual_sparsity.masking_hooks.binarization import (
    BinarizationType,
    build_binarization,
)
from contextual_sparsity.nn import Abs
from contextual_sparsity.utils.layer_names import (
    FC_ACT,
    FC_UP,
    MODEL_MAPS,
    block_id_to_layer_ids,
    block_id_to_mlp_id,
    get_block_ids,
    get_layer_ids,
    has_gate,
)


class PredictorType(Enum):
    up = "up"
    gate = "gate"


def build_partial_glu_pruning_masking_hooks(
    model_id: str,
    dense_model: nn.Module,
    layers_to_sparsify: Union[str, List[int], int],
    predictor_type: str,
    binarization_type: str = BinarizationType.topk.value,
    data_id: Optional[str] = None,
    calibration_data: Optional[DataLoader] = None,
    preprocess_batch: Optional[Callable] = None,
    keep: Optional[Union[int, List[int]]] = None,
    k: Optional[Union[int, List[int]]] = None,
    threshold: Optional[Union[int, List[int]]] = None,
) -> List[MaskingHook]:
    """
    Factory function for building either UP or GATE masking hooks based on the specified predictor type
    (either up or gate).
    """

    predictor_type = PredictorType[predictor_type]

    if predictor_type == PredictorType.gate:
        assert has_gate(model_id)

    block_ids = get_block_ids(model_id=model_id, block_names=layers_to_sparsify)

    if predictor_type == PredictorType.up:
        layer_ids = get_layer_ids(
            model_id=model_id, layer_type=FC_UP, layer_names=layers_to_sparsify
        )
    else:
        layer_ids = get_layer_ids(
            model_id=model_id, layer_type=FC_ACT, layer_names=layers_to_sparsify
        )
    activation_ids = [".".join([layer_id, "output"]) for layer_id in layer_ids]

    # Build the layer responsible for making the activations binary
    activation_binarization = build_binarization(
        activation_ids=activation_ids,
        model_id=model_id,
        dense_model=dense_model,
        data_id=data_id,
        calibration_data=calibration_data,
        binarization_type=binarization_type,
        threshold=threshold,
        preprocess_batch=preprocess_batch,
        keep=keep,
        k=k,
    )

    masking_hooks = []
    for i, block_id in enumerate(block_ids):
        up_layer_id, down_layer_id, gate_layer_id = block_id_to_layer_ids(
            model_id=model_id, block_id=block_id
        )
        mlp_layer_id = block_id_to_mlp_id(model_id=model_id, block_id=block_id)
        act_id = ".".join([block_id, MODEL_MAPS[model_id][FC_ACT]])

        mask_rows_of = []
        if predictor_type == PredictorType.up:
            predictor = dense_model.get_submodule(up_layer_id)
            if gate_layer_id is not None:
                mask_rows_of.append(gate_layer_id)
            binarization_id = ".".join([up_layer_id, "output"])
        else:
            predictor = dense_model.get_submodule(gate_layer_id)
            act_fn = dense_model.get_submodule(act_id)
            predictor = nn.Sequential(predictor, act_fn)
            mask_rows_of.append(up_layer_id)
            binarization_id = ".".join([act_id, "output"])

        # Predict the activations (using gate or up), apply the absolute value, and then make them binary
        masking_func = nn.Sequential(
            predictor,
            Abs(),
            activation_binarization[binarization_id],
        )

        # Wrap it into a masking hook, which contains references to where it should be attached
        masking_hook = MaskingHook(
            masking_func=masking_func,
            input_from=mlp_layer_id,
            mask_rows_of=mask_rows_of,
            mask_cols_of=[down_layer_id],
        )
        masking_hooks.append(masking_hook)

    return masking_hooks
