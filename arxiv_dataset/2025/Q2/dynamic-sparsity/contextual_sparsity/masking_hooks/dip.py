# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Callable, List, Optional, Union

import numpy as np
from scipy.special import expit, logit
from torch import nn
from torch.utils.data import DataLoader

from contextual_sparsity.mask import MaskingHook
from contextual_sparsity.masking_hooks.binarization import (
    BinarizationType,
    build_binarization,
)
from contextual_sparsity.nn import Abs
from contextual_sparsity.utils.layer_names import (
    FC_DOWN,
    FC_UP,
    block_id_to_layer_ids,
    block_id_to_mlp_id,
    get_block_ids,
    get_layer_ids,
    has_gate,
)


def build_dip_masking_hooks(
    model_id: str,
    dense_model: nn.Module,
    layers_to_sparsify: Union[str, List[int], int],
    data_id: Optional[str] = None,
    calibration_data: Optional[DataLoader] = None,
    preprocess_batch: Optional[Callable] = None,
    binarization_type: str = BinarizationType.topk.value,
    down_k: Optional[Union[int, List[int]]] = None,
    up_k: Optional[Union[int, List[int]]] = None,
    gate_k: Optional[Union[int, List[int]]] = None,
    down_keep: Optional[Union[float, List[float]]] = None,
    up_keep: Optional[Union[float, List[float]]] = None,
    gate_keep: Optional[Union[float, List[float]]] = None,
    down_threshold: Optional[Union[float, List[float]]] = None,
    up_threshold: Optional[Union[float, List[float]]] = None,
    gate_threshold: Optional[Union[float, List[float]]] = None,
) -> List[MaskingHook]:
    """
    Factory function for building DIP masking hooks
    """

    block_ids = get_block_ids(model_id=model_id, block_names=layers_to_sparsify)
    masking_hooks = []
    # If not specified, the sparsity of up and gate are the same
    if gate_k is None and gate_keep is None and gate_threshold is None:
        gate_keep = up_keep
        gate_k = up_k
        gate_threshold = up_threshold

    down_activation_ids = [
        ".".join([layer_id, "input"])
        for layer_id in get_layer_ids(
            model_id=model_id, layer_type=FC_DOWN, layer_names=layers_to_sparsify
        )
    ]

    # Create a binarization function for all the intermediate activations (inputs to down)
    activation_binarization = build_binarization(
        activation_ids=down_activation_ids,
        model_id=model_id,
        dense_model=dense_model,
        data_id=data_id,
        calibration_data=calibration_data,
        preprocess_batch=preprocess_batch,
        binarization_type=binarization_type,
        threshold=down_threshold,
        keep=down_keep,
        k=down_k,
    )

    # Do the same for all the inputs to the up layers
    up_activation_ids = [
        ".".join([layer_id, "input"])
        for layer_id in get_layer_ids(
            model_id=model_id, layer_type=FC_UP, layer_names=layers_to_sparsify
        )
    ]
    activation_binarization.update(
        build_binarization(
            activation_ids=up_activation_ids,
            model_id=model_id,
            dense_model=dense_model,
            data_id=data_id,
            calibration_data=calibration_data,
            preprocess_batch=preprocess_batch,
            binarization_type=binarization_type,
            threshold=up_threshold,
            keep=up_keep,
            k=up_k,
        )
    )

    # If the model has gate and a different binarization policy for the gate component, also compute them
    if (
        up_keep != gate_keep
        or up_k != gate_keep
        or up_threshold != gate_threshold
        and has_gate(model_id)
    ):
        activation_binarization.update(
            build_binarization(
                activation_ids=up_activation_ids,
                model_id=model_id,
                dense_model=dense_model,
                data_id=data_id,
                calibration_data=calibration_data,
                preprocess_batch=preprocess_batch,
                binarization_type=binarization_type,
                threshold=gate_threshold,
                keep=gate_keep,
                k=gate_k,
            )
        )

    for i, block_id in enumerate(block_ids):
        up_layer_id, down_layer_id, gate_layer_id = block_id_to_layer_ids(
            model_id=model_id, block_id=block_id
        )
        mlp_layer_id = block_id_to_mlp_id(model_id=model_id, block_id=block_id)

        #####################
        # Down Masking Hook #
        #####################
        down_masking_hook = MaskingHook(
            masking_func=nn.Sequential(
                Abs(),
                activation_binarization[".".join([down_layer_id, "input"])],
            ),
            input_from=down_layer_id,
            mask_cols_of=[down_layer_id],
            mask_rows_of=[],
        )
        masking_hooks.append(down_masking_hook)

        ###################
        # Up Masking Hook #
        ###################
        up_masking_hook = MaskingHook(
            masking_func=nn.Sequential(
                Abs(),
                activation_binarization[".".join([up_layer_id, "input"])],
            ),
            input_from=mlp_layer_id,
            mask_cols_of=[up_layer_id],
            mask_rows_of=[],
        )
        # Set the input to the input of the MLP block
        masking_hooks.append(up_masking_hook)

        if gate_layer_id is not None:
            gate_activation_id = ".".join([gate_layer_id, "input"])

            # Case 1: gate does not have a dedicated binarization
            # The selected columns are the same as up. We apply the same up_masking_hook to the columns of gate
            if gate_activation_id not in activation_binarization:
                up_masking_hook.mask_cols_of.append(gate_layer_id)

            # Case 2: gate has a different binarization function.
            # The selected columns are different. Make a new masking hook for gate
            else:
                gate_masking_hook = MaskingHook(
                    masking_func=nn.Sequential(
                        Abs(),
                        activation_binarization[gate_activation_id],
                    ),
                    input_from=mlp_layer_id,
                    mask_cols_of=[gate_layer_id],
                    mask_rows_of=[],
                )
                masking_hooks.append(gate_masking_hook)

    return masking_hooks


def optimal_up_keep_from_keep(
    keep: Union[float, List[float], np.ndarray],
) -> Union[float, List[float], np.ndarray]:
    # Best linear fit in logit space
    m, b = (0.9849930711017189, 0.29976477589716316)
    up_keep = expit(logit(keep) * m + b)
    return up_keep


def build_optimized_dip_masking_hooks(
    model_id: str,
    dense_model: nn.Module,
    data_id: str,
    layers_to_sparsify: Union[str, List[int], int],
    keep: Union[float, List[float]],
) -> List[MaskingHook]:
    if not has_gate(model_id):
        raise NotImplementedError("The optimized values are computed for LLMs with gating")

    keep = np.array(keep)
    up_keep = optimal_up_keep_from_keep(keep)
    down_keep = 3 * (keep - 2 / 3.0 * up_keep)

    return build_dip_masking_hooks(
        model_id=model_id,
        dense_model=dense_model,
        data_id=data_id,
        layers_to_sparsify=layers_to_sparsify,
        down_keep=down_keep,
        up_keep=up_keep,
    )
