# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from copy import deepcopy
from typing import List, Optional, Union

from torch import nn

from contextual_sparsity.mask import MaskingHook
from contextual_sparsity.nn.sparse.linear import SimulatedSparseLinear, SparseLinear
from contextual_sparsity.utils.submodule import set_submodule


def sparsify_linear(
    linear: nn.Linear,
    simulated: bool = False,
    layer_name: Optional[str] = None,
    **kwargs,
) -> SparseLinear:
    """
    Sparsify a linear layer, returning the corresponding wrapped Sparse Linear layer.
    """
    # We are interested in simulating a sparse block, we can just zero out the entries
    if simulated:
        SparseLinearClass = SimulatedSparseLinear
    else:
        # Actual sparse forward can be implemented by creating a subclass of SparseLinear and adding it here
        raise NotImplementedError()

    return SparseLinearClass(
        weight=linear.weight, bias=linear.bias, layer_name=layer_name, **kwargs
    )


def sparsify_model(
    masking_hooks: List[MaskingHook],
    dense_model: nn.Module,
    simulated: bool = True,
) -> nn.Module:
    """
    Sparsify the MLP layers corresponding to the masking hooks.
    """

    # Attach all the hooks
    for masking_hook in masking_hooks:
        for layer_id in masking_hook.mask_rows_of:
            linear_module = dense_model.get_submodule(layer_id)
            if isinstance(linear_module, nn.Linear):
                sparse_linear_module = sparsify_linear(
                    linear_module,
                    layer_name=masking_hook.mask_cols_of,
                    simulated=simulated,
                )
                set_submodule(dense_model, layer_id, sparse_linear_module)

        for layer_id in masking_hook.mask_cols_of:
            linear_module = dense_model.get_submodule(layer_id)
            if isinstance(linear_module, nn.Linear):
                sparse_linear_module = sparsify_linear(
                    linear_module,
                    layer_name=masking_hook.mask_cols_of,
                    simulated=simulated,
                )
            else:
                sparse_linear_module = linear_module
            set_submodule(dense_model, layer_id, sparse_linear_module)

        masking_hook.attach_to(dense_model)

    return dense_model


def build_sparse_model(
    dense_model: nn.Module,
    masking_hooks: Union[MaskingHook, List[MaskingHook]],
    simulated: bool = True,
    inplace: bool = True,
) -> nn.Module:
    """
    Apply all the masking hooks to a dense model, making it sparse.
    """
    if not inplace:
        dense_model = deepcopy(dense_model)

    masking_hooks = list(masking_hooks)

    return sparsify_model(
        dense_model=dense_model,
        masking_hooks=masking_hooks,
        simulated=simulated,
    )
