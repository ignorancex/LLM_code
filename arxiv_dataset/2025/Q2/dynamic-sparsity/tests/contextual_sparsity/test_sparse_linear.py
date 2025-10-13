# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Callable

import pytest
import torch
from torch import nn

from contextual_sparsity.nn.sparse.linear import SimulatedSparseLinear, SparseLinear
from contextual_sparsity.utils import sparsify_linear

input_dim = 3
output_dim = 4

device = "cpu"
secondary_device = "cpu"


@pytest.mark.parametrize(
    "sparsification_func",
    [
        lambda x: sparsify_linear(x, simulated=True),
    ],
    ids=["simulated"],
)
def test_sparse_linear_shapes(sparsification_func: Callable[[nn.Linear], SparseLinear]):
    # Create a DynamicSparseLinear layer
    dense_layer = nn.Linear(input_dim, output_dim, device=device)
    sparse_layer = sparsification_func(dense_layer)

    ###########
    # No Mask #
    ###########

    # Check the output is consistent
    # Without setting the active rows or columns, these should perform the same
    x = torch.randn((10, input_dim)).to(device)
    hat_y = sparse_layer(x)
    y = dense_layer(x)

    # Check the two layers perform the same operation
    assert torch.pow(y - hat_y, 2).sum() == 0, "Inconsistent output when nothing is sparsified"

    ############
    # Col Mask #
    ############

    # Set some columns as active
    col_mask = torch.tensor([0, 1, 1]).bool()
    sparse_layer.set_active(col_mask=col_mask)
    if isinstance(sparse_layer, SimulatedSparseLinear):
        masked_x = x
    else:
        masked_x = x[:, col_mask]
    hat_y = sparse_layer(masked_x)

    # Check devices
    assert str(x.device).startswith(device)

    # And shapes
    if not isinstance(sparse_layer, SimulatedSparseLinear):
        assert col_mask.sum() == sparse_layer.weight.shape[1]
        assert sparse_layer.bias.shape == dense_layer.bias.shape

    assert dense_layer.weight.shape == sparse_layer._weight.shape
    assert dense_layer.bias.shape == sparse_layer._bias.shape
    assert hat_y.shape[0] == x.shape[0]
    assert hat_y.shape[1] == y.shape[1]

    assert torch.pow(y - hat_y, 2).sum() > 0, "The two outputs should be different when masked"

    ###################
    # No Mask (Reset) #
    ###################
    sparse_layer.reset_active()
    hat_y = sparse_layer(x)

    # Check the two layers perform the same operation
    assert (
        torch.pow(y - hat_y, 2).sum() == 0
    ), "Inconsistent output when nothing is sparsified (reset does not work)."

    ############
    # Row Mask #
    ############

    row_mask = torch.tensor([1, 0, 0, 1]).bool()
    sparse_layer.set_active(row_mask=row_mask)
    print(sparse_layer._row_mask.shape)
    hat_y = sparse_layer(x)

    # Check devices
    assert str(x.device).startswith(device)

    # And shapes
    if not isinstance(sparse_layer, SimulatedSparseLinear):
        assert row_mask.sum() == sparse_layer.weight.shape[0]
        assert sparse_layer.bias.shape[0] == row_mask.sum()
        assert hat_y.shape[1] == row_mask.sum()

    assert dense_layer.weight.shape == sparse_layer._weight.shape
    assert dense_layer.bias.shape == sparse_layer._bias.shape
    assert hat_y.shape[0] == x.shape[0]

    ####################
    # Row and Col Mask #
    ####################

    sparse_layer.reset_active()
    sparse_layer.set_active(row_mask=row_mask, col_mask=col_mask)

    if isinstance(sparse_layer, SimulatedSparseLinear):
        masked_x = x
    else:
        masked_x = x[:, col_mask]

    hat_y = sparse_layer(masked_x)

    # Check devices
    assert str(x.device).startswith(device)

    # And shapes
    if not isinstance(sparse_layer, SimulatedSparseLinear):
        assert row_mask.sum() == sparse_layer.weight.shape[0]
        assert col_mask.sum() == sparse_layer.weight.shape[1]
        assert sparse_layer.bias.shape[0] == row_mask.sum()
        assert hat_y.shape[1] == row_mask.sum()

    assert dense_layer.weight.shape == sparse_layer._weight.shape
    assert sparse_layer._bias.shape == dense_layer.bias.shape
    assert hat_y.shape[0] == x.shape[0]
