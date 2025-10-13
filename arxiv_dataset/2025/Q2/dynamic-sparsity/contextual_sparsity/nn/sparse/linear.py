# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SparseLinear(nn.Module):
    allow_different_mask_size: bool = False

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
    ):
        super().__init__()

        self.store_weights(weight, bias)
        self.layer_name = layer_name

        # No masking is active
        self._row_mask = None
        self._col_mask = None

        self._output_correction = None

    def store_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        raise NotImplementedError()

    def load_weight_rows(self, mask: torch.BoolTensor) -> torch.Tensor:
        raise NotImplementedError()

    def load_weight_cols(self, mask: torch.BoolTensor) -> torch.Tensor:
        raise NotImplementedError()

    def load_weight_rows_and_cols(
        self, row_mask: torch.BoolTensor, col_mask: torch.BoolTensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def check_valid_mask(self, mask: torch.BoolTensor):
        # Check the mask has the correct shape
        if mask.ndim >= 2:
            # Here we are working with a batch_row_mask of shape [BATCH_SIZE, M]
            # Check we are selecting the same number of columns for each element of the batch
            n_selected_entries = mask.sum(-1)
            if (n_selected_entries == n_selected_entries[0]).sum() != len(
                n_selected_entries
            ) and not self.allow_different_mask_size:
                raise NotImplementedError(
                    "The batch of masks contains different number of entries per batch elements. This is not supported"
                )

    def check_valid_row_mask(self, row_mask: torch.BoolTensor):
        return self.check_valid_mask(row_mask)

    def check_valid_col_mask(self, col_mask: torch.BoolTensor):
        return self.check_valid_mask(col_mask)

    def set_active(
        self,
        row_mask: Optional[torch.BoolTensor] = None,
        col_mask: Optional[torch.BoolTensor] = None,
    ):
        # Basic shape checks
        if row_mask is not None:
            self.check_valid_row_mask(row_mask)

        if col_mask is not None:
            self.check_valid_col_mask(col_mask)

        # Update the masks
        self._row_mask = row_mask
        self._col_mask = col_mask

    def reset_active(self):
        self._row_mask = None
        self._col_mask = None

    def set_output_correction(self, correction: nn.Module):
        self._output_correction = correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._forward(x)
        if self._output_correction is not None:
            y = self._output_correction(y)
        return y


class SimulatedSparseLinear(SparseLinear):
    allow_different_mask_size: bool = True

    def store_weights(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        self.register_buffer("_weight", weight)
        self.register_buffer("_bias", bias)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        end_squeeze = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            end_squeeze = True

        # Apply the column mask by zeroing out the inputs
        if self._col_mask is not None:
            # One mask per batch
            if self._col_mask.ndim == 1:
                x = x * self._col_mask.to(x.device).unsqueeze(0)
            elif self._col_mask.ndim >= 2:
                x = x * self._col_mask.to(x.device)

        y = F.linear(x, self._weight, self._bias)

        # squeeze the batch dimension if it was inflated artificially
        if end_squeeze:
            y = y.squeeze(0)

        return y

    def extra_repr(self):
        return f"in_features={self._weight.shape[-1]}, out_features={self._weight.shape[-2]}, bias={self._bias is not None}"
