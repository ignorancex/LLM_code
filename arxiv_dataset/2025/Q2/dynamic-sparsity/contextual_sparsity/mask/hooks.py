# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Any, List, Optional

import torch
from torch import nn
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle

from contextual_sparsity.nn.sparse.linear import SparseLinear


class HooksContainer(nn.ModuleList):
    """
    A container that holds all the masking hooks for a model.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    def eval(self: T) -> T:
        return super().eval()

    def remove_all(self):
        while len(self) > 0:
            masking_hook = self.__getitem__(0)
            masking_hook.remove()


class MaskingHook(nn.Module):
    def __init__(
        self,
        masking_func: nn.Module,
        input_from: str,
        mask_cols_of: Optional[List[str]],
        mask_rows_of: Optional[List[str]],
        pre_forward: bool = True,
    ):
        """
        Class designed to apply a binary mask obtained by applying a function `masking_func` to the tensor obtained
        from `input_from`. The mask is applied to the layer referenced to `mask_col_of` and `mask_row_of`.

        Args:
            masking_func (nn.Module): Function that produces a binary mask starting from the input tensor obtained from
                `input_from`.
            input_from (torch.Tensor): Reference to the activations that need to be fed to the masking_func
            mask_cols_of (Optional[List[str]]): List of string references to layer names. The mask will be applied to the columns
                of the specified layers (if any).
            mask_rows_of (Optional[List[str]]): List of string references to layer names. The mask will be applied to the rows
                of the specified layers (if any).
            pre_forward (bool): whether to register a pre-forward (if True) or post-forward hook (if False).
        """

        super().__init__()

        self.masking_func = masking_func
        self.input_from = input_from
        if mask_cols_of is None:
            mask_cols_of = []
        if mask_rows_of is None:
            mask_rows_of = []

        self.mask_rows_of = mask_rows_of
        self.mask_cols_of = mask_cols_of
        self.pre_forward = pre_forward

        self._input_handle = None
        self._output_handles: List[RemovableHandle] = []
        self._masked_row_modules: List[nn.Module] = []
        self._masked_col_modules: List[nn.Module] = []
        self._attached_to = None
        self.dense_mode = False

    def is_attached(self) -> bool:
        """
        Checks if the hook is attached to the model.
        """
        return self._input_handle is not None

    def remove(self):
        """
        Removes the hook from the model.
        """
        if self._attached_to is not None:
            model = self._attached_to[0]
            if hasattr(model, "masking_hooks"):
                if self in model.masking_hooks:
                    # Find and remove self
                    idx = -1
                    for i, hook in enumerate(model.masking_hooks):
                        if hook == self:
                            idx = i
                    del model.masking_hooks[idx]
                # If not masking hook is left, delete the container
                if len(model.masking_hooks) == 0:
                    del model.masking_hooks

        if self._input_handle is not None:
            self._input_handle.remove()

        for handle in self._output_handles:
            handle.remove()

        self._attached_to = None
        self._input_handle = None
        self._output_handles = []
        self._masked_row_modules = []
        self._masked_col_modules = []

    def attach_to(self, model: nn.Module) -> nn.Module:
        """
        Attaches the hook to the model.

        Args:
            model (nn.Module): Model to attach the hook to.
        """
        self._attached_to = [model]

        # Check if not already attached
        if self.is_attached():
            raise RuntimeError("The masking hook is already attached.")

        # Get the input module
        input_module = model.get_submodule(self.input_from)

        # And the ones to mask
        for module_name in self.mask_rows_of:
            module = model.get_submodule(module_name)
            if not isinstance(module, SparseLinear):
                raise RuntimeError(
                    "The attribute mask_rows_of must point to a SparseLinear module."
                )
            self._masked_row_modules.append(module)
            # Register a handle to invalidate masks after forward
            self._output_handles.append(
                module.register_forward_hook(lambda m, x, y: m.reset_active())
            )

        for module_name in self.mask_cols_of:
            module = model.get_submodule(module_name)
            if not isinstance(module, SparseLinear):
                raise RuntimeError(
                    "The attribute mask_rows_of must point to a SparseLinear module."
                )
            self._masked_col_modules.append(module)
            # Register a handle to invalidate masks after forward
            self._output_handles.append(
                module.register_forward_hook(lambda m, x, y: m.reset_active())
            )

        if self.pre_forward:
            self._input_handle = input_module.register_forward_pre_hook(self._hook)
        else:
            self._input_handle = input_module.register_forward_hook(self._hook)

        # Add a masking_hooks
        if not hasattr(model, "masking_hooks"):
            model.add_module("masking_hooks", HooksContainer())

        assert isinstance(model.masking_hooks, HooksContainer)
        model.masking_hooks.append(self)

        return model

    def set_dense(self):
        """
        Disables the masking hook, replacing the binary mask with one consisting of all ones.
        """
        self.dense_mode = True

    def set_sparse(self):
        """
        Enables the masking hook after set_dense() is called.
        """
        self.dense_mode = False

    def set_masking_func(self, masking_func):
        """
        Sets a masking function to be applied to the tensor obtained from `input_from`.
        """
        self.masking_func = masking_func

    def _hook(self, *args, **kwargs):
        if self.pre_forward:
            return self._forward_pre_hook(*args, **kwargs)
        else:
            return self._forward_hook(*args, **kwargs)

    def _forward_hook(self, module, input, output):
        self._apply_masks(output)

    def _forward_pre_hook(self, module, args):
        assert len(args) == 1
        self._apply_masks(args[0])

    def _apply_masks(self, input: Any):
        if self.dense_mode:
            return
        mask = self.masking_func(input)
        for module in self._masked_row_modules:
            module.set_active(row_mask=mask)

        for module in self._masked_col_modules:
            module.set_active(col_mask=mask)

    def eval(self: T) -> T:
        self.masking_func.eval()
        return self

    def train(self: T, mode: bool = True) -> T:
        self.masking_func.train(mode)
        return self

    def __repr__(self) -> str:
        s = "MaskingHook("
        s += f"\n  (input_from): {'input' if self.pre_forward else 'output'} of {self.input_from}"
        s += f"\n  (mask_rows_of): {self.mask_rows_of}"
        s += f"\n  (mask_cols_of): {self.mask_cols_of}"
        s += "\n  (masking_func): " + self.masking_func.__repr__().replace("\n", "\n  ")
        s += "\n)"
        return s
