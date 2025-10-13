# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from contextual_sparsity.utils.submodule import set_submodule


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self._handles = []
        self._enabled = True
        self._attached_to = []

    @staticmethod
    @abstractmethod
    def from_module(linear: nn.Module, *args, **kwargs) -> "Adapter":
        """
        Creates an adapter from a Linear module
        """
        pass

    def enable(self):
        """
        Enables the Adapter
        """
        self._enabled = True

    def disable(self):
        """
        Disables the Adapter
        """
        self._enabled = False

    def attach_to(self, module: nn.Module) -> None:
        """
        Attaches the adapter to a module by registering it as a forward hook
        """
        self._handles.append(
            module.register_forward_hook(self.hook, with_kwargs=True, prepend=True)
        )
        self._attached_to.append(module)

    def hook(
        self, module: nn.Module, args: List[Any], kwargs: Dict[str, Any], out
    ) -> Optional[torch.Tensor]:
        """
        Hook function that can be registered as a forward hook to a linear module.
        See https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html
        for more details on the hook function and its arguments.
        """
        if self._enabled:
            return self._hook(module, args, kwargs, out)
        else:
            return None

    @abstractmethod
    def _hook(
        self,
        module: nn.Module,
        args: List[Any],
        kwargs: Dict[str, Any],
        out: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def remove(self):
        """
        Removes the Adapter from the list of handles
        """
        for handle in self._handles:
            handle.remove()
        self._attached_to = []


def add_adapters(sparse_model: nn.Module, adapter_conf: DictConfig):
    """
    Adds adapters to a specified sparse model using the provided adapter_conf.

    Args:
        sparse_model (nn.Module): the sparse model to be modified
        adapter_conf (DictConfig): the configuration of the adapters
    """
    if not hasattr(sparse_model, "adapters"):
        set_submodule(sparse_model, "adapters", nn.ModuleDict())

    device = next(sparse_model.parameters()).device
    make_adapter = instantiate(adapter_conf.model)

    # Add all the adapters
    for masking_hook in sparse_model.masking_hooks:
        for layer_id in masking_hook.mask_rows_of:
            linear_module = sparse_model.get_submodule(layer_id)
            adapter = make_adapter(linear_module).to(device)
            adapter.attach_to(linear_module)
            sparse_model.adapters[layer_id.replace(".", "_")] = adapter

        for layer_id in masking_hook.mask_cols_of:
            linear_module = sparse_model.get_submodule(layer_id)
            adapter = make_adapter(linear_module).to(device)
            adapter.attach_to(linear_module)
            sparse_model.adapters[layer_id.replace(".", "_")] = adapter
