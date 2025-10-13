# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Callable

import torch.nn as nn


class SimulatorResetHook:
    def __init__(self, hw_simulator_reset_fn: Callable, model: nn.Module, active: bool = True):
        super().__init__()
        self._input_handle = model.register_forward_hook(self._hook)
        self.hw_simulator_reset_fn = hw_simulator_reset_fn
        self.active = active

    def _hook(self, *args, **kwargs):
        if self.active:
            self.hw_simulator_reset_fn()

    def set_inactive(self):
        """
        The hook can be disabled when the token generation for the whole sequence does not happen at once with a
        single model forward. This happens either with sequential generation (auto-regressive, without teacher forcing)
        or when part the sequence (the prompt) is encoded first with a dense model.
        """
        self.active = False

    def set_active(self):
        self.active = True

    def is_attached(self) -> bool:
        return self._input_handle is not None

    def remove(self):
        if self.is_attached():
            self._input_handle.remove()
        self._input_handle = None
