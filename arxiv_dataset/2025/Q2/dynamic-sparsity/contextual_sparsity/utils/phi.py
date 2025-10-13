# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
from torch import nn

from contextual_sparsity.utils.layer_names import (
    FC_GATE,
    FC_UP,
    MLP,
    MODEL_MAPS,
    get_layer_ids,
)
from contextual_sparsity.utils.misc import split_gate_up_layer
from contextual_sparsity.utils.submodule import set_submodule


class Phi3SplitMLP(nn.Module):
    """
    Wrapper for Phi architecture with separate up and gate linear layer for consistency with the other LLMs.
    """

    def __init__(self, upgate_mlp: nn.Module):
        super().__init__()

        # Split the up and gate
        gate_proj, up_proj = split_gate_up_layer(upgate_mlp.gate_up_proj)

        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = upgate_mlp.down_proj
        self.activation_fn = upgate_mlp.activation_fn

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        gate = self.gate_proj(hidden_states)
        up_states = self.up_proj(hidden_states)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


def split_upgate(model: nn.Module, model_id: str):
    """
    Replaces all existing MLPs in Phi models with MLP copies that have separate up and gate layers.
    """
    layer_names = get_layer_ids(model_id=model_id, layer_type=MLP, layer_names="all")

    for layer_name in layer_names:
        upgate_mlp = model.get_submodule(layer_name)
        mlp = Phi3SplitMLP(upgate_mlp)
        set_submodule(model, layer_name, mlp)

    MODEL_MAPS[model_id][FC_GATE] = ".".join([MODEL_MAPS[model_id][MLP], "gate_proj"])
    MODEL_MAPS[model_id][FC_UP] = ".".join([MODEL_MAPS[model_id][MLP], "up_proj"])
