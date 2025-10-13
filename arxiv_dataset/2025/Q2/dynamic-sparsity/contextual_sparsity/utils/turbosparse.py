# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import List

from torch import nn

from contextual_sparsity.utils.layer_names import MLP, get_layer_ids
from contextual_sparsity.utils.submodule import set_submodule


class BambooMLPWithoutPredictor(nn.Module):
    def __init__(self, mlp: nn.Module):
        """
        Takes as input a BambooMLP module as defined in
        https://huggingface.co/PowerInfer/TurboSparse-Mistral-Instruct/blob/main/modeling_bamboo.py
        """
        super().__init__()
        self.config = mlp.config
        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size
        self.layer_id = mlp.layer_id
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn

    def forward(self, x, before_norm):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.act_fn(self.up_proj(x)))


def remove_turbosparse_predictors(model: nn.Module, model_id: str) -> List[nn.Module]:
    """
    Replaces all existing MLPs in TurboSparse models with MLP copies that do not use a predictor.
    """
    layer_names = get_layer_ids(model_id=model_id, layer_type=MLP, layer_names="all")

    predictors = []
    for layer_name in layer_names:
        mlp_with_predictor = model.get_submodule(layer_name)
        predictors.append(mlp_with_predictor.predictor)
        mlp_without_predictor = BambooMLPWithoutPredictor(mlp_with_predictor)
        set_submodule(model, layer_name, mlp_without_predictor)

    return predictors
