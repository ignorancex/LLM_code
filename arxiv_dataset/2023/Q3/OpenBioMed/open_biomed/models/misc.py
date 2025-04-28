from typing import List, Optional, Union

import logging

import torch
import torch.nn as nn

ACT_FN = {
    "softmax": nn.Softmax(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
}

class MLP(nn.Module):
    def __init__(self,
        input_dim: int=256,
        hidden_dim: Union[List[int], int]=[],
        output_dim: int=2,
        num_layers: Optional[int]=None,
        act_fn: str="relu",
        layer_norm: bool=False,
        act_last: bool=False,
    ) -> None:
        super().__init__()
        layers = []
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * num_layers
        else:
            if num_layers is not None:
                logging.warn("Hidden dim is specified, num_layers is ignored")
        hidden_dim = [input_dim] + hidden_dim + [output_dim]
        num_layers = len(hidden_dim) - 1
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            if i < num_layers - 1 or act_last:
                if layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim[i + 1]))
                layers.append(ACT_FN[act_fn])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)