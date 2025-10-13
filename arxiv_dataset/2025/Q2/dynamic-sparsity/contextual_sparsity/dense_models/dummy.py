# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from copy import deepcopy

import torch
from torch import nn

from contextual_sparsity.data.dummy import N_FEATURES, compute_labels
from contextual_sparsity.utils.layer_names import MODEL_MAPS, N_LAYERS


class Identity(nn.Module):
    """
    Mock identity module
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyOutput:
    """
    Mock dummy output for consistency with LLMs.
    """

    def __init__(self, loss):
        self.loss = loss


class DummyBlock(nn.Module):
    """
    Mock dummy block for consistency with LLMs.
    """

    def __init__(self, n_features: int):
        super().__init__()

        identity_layer = nn.Linear(n_features, n_features)
        identity_layer.weight.data = torch.eye(n_features)
        identity_layer.bias.data = torch.zeros(n_features)

        self.up = deepcopy(identity_layer)
        self.activation_fn = Identity()
        self.down = deepcopy(identity_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.activation_fn(x)
        x = self.down(x)
        return x


def compute_loss(output, labels):
    prediction = compute_labels(output)
    return torch.pow(prediction - labels, 2).float().sum(-1).mean()


class DummyModel(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_layers: int = MODEL_MAPS["dummy"][N_LAYERS],
        model_id=None,
        device="cpu",
    ):
        super().__init__()
        self.layers = nn.Sequential(*[DummyBlock(n_features) for _ in range(n_layers)])
        self.to(device)

    def forward(self, x, labels):
        output = self.layers(x)
        loss = compute_loss(output, labels)

        # We wrap the output into a dummy container to add the attribute .loss
        return DummyOutput(loss)
