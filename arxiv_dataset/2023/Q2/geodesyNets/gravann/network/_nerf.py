import torch
from torch import nn

from .layers import NERFLayer


class NERF(nn.Module):
    def __init__(self, in_features, skip=[], n_neurons=100, activation=nn.Sigmoid(), hidden_layers=8):
        super().__init__()
        self.in_features = in_features
        self.skip = skip

        self.net = nn.ModuleList()
        self.net.append(
            NERFLayer(in_features, n_neurons))

        for i in range(hidden_layers):
            if i in self.skip:
                self.net.append(NERFLayer(n_neurons + in_features, n_neurons))
            else:
                self.net.append(NERFLayer(n_neurons, n_neurons))

        self.net.append(NERFLayer(n_neurons, 1, activation=activation))

    def forward(self, x):
        # save for skip connection
        identity = x

        # compute first layer
        out = self.net[0].forward(x)

        # compute all other layers and apply skip where requested
        for layer_idx in range(1, len(self.net)):
            out = self.net[layer_idx].forward(out)
            if layer_idx in self.skip:
                out = torch.cat([out, identity], dim=1)

        return out
