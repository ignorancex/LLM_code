#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Fully connected (3-layer) architecture
"""
import torch.nn as nn


class FC(nn.Module):

    def __init__(self, num_classes: int, num_channels: int, hidden_dim: int):

        super(FC, self).__init__()

        # Save values
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        # Declare layers
        self.layers = nn.Sequential(
            nn.Linear(self.num_channels * self.hidden_dim, 200),
            nn.Tanh(),
            nn.Linear(200, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def reset_weights(self):
        """Reset the model parameters"""

        for l in self.layers:
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, x):
        '''Forward pass of the model'''

        x = x.reshape(-1, self.num_channels * self.hidden_dim)

        return self.layers(x)