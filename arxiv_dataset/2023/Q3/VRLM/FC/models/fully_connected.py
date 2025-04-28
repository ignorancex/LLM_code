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
            nn.ReLU(),
            nn.Linear(200, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        '''Forward pass of the model'''

        x = x.view(-1, self.num_channels * self.hidden_dim)

        return self.layers(x)