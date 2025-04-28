#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Linear model
"""
import torch.nn as nn


class LINEAR(nn.Module):

    def __init__(self, num_classes: int, input_dim: int):

        super(LINEAR, self).__init__()

        # Save values
        self.num_classes = num_classes
        self.input_dim = input_dim

        # Declare layers
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        '''Forward pass of the model'''

        x = x.reshape(-1, self.input_dim)

        return self.layers(x)