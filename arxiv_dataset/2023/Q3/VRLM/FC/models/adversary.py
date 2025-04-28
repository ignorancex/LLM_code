#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Create the adversary, which is just a tensor of the size of the data
    Also has 2-norm projection operator, which takes in a LIST of tensors (but len = 1 of the list)
"""

# Import relevant modules
import torch
import typing
import torch.nn as nn


# Create Architecture
class Adversary(nn.Module):
    def __init__(self, channels: int = 3, dims: int = 32) -> None:
        super(Adversary, self).__init__()

        # Create a parameter to tune
        weights = torch.zeros((channels, dims, dims))
        weights[0][0][0] = 1.0
        # norm = torch.norm(weights, p='fro')
        self.adversary_parameters = nn.Parameter(weights)

    def forward(self,
                data: torch.Tensor,
                target: torch.Tensor,
                model: torch.nn.Module,
                loss: typing.Optional) -> torch.Tensor:
        """Performs a forward pass of the model,
        computes the loss and returns the loss for gradient computation later"""

        # Compute the model loss, with the parameters changed
        out = model(data + self.adversary_parameters)
        l = loss(out, target) # the loss here will result in gradient computations with respect to both the parameters AND the adversary

        return l

    @torch.no_grad()
    def projection(self, tensor_list: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]:

        # First perform one pass through the list to get the two norm
        # sqrt(sum||x_i||_F^2) = ||X||_F
        norm = torch.sqrt(sum([torch.norm(tensor_list[i], p="fro") ** 2 for i in range(len(tensor_list))]))

        return [tensor_list[i] / max(norm, 1) for i in range(len(tensor_list))]