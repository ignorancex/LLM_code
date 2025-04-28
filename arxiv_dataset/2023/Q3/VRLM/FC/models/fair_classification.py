#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Create the fair term, which is just a tensor the size of the number of classes
"""

# Import relevant modules
import torch
import typing
import numpy as np
import torch.nn as nn


# Create Architecture
class FairTerm(nn.Module):
    def __init__(self, num_classes: int = 10, eta: float = 1e-1) -> None:
        super(FairTerm, self).__init__()

        # Create a parameter to tune
        weights = torch.ones(num_classes) / num_classes
        self.params = nn.Parameter(weights) # Save as parameters
        self.num_classes = num_classes

        # Save the threshold - anything less than 0 is projected back to 0
        self.threshold = nn.Threshold(0.0, 0.0)
        self.eta = eta

    def forward(self, loss_values: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass of the model,
        computes the loss and returns the loss for gradient computation later"""

        # Compute the model loss, scaled by the Y terms
        out = loss_values * self.params
        l = out.sum() - (self.eta / 2) * torch.norm(self.params, p=2) ** 2

        return l

    @torch.no_grad()
    def projection(self, tensor_list: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]:
        """Project onto a simplex"""

        # Sort the tensor
        sorted_tensor, _ = torch.sort(tensor_list[0].to(torch.device('cpu')), descending=True)

        # Scale and get cumulative sum
        cumsum = torch.cumsum(sorted_tensor, dim=-1)
        res = sorted_tensor - (cumsum - 1) / (torch.arange(sorted_tensor.shape[0]) + 1)

        # Find arg-maxs - i.e. rhos from the paper
        max_args = len(res) - np.argmax(res.numpy()[::-1] > 0) - 1
        scales = (1 - cumsum[max_args]) / (max_args + 1)

        # Project to positive orthant
        return [torch.nn.functional.threshold(tensor_list[0] + scales.to(tensor_list[0].device), 0.0, 0.0)]