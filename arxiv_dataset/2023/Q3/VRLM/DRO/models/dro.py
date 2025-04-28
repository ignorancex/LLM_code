#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    DRO problem (y-term)
"""

# Import relevant modules
import torch
import typing
import numpy as np
import torch.nn as nn


#  Create Architecture
class DRO(nn.Module):
    def __init__(self, number_data: int, number_devices: int, scale: float = 1.0, l1: float = 0.0) -> None:
        super(DRO, self).__init__()

        # Create a parameter to tune
        weights = torch.ones((number_data,))
        weights = weights / weights.sum()
        # norm = torch.norm(weights, p='fro')
        self.weights = nn.Parameter(weights)
        self.scale = scale
        self.l1 = l1
        self.number_data = number_data
        self.number_devices = number_devices

        # Save thresh-holder; gives element-wise max of zero and input
        self.max_0 = torch.nn.Threshold(threshold=0.0, value=0.0)

    def forward(self,
                data: torch.Tensor,
                target: torch.Tensor,
                model: torch.nn.Module,
                loss: typing.Optional,
                index_slice: typing.Optional = None) -> torch.Tensor:
        """Performs a forward pass of the model,
        computes the loss and returns the loss for gradient computation later"""

        # Compute the model loss, with the parameters changed
        out = model(data)
        l = loss(out, target) # returns a vector of length of number of local data
        l = self.number_devices * (l * self.weights[index_slice]).sum() # scale appropriately
        l -= (self.scale / 2) * torch.norm(self.weights - (1 / self.number_data), p=2) ** 2

        return l

    def get_optimal_y(self,
                data: torch.Tensor,
                target: torch.Tensor,
                model: torch.nn.Module,
                loss: typing.Optional) -> torch.Tensor:
        """Populates the gradients wrt to x"""

        # Compute the model loss, with the parameters changed
        model.zero_grad()
        out = model(data)
        l = loss(out, target) # returns a vector of length of number of local data

        # Compute the optimal y
        y_opt = self.projection_y([(l / self.scale) + (1 / self.number_data)])[0]
        l = (l * y_opt).sum() # scale appropriately
        l.backward()

        # Project using soft thresholding
        proj_x = self.projection_x([p.data.cpu().detach().clone() - p.grad.data.cpu().detach().clone() for p in model.parameters()], self.l1)
        opt_grad = [p.data.cpu().detach().clone().flatten() - proj_x[ind].flatten() for ind, p in enumerate(model.parameters())]
        return torch.cat(opt_grad).flatten()

    @torch.no_grad()
    def projection_x(self, list_of_tensors: list, l1: float = 0.0) -> list:
        '''Perform regularization'''

        # Save computation if regularizer is 0
        if l1 == 0.0:

            return list_of_tensors

        else:
            # Loop over all of the tensors
            for i, t in enumerate(list_of_tensors):
                # Modify current tensors
                list_of_tensors[i] = torch.sign(t) * self.max_0(torch.abs(t) - l1)

            return list_of_tensors

    @torch.no_grad()
    def number_non_zeros(self, list_of_tensors: list) -> tuple:
        '''Count the number of non-zero entries'''

        # Save values
        nnz = 0
        total_params = 0

        # Loop over the parameters
        for _, t in enumerate(list_of_tensors):
            total_params += torch.prod(torch.tensor(t.shape)).item()
            nnz += (torch.abs(t) > 1e-6).sum().item()

        # Return both the count and the ratio
        return nnz, (nnz / total_params)

    @torch.no_grad()
    def projection_y(self, tensor_list: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]:
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
