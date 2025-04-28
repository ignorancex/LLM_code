import torch
from torch import nn


# %% Substitute for linear mapping between layers of FCNN
class ConvexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        """
        Linear mapping function for ICNN with positive weights and no bias.
        This fucntion is from: https://github.dev/EUCLID-code/EUCLID-hyperelasticity-NN

        The output of (i+1) layer has the following form:  `z[k+1] = softplus(W) x z[k] + b[k]`, here
        `W` is the weight in regular `z[k+1] = Wx + b` format.

        :param int in_features: input shape of array
        :param int out_features: output shape of array
        """
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        # create weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight)  # init weight

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the output in forward step. See init doc.

        :param torch.Tensor X: input values
        :return torch.Tensor: output values
        """
        return torch.mm(X, torch.nn.functional.softplus(self.weight.T))


# %% Activation functions
class ExponentialNN(nn.Module):
    def __init__(self) -> None:
        """
        Exponential activation function. y = exp(x)
        """
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(input)


class PolynomialNN(nn.Module):
    def __init__(self, order: int = 2) -> None:
        """
        Exponential activation function. y = x**order
        """
        super().__init__()
        self.order = order

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.pow(input, self.order)


class WeightedExponentialNN(nn.Module):
    def __init__(self) -> None:
        """
        Exponential activation function. y = alpha*exp(x)
        """
        super().__init__()
        # self.weight = nn.Parameter(torch.full((1, 1), 0.1))
        self.weight = nn.Parameter(torch.empty((1, 1)))
        torch.nn.init.kaiming_uniform_(self.weight)  # init weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.weight) * torch.exp(input)
