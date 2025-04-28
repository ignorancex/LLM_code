import torch
from torch import nn


class Polynomial(nn.Module):
    """
    This is the polynomial layer for the Linka/Kuhl paper. This
    layer will only ever have a single input and the number of outputs
    will equal the number of nonlinear function types.
    """

    def __init__(self, order, output_features):
        super().__init__()
        self.order = order
        self.out_size = output_features
        self.weight = nn.Parameter(torch.full((output_features,), 1.))

    def forward(self, input):
        output = (input.expand(
            (self.out_size, input.shape[0]))**self.order * self.weight[:, None]).T
        return output


class PolynomialPW(nn.Module):
    """
    This is the polynomial layer for the Linka/Kuhl paper. This
    layer will only ever have a single input and the number of outputs
    will equal the number of nonlinear function types.
    """

    def __init__(self, order, output_features):
        super().__init__()
        self.order = order
        self.out_size = output_features
        self.weight = nn.Parameter(torch.full((output_features,), 1.))

    def forward(self, input):
        weights = nn.functional.softplus(self.weight[:, None])
        output = (input.expand((self.out_size, input.shape[0]))**self.order * weights).T
        return output


class Exponential(nn.Module):
    """
    This is the exponential layer for the Linka/Kuhl paper. This
    layer will only ever have a single input and single output.
    """

    def __init__(self, order):
        super().__init__()
        self.order = order
        self.weight = nn.Parameter(torch.full((1,), 0.))
        self.bias = nn.Parameter(torch.full((1,), 0.))

    def forward(self, input):
        output = self.weight[:, None] * torch.exp(input)**self.order + self.bias
        return output


class ExponentialPW(nn.Module):
    """
    This is the exponential layer for the Linka/Kuhl paper. This
    layer will only ever have a single input and single output.

    The input-output relation is as follows:
    output_vec = exp(input_vec)^order + bias_vec
    NOTE: the bias does not seem necessary
    """

    def __init__(self, order):
        super().__init__()
        self.order = order
        self.weight = nn.Parameter(torch.full((1,), 0.))
        self.bias = nn.Parameter(torch.full((1,), 0.))

    def forward(self, input):
        weights = nn.functional.softplus(self.weight[:, None])
        output = weights * torch.exp(input)**self.order + self.bias
        return output
