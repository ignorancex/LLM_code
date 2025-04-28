import sys
sys.path.append('../imports/')

from fibernn.mechanics import *
from nn_functionals import ConvexLinear, ExponentialNN, PolynomialNN, WeightedExponentialNN
from fibernn.io import Adios2NetworkDataset
from fibernn import mechanics, io
from typing import Callable
from torch import Tensor
from matplotlib import pyplot as plt
from torch.utils import data
from torch import nn
import torch
import sys
from loss_functions import *
from execs import def_exec

# Hyper-params
TAG = 'KHUL'
SEED = 4  # not really a hyperparam
BATCH_SIZE = 128
LEARNING_RATE = 2.5E-1
EPOCH_MAX = 200
WEIGHT_DECAY = 0


class KuhlNN(nn.Module):
    def __init__(self, hl_spec: list[int], in_size: int = 3):
        super().__init__()
        print('NN Identifier: Kuhl Neural Network')
        lspec = [in_size] + hl_spec
        self._linears = nn.ModuleList([ConvexLinear(i, j) for (i, j) in zip(lspec[:-1], lspec[1:])])
        self._passl = nn.ModuleList([nn.Linear(in_size, i) for i in lspec[1:]])
        self._activations = nn.ModuleList(nn.Softplus() for i in lspec[1:])
        self._factiv = nn.Softplus()
        self._flinear = ConvexLinear(hl_spec[-1], 1)  # final linear layer
        self._fpass = nn.Linear(in_size, 1)  # final pass through layer

    @staticmethod
    def convert2invariants(X: Tensor) -> Tensor:
        I1 = mechanics.computeI1(X).reshape(-1, 1) - 3
        I2 = mechanics.computeI2(X).reshape(-1, 1) - 3
        I3 = mechanics.computeI3(X).reshape(-1, 1) - 1
        # return torch.hstack((I1, I2, I3, I1 * I2, I2 * I3, I1 * I3))  # --> in_size = 6
        return torch.hstack((I1, I2, I3))

    def forward(self, X: torch.Tensor):
        invars = self.convert2invariants(X)
        z_prev = invars.clone()
        for lid, (lmap, activ, pmap) in enumerate(zip(self._linears,
                                                      self._activations, self._passl)):
            z_this = activ(lmap(z_prev) + pmap(invars))
            z_prev = z_this
            # print(lid, invars.mean())
        output = self._factiv(self._flinear(z_prev) + self._fpass(invars))
        return output
