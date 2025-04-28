import sys
sys.path.append('../imports/')

from fibernn.mechanics import *
from nn_functionals import *
from fibernn.io import Adios2NetworkDataset
from fibernn import mechanics, io
from typing import Callable
from torch import Tensor
from matplotlib import pyplot as plt
from torch.utils import data
from torch import nn
import torch
import sys
from loss_functions import NMSE_loss, NMSE_Abs_loss, NMSE_Sqr_loss
from drivers import TrainLoop, TestLoop, model_persistance
from execs import def_exec

# Hyper-params
TAG = 'FCNN'
SEED = 4  # not really a hyperparam
BATCH_SIZE = 256
LEARNING_RATE = 2.5E-1
EPOCH_MAX = 50
WEIGHT_DECAY = 0

# # torch init setup
# torch.manual_seed(SEED)
# device = 'cpu'


class FullyConnectNN(nn.Module):
    def __init__(self, hl_spec: list[int], in_size: int = 3):
        super().__init__()
        print('NN Identifier: Fully Connected Neural Network')
        lspec = [in_size] + hl_spec
        self._lmaps = nn.ModuleList([nn.Linear(i, j) for (i, j) in zip(lspec[:1], lspec[1:])])
        self._activations = nn.ModuleList(WeightedExponentialNN() for i in lspec[1:])
        self._output_layer = nn.Linear(hl_spec[-1], 1)

    @staticmethod
    def convert2invariants(X: Tensor) -> Tensor:
        I1 = mechanics.computeI1(X).reshape(-1, 1) - 3
        I2 = mechanics.computeI2(X).reshape(-1, 1) - 3
        I3 = mechanics.computeI3(X).reshape(-1, 1) - 1
        # return torch.hstack((I1, I2, I3, I1 * I2, I2 * I3, I1 * I3))  # --> in_size = 6
        return torch.hstack((I1, I2, I3))

    def forward(self, X: torch.Tensor):
        z_prev = self.convert2invariants(X)
        for lid, (lmap, activ) in enumerate(zip(self._lmaps, self._activations)):
            # print(z_prev.shape, lmap(z_prev).shape, activ.weights.shape)
            # print(lid, activ.weight)
            z_this = activ(lmap(z_prev))
            z_prev = z_this
        output = self._output_layer(z_prev)
        return output


if __name__ == '__main__':
    def_exec(FullyConnectNN(hl_spec=[512]), NMSE_loss,
             batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
             weight_decay=WEIGHT_DECAY, seed=SEED,
             epoch_max=EPOCH_MAX, tag='FCNN', eng_loss_only=False)
