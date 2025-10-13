from dataclasses import dataclass
from functools import partial

import numpy as np
from torch import nn

from .base import ActivationList, Architecture, make_activation


@dataclass
class CNN(Architecture):
    activation: ActivationList = "gelu"
    width: int = 32

    def create(self, input_shape, output_dim):
        conv3x3 = partial(nn.Conv2d, kernel_size=3, padding="same", bias=False)
        model = nn.Sequential(
            conv3x3(input_shape[0], self.width),
            make_activation(self.activation),
            nn.AvgPool2d(2),
            conv3x3(self.width, 2 * self.width),
            make_activation(self.activation),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(
                np.prod(input_shape[1:]) * self.width // 8, 4 * self.width, bias=False
            ),
            make_activation(self.activation),
            nn.Linear(4 * self.width, output_dim),
        )

        return model
