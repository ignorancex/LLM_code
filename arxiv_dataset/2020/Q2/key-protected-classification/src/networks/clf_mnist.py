import torch
import torch.nn as nn
import torch.nn.functional as F
from .clf_base import Net as _Net

class Net(_Net):

    def __init__(self, n_classes=10, d_key=-1, use_fixed_layer=False):
        super().__init__(n_classes, 256, d_key, use_fixed_layer)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2)
        )
