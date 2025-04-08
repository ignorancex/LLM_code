
import torch
import torch.utils.data as data_utils
import numpy as np

import random
from  matplotlib import pyplot as plt
import torchvision.transforms.functional as TF
from typing import Sequence


class RotateTransform(object):
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)