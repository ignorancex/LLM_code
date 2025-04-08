
import torch
import torch.utils.data as data_utils
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2

from PIL import Image

import random
from  matplotlib import pyplot as plt
import torchvision.transforms.functional as TF
from typing import Sequence

class Polar_transform(object):
    def __init__(self, height=224):  # maxr = W * alpha   width = W / 2 / alpha    ->  maxr = W / 2 / width * W
    
        self.height = height
    def __call__(self, img=None):
        
        if self.height == 0:
            return img
            
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        H, W, C = img.shape
        img = cv2.linearPolar(img, (H // 2, W // 2), W * W / 2 / self.height, cv2.WARP_FILL_OUTLIERS).reshape(H, W, C)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


        return img


class RotateTransform(object):
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)