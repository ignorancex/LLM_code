"""Based on Kornia RandomErasing.
Modified to add a fill mode: white, black, random_gray, random_rgb, uniform.
This mode can be fixed for the entire training or randomly selected for each batch.
"""
import random

import kornia.augmentation as ka
import torch
from kornia.core import where
from kornia.geometry.bbox import bbox_generator, bbox_to_mask

__all__ = ['CutOut']


class CutOut(ka.RandomErasing):
    modes = ['white', 'black', 'random_gray', 'random_rgb', 'uniform']

    def __init__(self, mode=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    def _generate_values(self, input):
        mode = self.mode or random.choice(self.modes)

        if mode == 'white': return torch.ones_like(input)
        elif mode == 'black': return torch.zeros_like(input)
        elif mode == 'random_gray': return torch.rand(1, dtype=input.dtype, device=input.device).expand_as(input)
        elif mode == 'random_rgb': return torch.rand((1, 3, 1, 1), dtype=input.dtype, device=input.device).expand_as(input)
        elif mode == 'uniform': return torch.rand_like(input)

    def apply_transform(self, input, params, flags, transform=None):
        _, c, h, w = input.size()
        values = self._generate_values(input)

        bboxes = bbox_generator(params["xs"], params["ys"], params["widths"], params["heights"])
        mask = bbox_to_mask(bboxes, w, h)  # Returns B, H, W
        mask = mask.unsqueeze(1).repeat(1, c, 1, 1).to(input)  # Transform to B, c, H, W
        transformed = where(mask == 1.0, values, input)
        return transformed

    def apply_transform_mask(self, input, params, flags, transform=None):
        _, c, h, w = input.size()
        values = self._generate_values(input)

        bboxes = bbox_generator(params["xs"], params["ys"], params["widths"], params["heights"])
        mask = bbox_to_mask(bboxes, w, h)  # Returns B, H, W
        mask = mask.unsqueeze(1).repeat(1, c, 1, 1).to(input)  # Transform to B, c, H, W
        transformed = where(mask == 1.0, values, input)
        return transformed
