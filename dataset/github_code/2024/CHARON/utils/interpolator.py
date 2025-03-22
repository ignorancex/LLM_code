from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolator(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.window_size = 120
        self.device = device
        self.sampling_interval = args.sampling_interval

    @torch.no_grad()
    def forward(self, sample, indexes):
        return F.interpolate(sample,
                             size=(self.window_size, sample.shape[3], sample.shape[4]),
                             mode='trilinear',
                             align_corners=False)

    def get_sampling_frequency(self):
        return 1 / self.sampling_interval

    def get_efficiency_ratio(self):
        return 1 - self.get_sampling_frequency()

    def get_efficient_buffer_sample(self, not_aug_inputs: torch.Tensor) -> Tuple[torch.Tensor, None]:

        not_aug_inputs = not_aug_inputs[:, :, ::self.sampling_interval, :, :]

        return not_aug_inputs, torch.arange(0, not_aug_inputs.shape[2], self.sampling_interval).unsqueeze(0).repeat(not_aug_inputs.shape[0], 1)
