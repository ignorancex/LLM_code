# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .hifigan import Generator as HiFiGANModel

logger = logging.getLogger(__name__)


class HiFiGANVocoder(nn.Module):
    def __init__(
        self, checkpoint_path: str, model_cfg: Dict[str, str], fp16: bool = False
    ) -> None:
        super().__init__()
        self.model = HiFiGANModel(model_cfg)
        state_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(state_dict["generator"])
        if fp16:
            self.model.half()
        logger.info(f"loaded HiFiGAN checkpoint from {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B x) T x D -> (B x) 1 x T
        model = self.model.eval()
        if len(x.shape) == 2:
            return model(x.unsqueeze(0).transpose(1, 2)).detach().squeeze(0)
        else:
            return model(x.transpose(-1, -2)).detach()
