from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from . import initialization as init


class SegmentationModel(nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder_tp)
        init.initialize_decoder(self.decoder_np)
        init.initialize_decoder(self.decoder_hv)

        init.initialize_head(self.segmentation_head_tp)
        init.initialize_head(self.segmentation_head_np)
        init.initialize_head(self.segmentation_head_hv)

        self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", self.decoder_tp),
                        ("np", self.decoder_np),
                        ("hv", self.decoder_hv),
                    ]
                )
            )
        self.segmentation_head = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", self.segmentation_head_tp),
                        ("np", self.segmentation_head_np),
                        ("hv", self.segmentation_head_hv),
                    ]
                )
            )
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        self.train()
        features = self.encoder(x)

        out_dict = OrderedDict()
        for branch_name in self.decoder.keys():
            decoder_output = self.decoder[branch_name](*features)
            masks = self.segmentation_head[branch_name](decoder_output)
            out_dict[branch_name] = masks

        # np_predicted, hv_predicted, tp_predicted
        return out_dict["np"], out_dict["hv"], out_dict["tp"]

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
