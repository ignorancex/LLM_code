import math

import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.functional import interpolate, relu

nc = 3


class DeepDecoder(nn.Module):
    def __init__(self,
                 num_filters,
                 img_size=64,
                 output_activ="sigmoid",
                 depth=5,
                 ups_first=True):
        super(DeepDecoder, self).__init__()

        conv = nn.ModuleList([
            torch.nn.Conv2d(num_filters, num_filters, (1, 1))
            for _ in range(depth - 1)
        ] + [torch.nn.Conv2d(num_filters, 3, (1, 1))])
        bn = nn.ModuleList(
            [torch.nn.BatchNorm2d(num_filters) for _ in range(depth - 1)])

        self.conv = conv
        self.bn = bn
        self.output_activ = output_activ
        self.ups_first = ups_first
        self.depth = depth
        self.img_size = img_size

    def forward(self, inp):
        bn = self.bn
        conv = self.conv

        out_stack = inp
        seed_pow = int(
            math.log2(self.img_size)
        ) - self.depth + 1  # Images are 2**6. Must have (depth - 1) upsamples to get from seed to 2

        for layer_idx in range(len(conv) - 1):
            cur_activ = bn[layer_idx](conv[layer_idx](out_stack))
            imsize = 2**(seed_pow + (layer_idx + 1)
                         )  # seed power + num upsamples AFTER this layer

            if (self.ups_first):
                cur_activ = interpolate(cur_activ,
                                        size=(imsize, imsize),
                                        mode='bilinear',
                                        align_corners=True)
                out_stack = relu(cur_activ)
            else:
                cur_activ = relu(cur_activ)
                out_stack = interpolate(cur_activ,
                                        size=(imsize, imsize),
                                        mode='bilinear',
                                        align_corners=True)

        if (self.output_activ == "sigmoid"):
            out = sigmoid(conv[-1](out_stack))
        elif (self.output_activ == "none"):
            out = conv[-1](out_stack)
        else:
            raise Exception(
                "Must specify either sigmoid or none as output activation type."
            )
        return out
